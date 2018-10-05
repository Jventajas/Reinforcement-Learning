import os
import tensorflow as tf
import numpy as np
import cv2
import gym
from scipy.signal import lfilter
from tensorflow.contrib import summary


def boltzmann(probs, epsilon=0.):
    random = tf.random_uniform(shape=(), minval=0, maxval=1)
    action = tf.cond(random > epsilon,
                     lambda: tf.multinomial(tf.log(probs), 1),
                     lambda: tf.multinomial(
                         tf.log(tf.ones_like(probs)), 1)
                     )
    return tf.squeeze(action)


def greedy(probs, epsilon=0.):
    random = tf.random_uniform(shape=(), minval=0, maxval=1)
    action = tf.cond(random > epsilon,
                     lambda: tf.argmax(probs, axis=1),
                     lambda: tf.multinomial(
                         tf.log(tf.ones_like(probs)), 1)
                     )
    return tf.squeeze(action)


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def gae(rews, vals, bval, gamma=0.99, tau=0.97):
    vboot = np.hstack((vals, bval))
    return discount(rews +  gamma * vboot[1:] - vals, tau * gamma)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class PreprocessObsWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._preprocess(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return self._preprocess(obs)

    def _preprocess(self, obs):
        x = obs[34:34 + 160, :160]
        x = x.mean(2)
        x = cv2.resize(x, (80, 80))
        x = x.astype(np.float32)
        x *= (1.0 / 255.0)
        x = np.reshape(x, [80, 80, 1])
        return x


def make_atari(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=15)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessObsWrapper(env)
    return env


class Logger:

    def __init__(self, name, logdir='logs'):
        dir = os.path.join(logdir, name)
        writer = summary.create_file_writer(dir, flush_millis=10000)

        def log_performance(rewards, tloss, ploss, vloss, entropy):
            with writer.as_default(), summary.record_summaries_every_n_global_steps(10):
                summary.scalar('Perf/Total Reward', tf.reduce_sum(rewards))
                summary.scalar('Perf/Episode Duration', tf.size(rewards))
                summary.scalar('Perf/Total Loss', tloss)
                summary.scalar('Perf/Policy Loss', tf.reduce_mean(ploss))
                summary.scalar('Perf/Value Loss', tf.reduce_mean(vloss))
                summary.scalar('Perf/Policy Entropy', tf.reduce_mean(entropy))


        def log_gradients(gnorms):
            with writer.as_default(), summary.record_summaries_every_n_global_steps(10):
                summary.histogram('Gradient Norms', gnorms)

        def log_weights(var_list):
            for var in var_list:
                with writer.as_default(), summary.record_summaries_every_n_global_steps(10):
                    summary.histogram(var.name, var)

        self.log_performance = log_performance
        self.log_gradients = log_gradients
        self.log_weights = log_weights
