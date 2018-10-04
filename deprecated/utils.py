import tensorflow as tf
import numpy.random as nr
import numpy as np
import cv2
import gym
from scipy.signal import lfilter


def boltzmann(probs, actions):
    return np.random.choice(actions, p=probs.numpy()[0])


def greedy(probs, actions, epsilon=0.1):
    if nr.rand() < epsilon:
        return nr.randint(0, actions)
    else:
        return np.argmax(probs)


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def gae(rews, vals, bval, tau=0.99):
    vboot = np.hstack((vals, bval))
    return discount(rews +  tau * vboot[1:] - vals, tau)


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
        self._skip       = skip

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

    def __init__(self, env, h, w):
        gym.Wrapper.__init__(self, env)
        self.h = h
        self.w = w

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._preprocess(obs, self.h, self.w)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return self._preprocess(obs, self.h, self.w)

    def _preprocess(self, obs, height, width):
        x = obs[34:34 + 160, :160]
        x = cv2.resize(x, (80, 80))
        x = cv2.resize(x, (height, width))
        x = x.mean(2)
        x = x.astype(np.float32)
        x *= (1.0 / 255.0)
        x = np.reshape(x, [height, width, 1])
        return x


def make_atari(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=15)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessObsWrapper(env, 42, 42)
    return env