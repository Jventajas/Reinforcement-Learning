import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from multiprocessing import Process
from deprecated.policies import ActorCritic
from deprecated.utils import discount, make_atari, greedy


def is_running(processes):
    return all([p.is_alive() for p in processes])


class Worker(Process):

    def __init__(self, queue, config):
        super().__init__()
        self.queue = queue
        self.steps = config.steps
        self.batch_size = config.batch_size
        self.daemon = True
        self.config = config

    def run(self):
        T = 0  # Worker step counter.
        env = make_atari(self.config.env)
        model = ActorCritic(env.action_space.n, policy=self.config.policy, device=self.config.device)
        loader = tfe.Checkpoint(model=model.policy)
        obs, act, rew = [], [], []
        ob = env.reset()
        done = False
        s = model.policy.s0
        cum_reward = 0
        ep_len = 0

        while T < self.steps:
            try:
                loader.restore(tf.train.latest_checkpoint(self.config.save_dir))
            except:
                continue
            t = 0  # Batch counter.
            s_init = s
            epsilon = 0.6 - (T / self.steps) * 0.5

            while not done and t < self.batch_size:
                logits, v, s = model.forward([ob], s)
                probs = tf.nn.softmax(logits)
                a = greedy(probs, env.action_space.n, epsilon=epsilon)
                next_ob, r, done, _ = env.step(a)
                obs.append(ob)
                act.append(a)
                rew.append(r)
                ob = next_ob
                t += 1
                T += 1
                cum_reward += r
                ep_len += 1

            d_rew = discount(rew, self.config.gamma)
            d_rew = (d_rew - np.mean(d_rew)) / (np.std(d_rew) + 1e-6)  # Stability constant.
            grads, loss = model.gradient(obs, d_rew, act, s_init)
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_norm)

            if done:
                print(f"Step: {T}, Len: {ep_len}, BR: {cum_reward}, TL: {loss:.4f}, Epsilon: {epsilon:.2f}")
                s = model.policy.s0
                done = False
                ob = env.reset()
                cum_reward = 0
                ep_len = 0

            obs.clear()
            act.clear()
            rew.clear()

            for i in range(len(grads)):
                grads[i] = grads[i].numpy()

            self.queue.put(grads)
