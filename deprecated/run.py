import numpy as np
import multiprocessing as mp
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from deprecated.policies import CRPolicy
from deprecated.utils import discount, boltzmann, make_atari


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)

CKPT_DIR = 'ckpt'
CKPT_PREFIX = os.path.join(CKPT_DIR, "ckpt")

N_PROC = 4  # Number of processes producing rollouts in parallel.
LR = 7e-4
GAMMA = 0.99
SAMPLES = 400000
MAX_GRAD_NORM = 0.5
ENV_NAME = 'Breakout-v0'
PROC_T = SAMPLES // N_PROC + 1


def sample(queue, env_name, steps):
    env = make_atari(env_name)
    model = CRPolicy(env.action_space.n)
    loader = tfe.Checkpoint(model=model)

    for roll in range(steps):
        # TODO: REMOVE THIS CRAP WHEN YOU FIX IT
        try:
            loader.restore(tf.train.latest_checkpoint(CKPT_DIR))
        except:
            continue

        obs, act, rews = [], [], []
        ob = env.reset()
        done = False
        s = model.s0

        while not done:
            logits, v, s = model([ob], s)
            probs = tf.nn.softmax(logits)
            a = boltzmann(probs, env.action_space.n)
            next_ob, r, done, _ = env.step(a)
            obs.append(ob)
            act.append(a)
            rews.append(r)
            ob = next_ob

        d_rews = discount(rews, GAMMA)
        d_rew = (d_rews - np.mean(d_rews)) / (np.std(d_rews) + 1e-6)

        with tf.GradientTape() as tape:
            logits, values, _ = model(obs, model.s0)
            values = tf.squeeze(values)
            advs = tf.constant(d_rew, dtype=tf.float32) - values
            policy = tf.nn.softmax(logits)

            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=act, logits=logits)
            p_loss = xentropy * tf.stop_gradient(advs)
            v_loss = tf.square(advs)
            e_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

            loss = tf.reduce_mean(p_loss + 0.5 * v_loss - 0.01 * e_loss)

        grads = tape.gradient(loss, model.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

        print("Step: {0}, Len: {1} BR: {2}, TL: {3:.4f}".format(roll, len(obs), np.sum(rews), loss))

        for i in range(len(grads)):
            grads[i] = grads[i].numpy()

        queue.put(grads)


if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    # env = make_atari(ENV_NAME)
    env = make_atari(ENV_NAME)
    model = CRPolicy(env.action_space.n)

    # Initialize weights of general model.
    model([env.reset()], model.s0)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LR)
    step = tf.train.get_or_create_global_step()

    saver = tfe.Checkpoint(optimizer=optimizer, model=model, optimizer_step=step)
    saver.restore(tf.train.latest_checkpoint(CKPT_DIR))
    roll_queue = mp.Queue(maxsize=10000)

    args = (roll_queue, ENV_NAME, PROC_T)
    procs = [mp.Process(target=sample, args=args) for _ in range(N_PROC)]

    for p in procs:
        p.daemon = True
        p.start()

    for t in range(1, SAMPLES + 1):
        grads = roll_queue.get()

        for i in range(len(grads)):
            grads[i] = tf.constant(grads[i], dtype=tf.float32)

        optimizer.apply_gradients(zip(grads, model.trainable_weights), global_step=step)
        saver.save(file_prefix=CKPT_PREFIX)

    for p in procs:
        p.join()
