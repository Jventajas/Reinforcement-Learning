import os
import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from a2c_graph.utils import make_atari
from a2c_graph.policies import ActorCritic
from a2c_graph.config.run_configuration import Config
from a2c_graph.utils import greedy, discount, gae, Logger

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_rollout(timestep):
    if os.path.exists(config.save_dir):
        loader.restore(sess, tf.train.latest_checkpoint(config.save_dir))

    observations, actions, rewards, values = [], [], [], []
    observation = env.reset()
    done = False
    while not done:
        action, value = model.forward([observation])
        next_observation, reward, done, _ = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        observation = next_observation

    model.policy.reset()
    advantages = gae(rewards, values, 0, gamma=config.gamma, tau=config.tau)
    discounted_rewards = discount(rewards, config.gamma)
    gradients, log_values = model.gradient(observations, advantages, discounted_rewards, actions)

    if timestep % 10 == 0:
        logger.log_performance(rewards, *log_values)

    model.policy.reset()
    return gradients


def process_initializer():
    global env, model, loader, config, sess, logger
    sess = tf.Session()
    config = Config("config/config.json")
    env = make_atari(config.env)
    # env = gym.make(config.env)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    step = tf.train.get_or_create_global_step()
    model = ActorCritic(env.action_space.n, sess, optimizer, step, policy=config.policy)
    logger = Logger(sess, f"Worker_{os.getpid()}")
    loader = tf.train.Saver(tf.trainable_variables() + [step])
    sess.as_default()


def train():
    mp.set_start_method('spawn', force=True)
    config = Config("config/config.json")
    env = make_atari(config.env)
    # env = gym.make(config.env)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    step = tf.train.get_or_create_global_step()
    model = ActorCritic(env.action_space.n, sess, optimizer, step, policy=config.policy)
    saver = tf.train.Saver(tf.trainable_variables() + [step], max_to_keep=1)
    logger = Logger(sess, 'global')
    pool = mp.Pool(processes=config.processes, initializer=process_initializer)

    if os.path.exists(config.save_dir):
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
    dur = time.time()

    for t in range(config.steps):
        roll = pool.map(generate_rollout, [t] * config.processes)
        gradients = [np.mean(tup, axis=0) for tup in zip(*roll)]
        gnorms = model.update(gradients)

        if t % 10 == 0:
            logger.log_gradients(gnorms)
            logger.log_weights()

        saver.save(sess, save_path=config.file_prefix,
                          global_step=step, write_meta_graph=False)
        print(f"Epoch took: {time.time() - dur:.2f}")
        dur = time.time()


if __name__ == '__main__':

    if config.use_tpu:
        tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()
        tpu_computation = tpu.rewrite(train, tpu_grpc_url)

        with tf.Session(tpu_grpc_url) as sess:
            sess.run(tpu.initialize_system())
            sess.run(tpu_computation)
            sess.run(tpu.shutdown_system())
    else:
        train()


