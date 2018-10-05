import os
import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from a2c_eager.utils import make_atari
from a2c_eager.policies import ActorCritic
from a2c_eager.config.run_configuration import Config
from a2c_eager.utils import boltzmann, greedy, discount, gae, Logger


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
tf.enable_eager_execution(config=tf_config)


def generate_gradients(timestep):
    loader.restore(tf.train.latest_checkpoint(config.save_dir))
    observations, actions, rewards, values = [], [], [], []
    ob = env.reset()
    done = False

    while not done:
        logits, v = model.forward([ob])
        probs = tf.nn.softmax(logits)
        a = boltzmann(probs)
        next_ob, r, done, _ = env.step(a)
        observations.append(ob)
        actions.append(a)
        rewards.append(r)
        values.append(v)
        ob = next_ob

    model.policy.reset()
    advantages = gae(rewards, values, 0, gamma=config.gamma, tau=config.tau)
    discounted_rewards = discount(rewards, config.gamma)
    gradients, log_values = model.gradient(observations, advantages, discounted_rewards, actions)
    model.policy.reset()
    logger.log_performance(rewards, *log_values)
    # Convert EagerTensors to numpy arrays for IPC.
    return [gradient.numpy() for gradient in gradients]


def initialize_process():
    global env, model, loader, config, logger
    config = Config("config/config.json")
    env = make_atari(config.env)
    # env = gym.make(config.env)
    model = ActorCritic(env.action_space.n, policy=config.policy)
    loader = tfe.Checkpoint(model=model.policy, optimizer_step=tf.train.get_or_create_global_step())
    logger = Logger("Worker_{}".format(os.getpid()))


def train():
    mp.set_start_method('spawn', force=True)

    config = Config("config/config.json")
    env = make_atari(config.env)
    # env = gym.make(config.env)
    step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    model = ActorCritic(env.action_space.n, policy=config.policy)
    saver = tfe.Checkpoint(optimizer=optimizer, model=model.policy, optimizer_step=step)
    pool = mp.Pool(processes=config.processes, initializer=initialize_process)
    saver.restore(tf.train.latest_checkpoint(config.save_dir))
    logger = Logger('global')

    #   Initialize model.
    model.forward([env.reset()])
    ts = time.time()

    for t in range(config.steps):
        gradients = []
        roll = pool.map(generate_gradients, [t] * config.processes)

        for tup in zip(*roll):
            averaged = np.mean(tup, axis=0)
            gradients.append(tf.constant(averaged, dtype=tf.float32))

        clipped, _ = tf.clip_by_global_norm(gradients, config.max_norm)
        gnorms = [tf.norm(grad) for grad in clipped]
        logger.log_gradients(gnorms)
        logger.log_weights(model.policy.trainable_variables)
        optimizer.apply_gradients(zip(clipped, model.policy.trainable_weights), global_step=step)
        saver.save(file_prefix=config.file_prefix)

        print("Epoch took: {}".format(time.time() - ts))
        ts = time.time()


def test():
    config = Config("config/config.json")
    env = make_atari(config.env)
    model = ActorCritic(env.action_space.n, policy=config.policy)
    saver = tfe.Checkpoint(model=model.policy)
    saver.restore(tf.train.latest_checkpoint(config.save_dir))
    ob = env.reset()
    s = model.policy.s0

    while True:
        env.render()
        logits, _, s = model.forward([ob], s)
        probs = tf.nn.softmax(logits)
        a = greedy(probs)
        ob, _, done, _ = env.step(a)

        if done:
            ob = env.reset()


if __name__ == '__main__':
    train()


