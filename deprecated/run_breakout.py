import multiprocessing as mp
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from deprecated.policies import ActorCritic
from deprecated.a3c.model import Worker, is_running
from deprecated.config.run_configuration import Config
from deprecated.utils import make_atari


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    config = Config("config/config.json")

    env = make_atari(config.env)
    model = ActorCritic(env.action_space.n, policy=config.policy, device=config.device)
    model.forward([env.reset()], model.policy.s0)   # Initialize weights of general model.
    optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr)
    step = tf.train.get_or_create_global_step()

    saver = tfe.Checkpoint(optimizer=optimizer, model=model.policy, optimizer_step=step)
    saver.restore(tf.train.latest_checkpoint(config.save_dir))
    queue = mp.Queue(maxsize=10000)

    workers = [Worker(queue, config) for _ in range(config.processes)]

    for p in workers:
        p.daemon = True
        p.start()

    while not queue.empty() or is_running(workers):
        grads = queue.get()
        for i in range(len(grads)):
            grads[i] = tf.constant(grads[i], dtype=tf.float32)

        optimizer.apply_gradients(zip(grads, model.policy.trainable_weights), global_step=step)
        saver.save(file_prefix=config.file_prefix)

    for p in workers:
        p.join()
