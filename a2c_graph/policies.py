import tensorflow as tf
import numpy as np
from tensorflow.initializers import orthogonal
from tensorflow.nn import elu
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.layers import conv2d, max_pooling2d, dense, flatten
from tensorflow.contrib.rnn import MultiRNNCell, GRUCell
from a2c_graph.utils import boltzmann


class FFPolicy:

    def __init__(self, nact):
        self.obs_ph = tf.placeholder(dtype=tf.float32)
        self.logits = dense(self.obs_ph, nact, kernel_initializer=orthogonal(0.01), name='logits')
        self.pi = tf.nn.softmax(self.logits, name='pi')
        self.action = boltzmann(self.pi)
        self.value = dense(self.logits, 1, kernel_initializer=orthogonal(), name='value')

    def forward(self, obs, sess):
        feed_dict = {self.obs_ph: obs}
        action, value = sess.run([self.logits, self.value, self.pi], feed_dict=feed_dict)
        return action, value

    def reset(self):
        pass


class RPolicy:

    def __init__(self, nact, rnn_units=256):
        cells = [GRUCell(rnn_units, kernel_initializer=orthogonal()) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)

        self.obs_ph = tf.placeholder(dtype=tf.float32)
        fc1 = dense(self.obs_ph, rnn_units, activation=elu, kernel_initializer=orthogonal(), name='fc1')
        expand = tf.expand_dims(fc1, axis=0, name='expand')
        rnn_out, self.state = dynamic_rnn(self.gru, expand, initial_state=self.state)
        reshape = tf.reshape(rnn_out, shape=[-1, rnn_units], name='reshape')

        self.logits = dense(reshape, nact, kernel_initializer=orthogonal(0.01), name='logits')
        self.pi = tf.nn.softmax(self.logits, name='pi')
        self.action = boltzmann(self.pi)
        self.value = dense(self.logits, 1, kernel_initializer=orthogonal(), name='value')

    def forward(self, obs, sess):
        feed_dict = {self.obs_ph: obs}
        action, value = sess.run([self.action, self.pi], feed_dict=feed_dict)
        return action, value

    def reset(self):
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)


class CRPolicy:

    def __init__(self, h, w, c, nact, rnn_units=256, cnn_units=32):
        cells = [GRUCell(rnn_units, kernel_initializer=orthogonal()) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, h, w, c])
        cv1 = conv2d(self.obs_ph, cnn_units, 3, strides=2, activation=elu, kernel_initializer=orthogonal(), name='cv1')
        cv2 = conv2d(cv1, cnn_units, 3, strides=2, activation=elu, kernel_initializer=orthogonal(), name='cv2')
        cv3 = conv2d(cv2, cnn_units, 3, strides=2, activation=elu, kernel_initializer=orthogonal(), name='cv3')
        cv4 = conv2d(cv3, cnn_units, 3, strides=2, activation=elu, kernel_initializer=orthogonal(), name='cv4')
        flat = flatten(cv4, name='flatten')
        fc1 = dense(flat, rnn_units, activation=elu, kernel_initializer=orthogonal(), name='fc1')
        expand = tf.expand_dims(fc1, axis=0, name='expand')
        rnn_out, self.state = dynamic_rnn(self.gru, expand, initial_state=self.state)
        reshape = tf.reshape(rnn_out, shape=[-1, rnn_units], name='reshape')

        self.logits = dense(reshape, nact, kernel_initializer=orthogonal(0.01), name='logits')
        self.pi = tf.nn.softmax(self.logits, name='pi')
        self.action = boltzmann(self.pi)
        value = dense(reshape, 1, kernel_initializer=orthogonal(), name='value')
        self.value = tf.squeeze(value)

    def forward(self, obs, sess):
        feed_dict = {self.obs_ph: obs}
        action, value = sess.run([self.action, self.value], feed_dict=feed_dict)
        return action, value

    def reset(self):
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)


class ActorCritic:

    def __init__(self, nact, sess, optimizer, step, policy, max_norm=0.5, size=80, vl_coef=0.5, e_coef=0.01):
        if policy == 'CRPolicy':
            self.policy = CRPolicy(size, size, 1, nact)
        elif policy == 'RPolicy':
            self.policy = RPolicy(nact)
        elif policy == 'FFPolicy':
            self.policy = FFPolicy(nact)
        else:
            raise ValueError(f'Invalid policy type: {policy}')

        advantages_ph = tf.placeholder(dtype=tf.float32)
        rewards_ph = tf.placeholder(dtype=tf.float32)
        actions_ph = tf.placeholder(dtype=tf.int32)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_ph, logits=self.policy.logits)
        p_loss = xentropy * advantages_ph
        v_loss = tf.square(rewards_ph - self.policy.value)
        e_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy.pi, logits=self.policy.logits)
        t_loss = tf.reduce_mean(p_loss + vl_coef * v_loss - e_coef * e_loss)
        grads = tf.gradients(t_loss, tf.trainable_variables())

        grads_ph = [tf.placeholder(dtype=tf.float32, shape=g.shape) for g in grads]
        clipped, _ = tf.clip_by_global_norm(grads_ph, max_norm)
        grad_norms = [tf.norm(grad) for grad in clipped]
        apply_grads = optimizer.apply_gradients(
            zip(clipped, tf.trainable_variables()), global_step=step)

        sess.run(tf.global_variables_initializer())

        def forward(obs):
            return self.policy.forward(obs, sess)

        def gradient(obs, advantages, rewards, actions):
            feed_dict = {
                self.policy.obs_ph: obs,
                advantages_ph: advantages,
                rewards_ph: rewards,
                actions_ph: actions
            }
            tl, pl, vl, ent, g = sess.run(
                [t_loss, p_loss, v_loss, e_loss, grads], feed_dict=feed_dict)
            return g, (tl, pl, vl, ent)

        def update(grads):
            feed_dict = {ph: val for ph, val in zip(grads_ph, grads)}
            _, norms = sess.run([apply_grads, grad_norms], feed_dict=feed_dict)
            return norms

        self.forward = forward
        self.gradient = gradient
        self.update = update
