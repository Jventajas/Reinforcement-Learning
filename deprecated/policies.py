import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.initializers import orthogonal
from tensorflow.python.ops.rnn import dynamic_rnn

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, MultiRNNCell


class FFPolicy(tf.keras.Model):

    def __init__(self, n_actions):
        super(FFPolicy, self).__init__()
        self.s0 = BasicLSTMCell(128).zero_state(1, dtype=tf.float32)
        self.fc1 = Dense(100, activation='relu')
        self.fc2 = Dense(100, activation='relu')
        self.pol = Dense(n_actions)
        self.val = Dense(1)

    def call(self, obs, state):
        x = tf.constant(obs, dtype=tf.float32)
        pi = self.fc1(x)
        v = self.fc2(x)
        return self.pol(pi), self.val(v), state


class RPolicy(tf.keras.Model):

    def __init__(self, n_actions):
        super(RPolicy, self).__init__()
        cells = [GRUCell(128, kernel_initializer=orthogonal(np.sqrt(2))) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.s0 = self.gru.zero_state(batch_size=1, dtype=tf.float32)
        self.fc1 = Dense(128, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.fc2 = Dense(100, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.fc3 = Dense(100, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.pol = Dense(n_actions, kernel_initializer=orthogonal(0.01))
        self.val = Dense(1, kernel_initializer=orthogonal(np.sqrt(2)))

    def call(self, obs, state):
        x = tf.constant(obs, dtype=tf.float32)
        x = self.fc1(x)
        x = tf.expand_dims(x, axis=0)
        x, state = dynamic_rnn(self.gru, x, initial_state=state)
        x = tf.reshape(x, shape=[-1, 128])
        pi = self.fc2(x)
        v = self.fc3(x)
        return self.pol(pi), self.val(v), state


class CRPolicy(tf.keras.Model):

    def __init__(self, n_actions):
        super(CRPolicy, self).__init__()
        cells = [GRUCell(128, kernel_initializer=orthogonal(np.sqrt(2))) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.s0 = self.gru.zero_state(batch_size=1, dtype=tf.float32)
        self.cv1 = Conv2D(32, 3, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.mp1 = MaxPool2D()
        self.cv2 = Conv2D(32, 3, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.mp2 = MaxPool2D()
        self.cv3 = Conv2D(32, 3, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.mp3 = MaxPool2D()
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.fc2 = Dense(100, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.fc3 = Dense(100, activation='relu', kernel_initializer=orthogonal(np.sqrt(2)))
        self.pol = Dense(n_actions, kernel_initializer=orthogonal(0.01))
        self.val = Dense(1, kernel_initializer=orthogonal(1))

    def call(self, obs, state):
        x = tf.constant(obs, dtype=tf.float32)
        x = self.cv1(x)
        x = self.mp1(x)
        x = self.cv2(x)
        x = self.mp2(x)
        x = self.cv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.expand_dims(x, axis=0)
        x, state = dynamic_rnn(self.gru, x, initial_state=state)
        x = tf.reshape(x, shape=[-1, 128])
        pi = self.fc2(x)
        v = self.fc3(x)
        return self.pol(pi), self.val(v), state


class ActorCritic:

    def __init__(self, n_actions, policy, device='/cpu:0'):
        self.device = device
        with tf.device(self.device):
            if policy == 'CRPolicy':
                self.policy = CRPolicy(n_actions)
            elif policy == 'RPolicy':
                self.policy = RPolicy(n_actions)
            elif policy == 'FFPolicy':
                self.policy = FFPolicy(n_actions)
            else:
                raise ValueError(f'Invalid policy type: {policy}')

    def forward(self, obs, state):
        with tf.device(self.device):
            return self.policy(obs, state)

    def gradient(self, obs, rewards, actions, state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                logits, values, _ = self.policy(obs, state)
                values = tf.squeeze(values)
                policy = tf.nn.softmax(logits)
                advantages = tf.constant(rewards, dtype=tf.float32) - values
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
                p_loss = xentropy * tf.stop_gradient(advantages)
                v_loss = tf.square(advantages)
                e_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
                loss = tf.reduce_mean(p_loss + 0.5 * v_loss - 0.01 * e_loss)
        grads = tape.gradient(loss, self.policy.trainable_weights)
        return grads, loss
