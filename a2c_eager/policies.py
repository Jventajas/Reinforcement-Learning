import tensorflow as tf
import numpy as np
from tensorflow.initializers import orthogonal
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.python.ops.rnn import dynamic_rnn

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, MultiRNNCell


class FFPolicy(tf.keras.Model):

    def __init__(self, n_actions):
        super(FFPolicy, self).__init__()
        self.pol = Dense(n_actions)
        self.val = Dense(1)

    def call(self, obs):
        x = tf.constant(obs, dtype=tf.float32)
        return self.pol(x), self.val(x)

    def reset(self):
        pass


class RPolicy(tf.keras.Model):

    def __init__(self, n_actions, rnn_units=256):
        super(RPolicy, self).__init__()
        cells = [GRUCell(rnn_units, kernel_initializer=orthogonal()) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)
        self.fc1 = Dense(rnn_units, activation='relu')
        self.pol = Dense(n_actions)
        self.val = Dense(1)

    def call(self, obs):
        x = tf.constant(obs, dtype=tf.float32)
        x = self.fc1(x)
        x = tf.expand_dims(x, axis=0)
        x, self.state = dynamic_rnn(self.gru, x, initial_state=self.state)
        x = tf.reshape(x, shape=[-1, 256])
        return self.pol(x), self.val(x)

    def reset(self):
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)


class CRPolicy(tf.keras.Model):

    def __init__(self, n_actions, rnn_units=256, conv_units=32):
        super(CRPolicy, self).__init__()
        cells = [GRUCell(rnn_units, kernel_initializer=orthogonal()) for _ in range(2)]
        self.gru = MultiRNNCell(cells)
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)
        self.cv1 = Conv2D(conv_units, 3, strides=2, activation='elu', kernel_initializer=orthogonal())
        self.cv2 = Conv2D(conv_units, 3, strides=2, activation='elu', kernel_initializer=orthogonal())
        self.cv3 = Conv2D(conv_units, 3, strides=2, activation='elu', kernel_initializer=orthogonal())
        self.cv4 = Conv2D(conv_units, 3, strides=2, activation='elu', kernel_initializer=orthogonal())
        self.flatten = Flatten()
        self.fc1 = Dense(rnn_units, kernel_initializer=orthogonal())
        self.pol = Dense(n_actions, kernel_initializer=orthogonal(0.01))
        self.val = Dense(1, kernel_initializer=orthogonal())

    def call(self, obs):
        x = tf.constant(obs, dtype=tf.float32)
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.expand_dims(x, axis=0)
        x, self.state = dynamic_rnn(self.gru, x, initial_state=self.state)
        x = tf.reshape(x, shape=[-1, 256])
        return self.pol(x), self.val(x)

    def reset(self):
        self.state = self.gru.zero_state(batch_size=1, dtype=tf.float32)


class ActorCritic:

    def __init__(self, n_actions, policy, vl_coef=0.5, e_coef=0.01):
        if policy == 'CRPolicy':
            self.policy = CRPolicy(n_actions)
        elif policy == 'RPolicy':
            self.policy = RPolicy(n_actions)
        elif policy == 'FFPolicy':
            self.policy = FFPolicy(n_actions)
        else:
            raise ValueError(f'Invalid policy type: {policy}')

        def forward(obs):
            return self.policy(obs)

        def gradient(observations, advantages, rewards, actions):
            with tf.GradientTape() as tape:
                logits, values = self.policy(observations)
                values = tf.squeeze(values)
                policy = tf.nn.softmax(logits)
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
                ploss = xentropy * tf.constant(advantages, dtype=tf.float32)
                vloss = tf.square(tf.constant(rewards, dtype=tf.float32) - values)
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
                tloss = tf.reduce_mean(ploss + vl_coef * vloss - e_coef * entropy)
            grads = tape.gradient(tloss, self.policy.trainable_weights)
            return grads, (tloss, ploss, vloss, entropy)

        self.forward = forward
        self.gradient = gradient
