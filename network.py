# I don't usually use keras but it seems simpler than doing things directly,
# and I'm not trying to do anything unusual like batch a whole population of
# networks, so I'll take that route for this assignment.

import numpy as np
import tensorflow as tf

def FFANN_factory(hidden_layer_sizes):
    def network_factory(obs_input):
        next_input = obs_input
        for hidden_layer, hidden_layer_size in enumerate(hidden_layer_sizes):
            next_input = tf.keras.layers.Dense(hidden_layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal')(next_input)
        return next_input

    return network_factory

class Network:
    def __init__(self, obs_shape, network_factory, learning_rate, is_policy, action_shape=None, action_modes=None):
        self.obs_shape = obs_shape
        self.network_factory = network_factory
        self.learning_rate = learning_rate
        self.is_policy = is_policy
        self.action_shape = action_shape
        self.action_modes = action_modes

        obs_input = tf.keras.layers.Input(shape=obs_shape)

        last_layer = network_factory(obs_input)

        if is_policy:
            output_shape = action_shape + (action_modes, 3)
            output_size = np.prod(output_shape)
        else:
            output_shape = (1,)
            output_size = 1

        flat_linear_output = tf.keras.layers.Dense(output_size)(last_layer)
        linear_output = tf.keras.layers.Reshape(output_shape)(flat_linear_output)

        self.keras_network = tf.keras.Model(obs_input, linear_output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def apply(self, S):
        return self.keras_network(S)

    @tf.function
    def fit(self, S, A, Q):
        Q_predicted = self.apply(S, A)
        # tf.print(S[0], A[0], Q[0], Q_predicted[0])
        Q_loss = tf.reduce_sum((Q_predicted - Q) ** 2)

        Q_gradient = tf.gradients(Q_loss, self.keras_network.weights)
        self.optimizer.apply_gradients(zip(Q_gradient, self.keras_network.weights))

        return Q_loss

    def copy_from(self, other, amount):
        for self_w, other_w in zip(self.keras_network.weights, other.keras_network.weights):
            self_w.assign(self_w*(1-amount) + other_w*amount)

    def copy(self):
        other = Network(self.obs_shape, self.network_factory, self.learning_rate, self.is_policy, self.action_shape, self.action_modes)
        other.copy_from(self, 1)
        return other

    def zero_like(self):
        other = Network(self.obs_shape, self.network_factory, self.learning_rate, self.is_policy, self.action_shape, self.action_modes)
        for other_w in other.keras_network.weights:
            other_w.assign(tf.zeros_like(other_w))
        return other

    def zero_self(self):
        for w in self.keras_network.weights:
            w.assign(tf.zeros_like(w))
