import numpy as np
import scipy.special
import tensorflow as tf

import experience_store

debug_plot_values = False
debug_print_action_calc = False

class RandomAgent:
    def __init__(self, rng, actor_count, action_shape, stoch_persistence):
        self.rng = rng

        self.actor_count = actor_count

        self.stoch_persistence = stoch_persistence

        self.action_shape = action_shape

        self.cur_action = self.rng.normal(size=(self.actor_count,) + self.action_shape)

    def act(self, obs, actor=0, temp=1):
        if self.rng.random() > self.stoch_persistence:
            self.cur_action[actor] = self.rng.normal(size=self.action_shape)

        return self.cur_action[actor]

    def store(self, actor, obs, action, reward, terminal):
        pass

    def train(self, epochs=1):
        return np.array(0.), np.array(0.)

    def get_weights(self):
        return None

class AdvantageAgent:
    def __init__(self, rng, actor_count, policy_network, value_network, t_max, discount_factor, stoch_persistence, entropy_weight):
        self.rng = rng

        self.actor_count = actor_count

        self.policy_network = policy_network
        self.value_network = value_network

        self.t_max = t_max
        self.discount_factor = discount_factor
        self.stoch_persistence = stoch_persistence

        self.beta_entropy_weight = entropy_weight
        self.minibatch_size = 128
        self.use_tqdm = False

        self.obs_shape = self.policy_network.obs_shape
        self.action_shape = self.policy_network.action_shape
        self.action_modes = self.policy_network.action_modes

        self.stoch_mode_t = self.rng.random((self.actor_count,) + self.action_shape)
        self.stoch_gauss_t = self.rng.random((self.actor_count,) + self.action_shape)
        # this is a prominent term in the Gaussian quantile formula, precomputed for efficiency
        self.stoch_gauss_inverf = np.sqrt(2) * scipy.special.erfinv(2 * self.stoch_gauss_t - 1)
        # this is a normalization factor to bias exploration down with high-dimensional actions
        self.stdev_norm = np.prod(self.action_shape)

        self.experience_buffer = [experience_store.NStepTDBuffer(self.obs_shape, self.action_shape, self.t_max, self.discount_factor) for _ in range(self.actor_count)]

    def act(self, obs, actor=0, temp=1):
        if self.rng.random() > self.stoch_persistence:
            self.stoch_mode_t[actor] = self.rng.random(self.action_shape)
            self.stoch_gauss_t[actor] = self.rng.random(self.action_shape)
            self.stoch_gauss_inverf[actor] = np.sqrt(2) * scipy.special.erfinv(2 * self.stoch_gauss_t[actor] - 1)

        # shape: self.action_shape + (self.action_modes, 3), where the inner three values are the weight, mean, and standard deviation for each mode
        output = np.array(self.policy_network.apply(np.expand_dims(obs, axis=0))[0])
        weights = output[..., 0]
        pre_means = output[..., 1]
        pre_stdevs = output[..., 2]

        softmax_weights = np.exp(weights) / np.sum(np.exp(weights), axis=-1, keepdims=True)

        means = np.clip(pre_means, -1, 1)

        # this is the formula for the sigmoid function, with certain biases added
        stdevs = 1/(1 + np.exp(pre_stdevs))/self.stdev_norm+0.0001

        cum_weights = np.cumsum(softmax_weights, axis=-1)
        sel_mode = np.argmax(np.expand_dims(self.stoch_mode_t[actor], axis=-1) < cum_weights, axis=-1, keepdims=True)

        sel_means = np.squeeze(np.take_along_axis(means, sel_mode, axis=-1), axis=-1)
        sel_stdevs = np.squeeze(np.take_along_axis(stdevs, sel_mode, axis=-1), axis=-1)

        # this is the quantile formula of the Gaussian distribution
        sel_actions = sel_means + temp * sel_stdevs * self.stoch_gauss_inverf[actor]

        if actor == 0 and debug_print_action_calc:
            print(self.stoch_mode_t[actor])
            for i in range(self.action_modes):
                print(softmax_weights[..., i], means[..., i], stdevs[..., i], "*" if (sel_mode == i).all() else "")
            print(self.stoch_gauss_t[actor])
            print(sel_actions)

        return sel_actions

    @tf.function
    def log_prob_actions(self, obss, actions):
        # take the weighted probability of each coordinate of each action over modes,
        #   then transform into log space and add over coordinates
        # maintains the batch axis

        # shape: (n,) + self.action_shape + (self.action_modes, 3)
        outputs = self.policy_network.apply(obss)
        weights = outputs[..., 0]
        pre_means = outputs[..., 1]
        pre_stdevs = outputs[..., 2]

        softmax_weights = tf.nn.softmax(weights, axis=-1)

        means = tf.clip_by_value(pre_means, -1, 1)

        stdevs = tf.nn.sigmoid(pre_stdevs)/self.stdev_norm+0.0001

        # want to get the PDF of each real action in each of the modes
        # this is the PDF formula of the Gaussian distribution
        mode_PDFs = tf.exp(-0.5 * ((means - tf.expand_dims(actions, axis=-1)) / stdevs)**2) / (stdevs * np.sqrt(2*np.pi))

        # take the weighted average over modes to get the probability of each coordinate of each action
        weighted_PDF = tf.reduce_sum(softmax_weights * mode_PDFs, axis=-1)+0.0001
        log_PDF = tf.math.log(weighted_PDF)

        # add in log space to get joint probability of all action coordinates, keeping the batch axis 0
        total_log_PDF = tf.reduce_sum(log_PDF, axis=range(1, len(log_PDF.shape)))

        return total_log_PDF

    @tf.function
    def action_entropies(self, obss):
        # Gets the entropy of the action distribution for each observation.
        # Unfortunately, I did not have time to find an analytic solution
        #   to the problem of finding the entropy of the mixture-of-Gaussians
        #   distribution. This is an open problem as described in
        #   https://isas.iar.kit.edu/pdf/MFI08_HuberBailey.pdf.
        # I have some novel techniques for describing the information theory
        #   of continuous random variables but I don't know if they will apply
        #   here.
        # Instead, I take the approach of overestimating entropy by assuming
        #   zero overlap between the component distributions.

        # shape: (n,) + self.action_shape + (self.action_modes, 3)
        outputs = self.policy_network.apply(obss)
        weights = outputs[..., 0]
        pre_means = outputs[..., 1]
        pre_stdevs = outputs[..., 2]
        stdevs = tf.nn.sigmoid(pre_stdevs)/self.stdev_norm+0.0001

        means = tf.clip_by_value(pre_means, -1, 1)

        # By my bag-of-tricks theorem (there is probably a better name for it somewhere),
        #   the entropy of a discrete mixture of RVs, selected according to some index RV
        #   X, is equal to to entropy of X plus the expected entropy of the selected RV.
        # Do not worry about the fact that I am mixing differential and discrete entropies.
        #   In my work unifying discrete and continuous information theory, I have shown
        #   that differential entropy is the finite deviation, in bits or nats, between the infinite
        #   entropies of your continuous random variable of interest and the unit uniform
        #   random variable of the same dimension.
        # So differential entropy and discrete entropy have the same units - nats - and all is well.

        mode_entropies = 0.5 * tf.math.log(2 * np.pi * stdevs**2) + 0.5

        # Because of clipping at +/-1, the above formula isn't quite right. Penalize means near +/- 1.
        mode_entropies -= means**4

        if self.action_modes > 1:
            softmax_weights = tf.nn.softmax(weights, axis=-1)
            weight_entropy_terms = -tf.math.log(softmax_weights)
            coordinate_entropies = tf.reduce_sum(tf.math.multiply_no_nan(weight_entropy_terms + mode_entropies, softmax_weights), axis=-1)
        else:
            coordinate_entropies = tf.reduce_sum(mode_entropies, axis=-1)

        # now we add the entropy of each coordinate since they are independent in this version.
        total_entropies = tf.reduce_sum(coordinate_entropies, axis=range(1, len(coordinate_entropies.shape)))

        return total_entropies

    def store(self, actor, obs, action, reward, terminal):
        self.experience_buffer[actor].store(obs, action, reward, terminal)

    @tf.function
    def train_iter(self, S, A, R):
        cur_value = self.value_network.apply(S)[:,0]
        adv = R - cur_value

        log_prob_actions = self.log_prob_actions(S, A)
        action_entropies = self.action_entropies(S)
        obj = tf.reduce_sum(log_prob_actions * adv + self.beta_entropy_weight * action_entropies)

        obj_gradient = tf.gradients(-obj, self.policy_network.keras_network.weights)
        self.policy_network.optimizer.apply_gradients(zip(obj_gradient, self.policy_network.keras_network.weights))

        value_loss = tf.reduce_sum(adv ** 2)
        value_gradient = tf.gradients(value_loss, self.value_network.keras_network.weights)
        self.value_network.optimizer.apply_gradients(zip(value_gradient, self.value_network.keras_network.weights))

        return obj, value_loss

    def train(self, epochs=1):
        all_S = []
        all_A = []
        all_R = []
        # S2 and dt are used for the TD updates.
        # S2 is the state after all short-term reward has been received.
        # dt is the number of timesteps separating S from S2.
        # If dt is 0, then S2 is full of garbage values because the episode ended.
        all_dt = []
        all_S2 = []

        for buffer in self.experience_buffer:
            S, A, R, dt, S2 = buffer.report()
            buffer.clear()

            all_S.append(S)
            all_A.append(A)
            all_R.append(R)
            all_dt.append(dt)
            all_S2.append(S2)

        all_S = np.concatenate(all_S, axis=0)
        all_A = np.concatenate(all_A, axis=0)
        all_R = np.concatenate(all_R, axis=0)
        all_dt = np.concatenate(all_dt, axis=0)
        all_S2 = np.concatenate(all_S2, axis=0)

        if debug_plot_values:
            init_pred_R = np.array(self.value_network.apply(all_S))

        terminal = (all_dt == 0)

        # use TD to estimate the total (short-term + long-term) expected reward
        if not np.all(terminal):
            all_R = np.where(terminal, all_R, all_R + self.discount_factor ** all_dt * np.array(self.value_network.apply(all_S2))[:,0])
            all_R = np.float32(all_R)
        del all_dt
        del all_S2

        total_training_samples = all_S.shape[0]
        num_minibatches = (total_training_samples * epochs) // self.minibatch_size
        last_minibatch_size = (total_training_samples * epochs) % self.minibatch_size

        total_obj = 0
        total_value_loss = 0

        if self.use_tqdm:
            import tqdm
            minibatches = tqdm.tqdm(range(num_minibatches))
        else:
            minibatches = range(num_minibatches)

        for _ in minibatches:
            sample_indices = self.rng.integers(total_training_samples, size=(self.minibatch_size))
            S = all_S[sample_indices]
            A = all_A[sample_indices]
            R = all_R[sample_indices]
            if S.size > 0:
                obj, value_loss = self.train_iter(S, A, R)
                total_obj += obj
                total_value_loss += value_loss

        if last_minibatch_size:
            sample_indices = self.rng.integers(total_training_samples, size=(last_minibatch_size))
            S = all_S[sample_indices]
            A = all_A[sample_indices]
            R = all_R[sample_indices]
            if S.size > 0:
                obj, value_loss = self.train_iter(S, A, R)
                total_obj += obj
                total_value_loss += value_loss

        value_rmse = np.sqrt(np.array(total_value_loss) / total_training_samples / epochs)
        mean_obj = np.array(total_obj) / total_training_samples / epochs

        if debug_plot_values:
            import matplotlib.pyplot as plt
            plt.subplot(1, 3, 1)
            plt.plot(all_R)
            plt.subplot(1, 3, 2)
            plt.plot(init_pred_R)

            final_pred_R = np.array(self.value_network.apply(all_S))

            plt.subplot(1, 3, 3)
            plt.plot(final_pred_R)
            plt.show()

        # if self.target_Q_network_update_rate:
        #     total_target_update = 1 - (1-self.target_Q_network_update_rate)**total_training_samples
        #     self.target_Q_network.copy_from(self.Q_network, amount=total_target_update)

        # print(self.Q_network.keras_network(np.linspace(-1, 1, 11).reshape(-1, 1)))

        return value_rmse, mean_obj

    def get_weights(self):
        return self.policy_network.keras_network.get_weights(), self.value_network.keras_network.get_weights()
