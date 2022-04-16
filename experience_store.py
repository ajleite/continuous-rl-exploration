import numpy as np

import util

# Unfortunately, it is not reasonable to use a single programmatical method for both the Monte Carlo trajectory buffer and the TD(0) experience buffer,
# but the two methods can share an interface.

# The buffers expect observations to be float32s normalized to the range [0, 1].
# Please note that the buffers receive int32 action indices as input,
# but produce one-hot encodings corresponding to these indices as output.

class NStepTDBuffer:
    def __init__(self, obs_shape, action_shape, t_max, discount_factor, buffer_size=10000):
        self.S_samples = np.zeros((buffer_size,)+obs_shape, dtype=np.float32)
        self.A_samples = np.zeros((buffer_size,)+action_shape, dtype=np.float32)
        self.R_samples = np.zeros((buffer_size,), dtype=np.float32)
        self.dt_samples = np.zeros((buffer_size,), dtype=np.int32)
        self.S2_samples = np.zeros((buffer_size,)+obs_shape, dtype=np.float32)

        self.t_max = t_max
        self.discount_factor = discount_factor

        # buffer_size represents the size of the buffer.
        # cur_index represents the next index that will be written.
        # filled represents whether the buffer has been filled at least once (can be sampled freely).
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.filled = False

        self.episode_buffer = []

    def store_episode(self, observations, actions, rewards):
        ''' store_episode should be called at the end of an episode.

            observations, actions, and rewards should be numpy arrays
            whose shapes align along their first axis (timestep axis).

            - observations should be a float32 array whose subsequent axes match obs_shape.
            - actions should be an int32 array with no subsequent axes.
            - rewards should be a float32 array with no subsequent axis,
              and it should represent the rewards for the transitions
              following the (observation, action) pairs in the corresponding indices. '''

        trajectory_length = rewards.shape[0]

        # discard unusable information ASAP
        if trajectory_length > self.buffer_size:
            print(f"WARNING: experience buffer overflow (trajectory length {trajectory_length}, buffer size {self.buffer_size}). Clipping beginning of trajectory.")
            observations = observations[-self.buffer_size:]
            actions = actions[-self.buffer_size:]
            rewards = rewards[-self.buffer_size:]
            trajectory_length = self.buffer_size

        t_max = self.t_max
        if t_max <= 0:
            t_max = trajectory_length

        # calculate trajectory rewards for each timestep
        # (ensure double precision for this calculation)
        traj_rewards = np.array(rewards, dtype=np.float64)
        for offset in range(1, min(trajectory_length, t_max)):
            traj_rewards[:-offset] += rewards[offset:] * self.discount_factor**offset
        traj_rewards = np.float32(traj_rewards)
        # print(traj_rewards)

        if trajectory_length <= t_max:
            dt = np.zeros(trajectory_length, dtype=np.int32)
            s2 = observations
        else:
            dt_incomplete = np.full(trajectory_length - t_max, t_max, dtype=np.int32)
            dt_complete = np.zeros(t_max, dtype=np.int32)
            dt = np.concatenate([dt_incomplete, dt_complete], axis=0)

            s2_incomplete = observations[t_max:]
            s2_complete = np.zeros_like(observations[:trajectory_length - t_max])
            s2 = np.concatenate([s2_incomplete, s2_complete], axis=0)

        # store the trajectory in the buffer
        # (split based on whether we will wrap around the end of the buffer)
        will_loop = self.cur_index + trajectory_length >= self.buffer_size
        if will_loop:
            can_store = self.buffer_size - self.cur_index
            self.S_samples[self.cur_index:] = observations[:can_store]
            self.A_samples[self.cur_index:] = actions[:can_store]
            self.R_samples[self.cur_index:] = traj_rewards[:can_store]
            self.dt_samples[self.cur_index:] = dt[:can_store]
            self.S2_samples[self.cur_index:] = s2[:can_store]

            leftover = trajectory_length - can_store
            if leftover:
                self.S_samples[:leftover] = observations[can_store:]
                self.A_samples[:leftover] = actions[can_store:]
                self.R_samples[:leftover] = traj_rewards[can_store:]
                self.dt_samples[:leftover] = dt[can_store:]
                self.S2_samples[:leftover] = s2[can_store:]

            self.filled = True
            self.cur_index = (self.cur_index + trajectory_length) - self.buffer_size
        else:
            new_index = self.cur_index + trajectory_length

            self.S_samples[self.cur_index:new_index] = observations
            self.A_samples[self.cur_index:new_index] = actions
            self.R_samples[self.cur_index:new_index] = traj_rewards
            self.dt_samples[self.cur_index:new_index] = dt
            self.S2_samples[self.cur_index:new_index] = s2

            self.cur_index = new_index

    def store(self, obs, action, reward, terminal):
        ''' This is a convenience function to allow the MonteCarloBuffer to act more like the TD0Buffer.
            It matches the signature of TD0Bufer.store. '''

        self.episode_buffer.append((obs, action, reward))

        # This is an unusual idiom that allows one to effectively take the transpose of a Python list.
        # I often use it in RL contexts.
        if terminal:
            all_S, all_A, all_R = [np.array(all_samples) for all_samples in zip(*self.episode_buffer)]
            self.store_episode(all_S, all_A, all_R)
            self.episode_buffer = []

    def report(self):
        if self.filled:
            return (self.S_samples, self.A_samples, self.R_samples, self.dt_samples, self.S2_samples)
        else:
            return (self.S_samples[:self.cur_index], self.A_samples[:self.cur_index], self.R_samples[:self.cur_index],
                self.dt_samples[:self.cur_index], self.S2_samples[:self.cur_index])

    def clear(self):
        self.cur_index = 0
        self.filled = False

        self.episode_buffer = []
