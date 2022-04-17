import numpy as np

import gym

import pybullet_data
import pybulletgym

def downscale(image):
    new_image = np.zeros((image.shape[0]//2, image.shape[1]//2, image.shape[2]), image.dtype)
    for new_row in range(new_image.shape[0]):
        old_row = new_row * 2
        for new_col in range(new_image.shape[1]):
            old_col = new_col * 2
            new_image[new_row, new_col] = image[old_row, old_col]
    return new_image

def preprocess(image):
    bg = image[0, -1]

    # restrict to the playing field
    field = image[17:97]

    ball_color = np.uint8([[[236, 236, 236]]])
    P1_color = np.uint8([[[92, 186, 92]]])
    P2_color = np.uint8([[[213, 130, 74]]])

    # disentangle features insofar as possible
    is_ball = np.all(field == ball_color, axis=-1)
    is_P1 = np.all(field == P1_color, axis=-1)
    is_P2 = np.all(field == P2_color, axis=-1)
    new_image = np.float32(np.stack([is_ball, is_P1, is_P2], axis=2))

    return new_image

class PongTask:
    def __init__(self, rng):
        self.pong_env = gym.make('PongNoFrameskip-v0')
        self.obs_buffer = [None, None, None]
        self.obs_index = -3
        self.points = 0
        self.rng = rng
        self.real_reset()
    def real_reset(self):
        self.obs_index = -3
        self.points = 0

        self.pong_env.seed(int(self.rng.integers(2**31)))
        raw_obs = self.pong_env.reset()
        obs = preprocess(downscale(raw_obs))
        self.obs_buffer[self.obs_index] = obs
        self.obs_index += 1

        return np.concatenate([obs, obs], axis=2)

    def reset(self):
        if self.obs_index >= 0:
            self.obs_buffer[0] = self.obs_buffer[self.obs_index-1]
            self.obs_index = -3
        return np.concatenate([self.obs_buffer[0], self.obs_buffer[0]], axis=2)

    def step(self, action):
        if action == 0:
            action = 2
        else:
            action = 5
        raw_obs, reward, terminal, info = self.pong_env.step(action)

        if terminal:
            if not reward:
                if self.points < 0:
                    reward = -1.
                else:
                    reward = 1.
    
            combined_obs = self.real_reset()
        else:
            obs = preprocess(downscale(raw_obs))

            if self.obs_index < 0:
                old_obs = self.obs_buffer[0]
            else:
                old_obs = self.obs_buffer[self.obs_index]

            self.obs_buffer[self.obs_index] = obs
            self.obs_index += 1
            if self.obs_index >= 3:
                self.obs_index = 0

            combined_obs = np.concatenate([obs, old_obs], axis=2)

            if reward:
                terminal = 1
                self.points += reward

        return combined_obs, reward, terminal, info
    def render(self):
        view = self.pong_env.render()



class CartPoleTask:
    def __init__(self, rng):
        import render_cartpole
        self.cartpole_env = gym.make("InvertedPendulumMuJoCoEnv-v0")
        self.cartpole_env.seed(int(rng.integers(2**63-1)))
        self.obs_shape = (4,)
        self.action_shape = (1,)
        self.ts = 0
    def reset(self):
        self.ts = 0
        return self.cartpole_env.reset()
    def step(self, action):
        o, r, t, _ = self.cartpole_env.step(action)
        self.ts += 1
        return o, -1. if t and self.ts != 1000 else 0., t, _
    def render(self):
        return self.cartpole_env.render()
    def get_return(self):
        return self.ts

class TrivialTask:
    def __init__(self, rng):
        self.rng = rng
        self.reset()
    def step(self, action):
        orig_lt0 = self.loc < 0
        if action:
            self.loc += .1
        else:
            self.loc -= .1

        if self.loc <= -1 or self.loc >= 1:
            return (self.loc, -1, True, None)

        now_lt0 = self.loc < 0
        if orig_lt0 and not now_lt0 or now_lt0 and not orig_lt0:
            return (self.loc, 1, True, None)

        return (np.array((self.loc,)), 0, False, None)
    def reset(self):
        self.loc = self.rng.random()*2-1
        return np.array((self.loc,))
