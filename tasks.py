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
	new_image = np.stack([is_ball, is_P1, is_P2], axis=2)

	return new_image

class PongTask:
	def __init__(self):
		self.pong_env = gym.make('PongNoFrameskip-v0')
		self.obs_buffer = [None, None, None, None]
		self.obs_index = -4
	def reset(self):
		self.obs_buffer = [None, None, None, None]
		self.obs_index = -4

		obs = self.pong_env.reset()
		self.obs_buffer[self.obs_index] = obs
		self.obs_index += 1

		return preprocess(downscale(obs))
	def step(self, action):
		obs, reward, terminal, info = self.pong_env.step(action)
		if self.obs_index < 0:
			old_obs = self.obs_buffer[0]
		else:
			old_obs = self.obs_buffer[self.obs_index]

class CartPoleTask:
	def __init__(self):
		self.cartpole_env = gym.make("InvertedPendulumMuJoCoEnv-v0")
	def reset(self):
		return self.cartpole_env.reset()
	def step(self, action):
		return self.cartpole_env.step(((-1.,), (1.,))[action])
	def render(self):
		return self.cartpole_env.render(self)

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
