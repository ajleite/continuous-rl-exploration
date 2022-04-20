import numpy as np

import gym

import pybullet_data
import pybulletgym

class TrivialContinuousTask:
    def __init__(self, rng):
        self.obs_shape = (6,)
        self.action_shape = self.obs_shape
    def reset(self):
        self.value = np.random.random(size=self.obs_shape)*2-1
        self.cumulative_r = 0
        self.ts = 0
        return self.value.copy()
    def step(self, action):
        self.value += action*0.01 + np.random.normal(size=self.obs_shape)*0.05
        r = -np.sum(self.value*self.value)
        self.cumulative_r += r
        self.ts += 1
        return self.value.copy(), r, self.ts == 200, None
    def render(self):
        print(f'State {self.ts}: {self.value}')
    def get_return(self):
        return self.cumulative_r

class CartPoleTask:
    def __init__(self, rng):
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

class HalfCheetahTask:
    def __init__(self, rng, to_render=False):
        self.env = gym.make("HalfCheetahMuJoCoEnv-v0")
        if to_render:
            self.env.render('human')
        self.env.seed(int(rng.integers(2**63-1)))

        self.obs_shape = (17,)
        self.action_shape = (6,)
        self.cumulative_r = 0
        self.timestep = 0
    def reset(self):
        self.cumulative_r = 0
        self.timestep = 0
        return self.env.reset()
    def step(self, action):
        o, r, t, _ = self.env.step(action)
        self.timestep += 1
        self.cumulative_r += r
        return o, r, t or self.timestep == 500, _
    def render(self):
        return self.env.render('human')
    def get_return(self):
        return self.cumulative_r

class HalfCheetahVAETask:
    def __init__(self, rng, no_state=False):
        import vae
        self.vae = vae.make_default_vae(latent_dim=8)

        self.rng = rng

        self.env = gym.make("HalfCheetahMuJoCoEnv-v0")
        self.env.seed(int(rng.integers(2**63-1)))
        self.env.env._render_width = 96
        self.env.env._render_height = 96

        self.obs_shape = (self.vae.latent_dim,)
        self.action_shape = (6,)
        self.cumulative_r = 0

        self.no_state = no_state

        if self.no_state:
            self.im_buffer = np.zeros((1024, 96, 96, 3))
            self.im_buffer_i = 0
        else:
            self.vae.inference_net.load_weights('out/cheetah_inf.tfdat')
            self.vae.generator_net.load_weights('out/cheetah_gen.tfdat')
    def reset(self):
        self.cumulative_r = 0
        self.env.reset()

        if self.no_state:
            obs = np.zeros(self.obs_shape)
        else:
            vis_obs = self.env.render('rgb_array')
            obs = self.vae.sample_latent(vis_obs)

        return obs

    def step(self, action):
        _, r, t, _ = self.env.step(action)

        if (not self.no_state) or self.rng.random() < 0.1:
            vis_obs = self.env.render('rgb_array')

            if self.no_state:
                self.im_buffer[self.im_buffer_i] = vis_obs
                self.im_buffer_i += 1
                if self.im_buffer_i == self.im_buffer.shape[0]:
                    self.im_buffer = np.concatenate([self.im_buffer, np.zeros((1024, 96, 96, 3))], axis=0)

        if self.no_state:
            obs = np.zeros(self.obs_shape)
        else:
            obs = self.vae.sample_latent(vis_obs)

        self.cumulative_r += r
        return obs, r, t, _
    def render(self):
        return self.env.render()
    def get_return(self):
        return self.cumulative_r
    def get_samples(self):
        return self.im_buffer[:self.im_buffer_i]

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
