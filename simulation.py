import pickle

import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes, path=None):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes

        self.loss_samples = []
        self.episode_rewards = []
        self.episode_behavior_entropy = []

        # self.best_weights = None
        # self.best_100_episode_return = None
        self.path = path

    def save_trace(self):
        if not self.path:
            return

        to_save = {'best_weights': self.best_weights, 'best_100_episode_return': self.best_100_episode_return,
            'episode_rewards': self.episode_rewards, 'loss_samples': self.loss_samples, 'episode_behavior_entropy': self.episode_behavior_entropy}

        pickle.dump(to_save, open(self.path,'wb'))

    def run(self, render=False):
        timestep = 0

        for n in range(self.num_episodes):
            for traj in range(self.agent.actor_count):
                s = self.task.reset()
                t = False
                total_r = 0

                last_a = None
                ep_length = 0
                while not t:
                    if n % 10:
                        a = self.agent.act(s, traj)
                    else:
                        a = self.agent.act(s, traj, greedy=True)

                    s2, r, t, _ = self.task.step(a)
                    ep_length += 1
                    if ep_length == 1000:
                        t = True
                    self.agent.store(traj, s, a, r, t)
                    s = s2

                    total_r += r
                    timestep += 1

                    if timestep % 2000 == 0:
                        if self.episode_rewards and self.loss_samples:
                            print('episode', n, 'last episode:', 'reward', self.episode_rewards[-1], 'loss', self.loss_samples[-1], 'entropy', self.episode_behavior_entropy[-1])
                        self.save_trace()

                    if render and traj == 0:
                        self.task.render()


                # episode is over, record it
                self.episode_rewards.append((timestep, total_r))
                print((n, ep_length, total_r))
                continue

                # calculate behavior entropy
                total_actions = n_left + n_right
                p_left = n_left / total_actions
                p_right = n_right / total_actions
                if p_left == 0 or p_right == 0:
                    entropy = 0.
                else:
                    entropy = -np.log2(p_left)*p_left + -np.log2(p_right)*p_right
                self.episode_behavior_entropy.append((timestep, entropy))
                n_left = 0
                n_right = 0

                # calculate running average
                if n >= 100:
                    mean_reward = np.mean(self.episode_rewards[-100:], axis=0)[1]
                    if self.best_100_episode_return is None or mean_reward > self.best_100_episode_return:
                        self.best_100_episode_return = mean_reward
                        self.best_weights = self.agent.Q_network.keras_network.get_weights()
                    if not n % 100:
                        print('mean reward for episode', n-100, 'to', n, mean_reward)
                        print('best is', self.best_100_episode_return)

            value_rmse, mean_obj = self.agent.train(16)
            print(value_rmse, mean_obj)

        # everything is done!
        self.save_trace()
