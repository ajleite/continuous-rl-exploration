import pickle

import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes, path=None):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes

        self.training_episode_rewards = []
        self.eval_episode_rewards = []

        self.value_rmse = []
        self.mean_obj = []

        self.best_weights = None
        self.best_100_episode_return = None
        self.path = path

    def save_trace(self):
        if not self.path:
            return

        to_save = {'best_weights': self.best_weights, 'best_100_episode_return': self.best_100_episode_return,
            'training_episode_rewards': self.training_episode_rewards, 'eval_episode_rewards': self.eval_episode_rewards,
            'value_rmse': self.value_rmse, 'mean_obj': self.mean_obj}

        pickle.dump(to_save, open(self.path,'wb'))

    def run(self, render=False):
        timestep = 0

        for n in range(self.num_episodes):
            # 1. gather training trajectories
            training_rewards = []
            training_lengths = []

            for traj in range(self.agent.actor_count):
                s = self.task.reset()
                t = False

                while not t:
                    a = self.agent.act(s, traj)

                    s2, r, t, _ = self.task.step(a)
                    self.agent.store(traj, s, a, r, t)
                    s = s2

                    timestep += 1

                    if render and traj == 0:
                        self.task.render()

                # episode is over, record it
                training_rewards.append(self.task.get_return())
                print((n, 't', traj, self.task.get_return()))

            self.training_episode_rewards.append(training_rewards)

            # 2. train for 16 epochs
            value_rmse, mean_obj = self.agent.train(16)
            self.value_rmse.append(value_rmse)
            self.mean_obj.append(mean_obj)
            print(n, 'l', value_rmse, mean_obj)

            # 3. gather greedy evaluation trajectories every 50 training episodes
            if n % 50 / self.agent.actor_count >= 1:
                continue

            eval_rewards = []
            eval_lengths = []
            for i in range(100):
                s = self.task.reset()
                t = False

                while not t:
                    a = self.agent.act(s, traj, temp=0)

                    s2, r, t, _ = self.task.step(a)
                    s = s2

                    if render and i == 0:
                        self.task.render()

                # episode is over, record it
                eval_rewards.append(self.task.get_return())
                print((n, 'e', i, self.task.get_return()))

            self.eval_episode_rewards.append(eval_rewards)

            # maintain best stats
            if self.best_100_episode_return is None or np.mean(eval_rewards) > self.best_100_episode_return:
                self.best_100_episode_return = np.mean(eval_rewards)
                self.best_weights = self.agent.get_weights()
            print('cycle', n, 'mean eval:', np.mean(eval_rewards), 'best:', self.best_100_episode_return)
            self.save_trace()

        # everything is done!
        self.save_trace()
