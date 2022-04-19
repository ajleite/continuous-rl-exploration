import pickle

import numpy as np
import tensorflow as tf

import gym

import experience_store
import agent
import network
import simulation
import tasks

def test_REINFORCE_cartpole(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task = tasks.CartPoleTask(task_rng)

    path = f'out/REINFORCE-cartpole-{seed}.pickle'

    # expected time to switch action distribution is 20 timesteps
    policy_network = network.Network(task.obs_shape, network.FFANN_factory([40, 20]), 0.0001, True, task.action_shape, 1)
    value_network = network.Network(task.obs_shape, network.FFANN_factory([40, 20]), 0.0001, False, task.action_shape, 1)
    ag = agent.AdvantageAgent(agent_rng, 1, policy_network, value_network, 0, 0.95, 0.965, 0)

    sim = simulation.Simulation(ag, task, 4000)
    sim.run(False)


def test_A2C_cartpole(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task = tasks.CartPoleTask(task_rng)

    path = f'out/A2C-cartpole-{seed}.pickle'

    # expected time to switch action distribution is 20 timesteps
    policy_network = network.Network(task.obs_shape, network.FFANN_factory([40, 20]), 0.0001, True, task.action_shape, 3)
    value_network = network.Network(task.obs_shape, network.FFANN_factory([40, 20]), 0.0001, False, task.action_shape, 3)
    ag = agent.AdvantageAgent(agent_rng, 25, policy_network, value_network, 0, 0.95, 0.965, 0.05)

    sim = simulation.Simulation(ag, task, 160)
    sim.run(False)


def test_A2CTD_cartpole(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task = tasks.CartPoleTask(task_rng)

    path = f'out/A2CTD-cartpole-{seed}.pickle'

    # expected time to switch action distribution is 20 timesteps
    policy_network = network.Network(task.obs_shape, network.FFANN_factory([100, 50]), 0.0001, True, task.action_shape, 3)
    value_network = network.Network(task.obs_shape, network.FFANN_factory([100, 50]), 0.001, False, task.action_shape, 3)
    ag = agent.AdvantageAgent(agent_rng, 10, policy_network, value_network, 8, 0.99, 0.965, 0.05)

    sim = simulation.Simulation(ag, task, 400)
    sim.run(False)


def test_REINFORCE_cheetah(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task = tasks.HalfCheetahTask(task_rng)

    path = f'out/REINFORCE-cheetah-{seed}.pickle'

    # expected time to switch action distribution is 20 timesteps
    policy_network = network.Network(task.obs_shape, network.FFANN_factory([160, 80]), 0.0001, True, task.action_shape, 1)
    value_network = network.Network(task.obs_shape, network.FFANN_factory([160, 80]), 0.0001, False, task.action_shape, 1)
    ag = agent.AdvantageAgent(agent_rng, 1, policy_network, value_network, 0, 0.99, 0.965, 0)

    sim = simulation.Simulation(ag, task, 2500)
    sim.run(False)

if __name__ == '__main__':
    for i in range(20):
        test_REINFORCE_cartpole(i)
        test_A2C_cartpole(i)

    for i in range(5):
        test_REINFORCE_cheetah(i)

    import sys
    a = sys.argv[1]
    t = sys.argv[2]
    seed = int(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4] == '-r':
        replay = True
    else:
        replay = False

    if t == 'pong':
        tas = pong_config
    elif t == 'pong-conv':
        tas = lambda rng: pong_config(rng, boring_network=True)
    elif t == 'cartpole':
        tas = cart_pole_config
    else:
        print('invalid task')
        sys.exit()

    if a == 'MC':
        test_MC_Agent(seed, tas, replay=replay)
    elif a == 'FQI':
        test_FQI_Agent(seed, tas)
    elif a == 'DQN':
        test_DQN_Agent(seed, tas)
    else:
        print('invalid agent')
