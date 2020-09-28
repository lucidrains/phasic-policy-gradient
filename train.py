import fire
from collections import deque
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# helpers

def exists(val):
    return val is not None

# networks

# agent

# replay buffer

# main

def main(
    env_name = 'LunarLander-v2',
    num_episodes = 100,
    max_timesteps = 300,
    seed = None,
    render = False
):
    env = gym.make(env_name)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    for eps in range(num_episodes):
        print(f'running episode {eps}')
        env.reset()

        for timestep in range(max_timesteps):
            if render:
                env.render()

            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if done:
                break

        if render:
            env.close()

if __name__ == '__main__':
    fire.Fire(main)
