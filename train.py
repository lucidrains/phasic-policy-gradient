import fire
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# helpers

def exists(val):
    return val is not None

# networks

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden), self.value_head(hidden)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

# agent

class PPG:
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        lr,
        betas
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.epochs = epochs

    def learn(self, memories):
        memories.clear()

# replay buffer

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done'])

class ReplayBuffer:
    def __init__(self, max_length):
        self.memories = deque([])
        self.max_length = max_length

    def __len__(self):
        return len(self.memories)

    def clear(self):
        self.memories.clear()

    def append(self, el):
        self.memories.append(el)
        if len(self.memories) > self.max_length:
            self.memories.popleft()

# main

def main(
    env_name = 'LunarLander-v2',
    num_episodes = 100,
    max_timesteps = 300,
    actor_hidden_dim = 256,
    critic_hidden_dim = 256,
    max_memories = 300,
    lr = 3e-4,
    betas = (0.9, 0.999),
    eps_clip = 0.2,
    value_clip = 0.2,
    update_timesteps = 2000,
    epochs = 4,
    seed = None,
    render = False
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = ReplayBuffer(max_memories)
    agent = PPG(state_dim, num_actions, actor_hidden_dim, critic_hidden_dim, epochs, lr, betas)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0

    for eps in range(num_episodes):
        print(f'running episode {eps}')
        state = env.reset()

        for timestep in range(max_timesteps):
            time += 1

            if render:
                env.render()

            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            state, reward, done, _ = env.step(action)

            memory = Memory(state, action, action_log_prob, reward, done)
            memories.append(memory)

            if timestep % update_timesteps == 0:
                agent.learn(memories)

            if done:
                break

        if render:
            env.close()

if __name__ == '__main__':
    fire.Fire(main)
