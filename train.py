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

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

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
        epochs_aux,
        lr,
        betas,
        gamma,
        beta_s,
        eps_clip,
        value_clip
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

    def learn(self, memories):
        rewards = []
        discounted_reward = 0
        for mem in reversed(memories.data):
            reward, done = mem.reward, mem.done
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - float(done)))
            rewards.insert(0, discounted_reward)

        states = []
        actions = []
        old_log_probs = []

        for mem in memories.data:
            states.append(torch.from_numpy(mem.state))
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()
        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(rewards).float().to(device)
        rewards = normalize(rewards)

        for _ in range(self.epochs):
            action_probs, _ = self.actor(states)
            values = self.critic(states)
            dist = Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = (action_log_probs - old_log_probs).exp()

            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

            value_loss = 0.5 * F.mse_loss(values.flatten(), rewards)

            update_network_(policy_loss, self.opt_actor)
            update_network_(value_loss, self.opt_critic)

        old_action_probs, _ = self.actor(states)
        old_action_logprobs = old_action_probs.log().detach_()

        for _ in range(self.epochs_aux):
            action_probs, policy_values = self.actor(states)
            action_logprobs = action_probs.log()
            aux_loss = 0.5 * F.mse_loss(policy_values.flatten(), rewards)
            policy_loss = aux_loss + F.kl_div(action_logprobs, old_action_logprobs, log_target = True, reduction = 'batchmean')

            update_network_(policy_loss, self.opt_actor)

            values = self.critic(states)
            value_loss = 0.5 * F.mse_loss(values.flatten(), rewards)

            update_network_(value_loss, self.opt_critic)

        memories.clear()

# replay buffer

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done'])

class ReplayBuffer:
    def __init__(self, max_length):
        self.memories = deque([])
        self.max_length = max_length

    def __len__(self):
        return len(self.memories)

    @property
    def data(self):
        return self.memories

    def clear(self):
        self.memories.clear()

    def append(self, el):
        self.memories.append(el)
        if len(self.memories) > self.max_length:
            self.memories.popleft()

# main

def main(
    env_name = 'LunarLander-v2',
    num_episodes = 100000,
    max_timesteps = 400,
    actor_hidden_dim = 64,
    critic_hidden_dim = 64,
    max_memories = 2000,
    lr = 0.002,
    betas = (0.9, 0.999),
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.2,
    beta_s = .01,
    update_timesteps = 2000,
    epochs = 1,
    epochs_aux = 6,
    seed = None,
    render = False,
    render_every_eps = 500
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = ReplayBuffer(max_memories)
    agent = PPG(state_dim, num_actions, actor_hidden_dim, critic_hidden_dim, epochs, epochs_aux, lr, betas, gamma, beta_s, eps_clip, value_clip)

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    updated = False

    for eps in range(num_episodes):
        print(f'running episode {eps}')
        render = eps % render_every_eps == 0
        state = env.reset()
        for timestep in range(max_timesteps):
            time += 1

            if updated and render:
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

            if time % update_timesteps == 0:
                agent.learn(memories)
                updated = True

            if done:
                if render:
                    updated = False
                break

        if render:
            env.close()

if __name__ == '__main__':
    fire.Fire(main)
