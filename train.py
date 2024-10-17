from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from collections import deque, namedtuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import AdamAtan2

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data

Memory = namedtuple('Memory', [
    'state',
    'action',
    'action_log_prob',
    'reward',
    'done',
    'value'
])

AuxMemory = namedtuple('AuxMemory', [
    'state',
    'target_value',
    'old_values'
])

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

# "bro" mlp

class ReluSquared(Module):
    def forward(self, x):
        return x.sign() * F.relu(x) ** 2

class BroMLP(Module):

    def __init__(
        self,
        dim,
        dim_out = None,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
    ):
        super().__init__()
        """
        following the design of BroNet https://arxiv.org/abs/2405.16158v1
        """

        dim_out = default(dim_out, dim)
        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []
        mixers = []

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            ReluSquared()
        )

        dim_inner = dim_hidden * expansion_factor

        for _ in range(depth):

            layer = nn.Sequential(
                nn.Linear(dim_hidden, dim_inner),
                nn.Dropout(dropout),
                nn.LayerNorm(dim_inner, bias = False),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.LayerNorm(dim_hidden, bias = False),
            )

            nn.init.constant_(layer[-1].weight, 1e-5)
            layers.append(layer)

            mixer = nn.Parameter(torch.ones(dim_hidden))
            mixers.append(mixer)

        # final layer out

        self.layers = ModuleList(layers)
        self.learned_mixers = nn.ParameterList(mixers)

        self.proj_out = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):

        x = self.proj_in(x)

        for layer, mix in zip(self.layers, self.learned_mixers):

            branch_out = layer(x)
            x = x * mix + branch_out

        return self.proj_out(x)

# networks

class Actor(Module):
    def __init__(self, state_dim, hidden_dim, num_actions, mlp_depth = 2):
        super().__init__()
        self.net = BroMLP(
            state_dim,
            dim_out = hidden_dim,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth
        )

        self.action_head = nn.Sequential(
            BroMLP(hidden_dim, num_actions, depth = 1),
            nn.Softmax(dim=-1)
        )

        self.value_head = BroMLP(hidden_dim, 1, depth = 2)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden), self.value_head(hidden)

class Critic(Module):
    def __init__(self, state_dim, hidden_dim, mlp_depth = 6):  # recent paper has findings that show scaling critic is more important than scaling actor
        super().__init__()
        self.net = BroMLP(
            state_dim,
            dim_out = 1,
            dim_hidden = hidden_dim,
            depth = mlp_depth
        )

    def forward(self, x):
        return self.net(x)

# agent

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

class PPG:
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        regen_reg_rate,
        save_path = './ppg.pt'
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)

        self.opt_actor = AdamAtan2(self.actor.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)
        self.opt_critic = AdamAtan2(self.critic.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)

        self.opt_aux_actor = AdamAtan2(self.actor.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)
        self.opt_aux_critic = AdamAtan2(self.critic.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path))
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories, aux_memories, next_state):
        # retrieve and prepare data from memory for training

        (
            states,
            actions,
            old_log_probs,
            rewards,
            dones,
            values
        ) = zip(*memories)

        actions = [torch.tensor(action) for action in actions]
        masks = [(1. - float(done)) for done in dones]

        # calculate generalized advantage estimate

        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = list(values) + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).float().to(device)

        # store state and target values to auxiliary memory buffer for later training

        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training

        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # policy phase training, similar to original PPO

        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)

                # calculate value loss and update value network separate from policy network

                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

    def learn_aux(self, aux_memories):
        # gather states and target values into one tensor

        states, rewards, old_values = zip(*aux_memories)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # get old action predictions for minimizing kl divergence and clipping respectively

        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # prepared dataloader for auxiliary phase training

        dl = create_shuffled_dataloader([states, old_action_probs, rewards, old_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)

        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in tqdm(dl, desc=f'auxiliary epoch {epoch}'):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss

                aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                policy_loss = aux_loss + loss_kl

                update_network_(policy_loss, self.opt_aux_actor)

                # paper says it is important to train the value network extra during the auxiliary phase

                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_aux_critic)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 1000,
    actor_hidden_dim = 32,
    critic_hidden_dim = 256,
    minibatch_size = 64,
    lr = 0.0005,
    betas = (0.9, 0.999),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    regen_reg_rate = 1e-3,
    update_timesteps = 5000,
    num_policy_updates_per_aux = 32,
    epochs = 1,
    epochs_aux = 4,
    seed = None,
    render = False,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = False,
    video_folder = './lunar-recording',
    load = False
):
    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = deque([])
    aux_memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        regen_reg_rate
    )

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, info = env.reset(seed = seed)

        for timestep in range(max_timesteps):
            time += 1

            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            value = agent.critic(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            memory = Memory(state, action, action_log_prob, reward, done, value)
            memories.append(memory)

            state = next_state

            if divisible_by(time, update_timesteps):
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if divisible_by(num_policy_updates, num_policy_updates_per_aux):
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

            if done:
                break

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
