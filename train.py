import os
import fire
from collections import deque, namedtuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import AdamAtan2

import gymnasium as gym

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])

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

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)

        if exists(m.bias):
            torch.nn.init.zeros_(m.bias)

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
        self.apply(init_)

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
        self.apply(init_)

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
        regen_reg_rate
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = AdamAtan2(self.actor.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)
        self.opt_critic = AdamAtan2(self.critic.parameters(), lr=lr, betas=betas, regen_reg_rate=regen_reg_rate)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, f'./ppg.pt')

    def load(self):
        if not os.path.exists('./ppg.pt'):
            return

        data = torch.load(f'./ppg.pt')
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories, aux_memories, next_state):
        # retrieve and prepare data from memory for training
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate
        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

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
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

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

                update_network_(policy_loss, self.opt_actor)

                # paper says it is important to train the value network extra during the auxiliary phase
                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

# main

def main(
    env_name = 'LunarLander-v2',
    num_episodes = 50000,
    max_timesteps = 500,
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
    epochs_aux = 6,
    seed = None,
    render = False,
    render_every_eps = 250,
    save_every = 1000,
    clear_videos = False,
    video_folder = './lunar-recording',
    load = False
):
    env = gym.make(env_name, render_mode = "rgb_array")

    if render:
        if clear_videos:
            from shutil import rmtree
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = "lunar-video",
            episode_trigger = lambda eps_num: (eps_num % render_every_eps) == 0,
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
    updated = False
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc='episodes'):

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

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                break

        if eps % save_every == 0:
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
