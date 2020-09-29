import fire
from collections import deque, namedtuple

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done'])
AuxMemory = namedtuple('Memory', ['state', 'target_value'])

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

# dataloaders

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

# agent

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = F.smooth_l1_loss(value_clipped.flatten(), rewards, reduction = 'none')
    value_loss_2 = F.smooth_l1_loss(values.flatten(), rewards, reduction = 'none')
    return 0.5 * torch.mean(torch.max(value_loss_1, value_loss_2))

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
        gamma,
        beta_s,
        eps_clip,
        value_clip
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

    def learn(self, memories, aux_memories):
        # get discounted sum of rewards
        rewards = []
        discounted_reward = 0
        for mem in reversed(memories):
            reward, done = mem.reward, mem.done
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - float(done)))
            rewards.insert(0, discounted_reward)

        # retrieve and prepare data from memory for training
        states = []
        actions = []
        old_log_probs = []

        for mem in memories:
            states.append(torch.from_numpy(mem.state))
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(rewards).float().to(device)
        rewards = normalize(rewards)

        # store state and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(states, rewards)
        aux_memories.append(aux_memory)

        # get old value as reference for clipping new value updates
        old_values = self.critic(states).detach_()

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
                advantages = rewards - values.detach()
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
        for state, reward in aux_memories:
            states.append(state)
            rewards.append(reward)

        states = torch.cat(states)
        rewards = torch.cat(rewards)

        # get old action and value predictions for minimizing kl divergence and clipping respectively
        old_action_probs, _ = self.actor(states)
        old_action_logprobs = old_action_probs.log().detach_()
        old_values = self.critic(states).detach_()

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader([states, old_action_logprobs, rewards, old_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        for epoch in range(self.epochs_aux):
            for states, old_action_logprobs, rewards, old_values in tqdm(dl, desc=f'auxiliary epoch {epoch}'):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                aux_loss = 0.5 * F.mse_loss(policy_values.flatten(), rewards)
                loss_kl = F.kl_div(action_logprobs, old_action_logprobs, log_target = True, reduction = 'batchmean')
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
    max_timesteps = 400,
    actor_hidden_dim = 64,
    critic_hidden_dim = 64,
    minibatch_size = 64,
    lr = 0.0005,
    betas = (0.9, 0.999),
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.2,
    beta_s = .01,
    update_timesteps = 5000,
    num_policy_updates_per_aux = 32,
    epochs = 1,
    epochs_aux = 3,
    seed = None,
    render = False,
    render_every_eps = 500
):
    env = gym.make(env_name)
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
        gamma,
        beta_s,
        eps_clip,
        value_clip
    )

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    updated = False
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc='episodes'):
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
                agent.learn(memories, aux_memories)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                if render:
                    updated = False
                break

        if render:
            env.close()

if __name__ == '__main__':
    fire.Fire(main)
