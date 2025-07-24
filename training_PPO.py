import collections
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from environment import QuantumEnvironment

# Hyperparameters
learning_rate = 5e-5
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
batch_size = 64
ppo_epochs = 4
update_interval = 1024


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        # Topology embedding conv
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Path embedding
        self.path_fc1 = nn.Linear(32, 128)
        self.path_fc2 = nn.Linear(128, 32)

        # Transformer attention block
        self.q_proj = nn.Linear(32 * 14 * 14, 128)
        self.k_proj = nn.Linear(32, 128)
        self.v_proj = nn.Linear(32, 128)

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.attn_out = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 4)
        self.fc_v = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward_features(self, x, y):
        B, P, D_in = y.shape

        # Topology embedding for Q
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        q = self.q_proj(x).unsqueeze(1)  # [B, 1, D]

        # Path embedding for K and V
        y = y.view(B * P, D_in)
        y = F.relu(self.path_fc1(y))
        y = F.relu(self.path_fc2(y))
        y = y.view(B, P, -1)

        k = self.k_proj(y)  # [B, P, D]
        v = self.v_proj(y)  # [B, P, D]

        # Attention using transformer
        attn_out, _ = self.attn(q, k, v)  # [B, 1, D]
        fused = F.relu(self.attn_out(attn_out.squeeze(1)))  # [B, D]
        return fused

    def pi(self, x, y, mask=None, softmax_dim=0):
        z = self.forward_features(x, y)
        logits = self.fc2(z)
        if mask is not None:
            return self.masked_softmax(logits, mask, dim=softmax_dim)
        else:
            return F.softmax(logits, dim=softmax_dim)

    def v(self, x, y):
        z = self.forward_features(x, y)
        return self.fc_v(z)

    def masked_softmax(self, logits, mask, dim=-1, eps=1e-8):
        mask = mask.to(logits.dtype)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        probs = F.softmax(logits, dim=dim)
        probs = probs * mask
        probs = probs / (probs.sum(dim=dim, keepdim=True) + eps)
        return probs

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst, mask_lst = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done_mask = transition
            s_lst.append(np.squeeze(s['obs'], axis=0))
            s_p_lst.append(np.squeeze(s['padding_paths'], axis=0))
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(np.squeeze(s_prime['obs'], axis=0))
            s_p_prime_lst.append(np.squeeze(s_prime['padding_paths'], axis=0))
            prob_a_lst.append([prob_a])
            done_mask_lst.append([done_mask])
            mask_lst.append(s['valid_mask'])

        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst, mask_lst = \
            (np.array(s_lst), np.array(s_p_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst),
             np.array(s_p_prime_lst), np.array(prob_a_lst), np.array(done_mask_lst), np.array(mask_lst))
        self.data = []

        return (torch.tensor(s_lst, dtype=torch.float), torch.tensor(s_p_lst, dtype=torch.float),
                torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst, dtype=torch.float),
                torch.tensor(s_prime_lst, dtype=torch.float),
                torch.tensor(s_p_prime_lst, dtype=torch.float), torch.tensor(prob_a_lst), torch.tensor(done_mask_lst),
                torch.tensor(mask_lst))

    def sample_action(self, state):
        obs = torch.from_numpy(state['obs']).float()
        obs_path = torch.from_numpy(state['padding_paths']).float()
        valid_mask = torch.tensor(state['valid_mask']).float().unsqueeze(0)

        prob = self.pi(obs, obs_path, mask=valid_mask, softmax_dim=1)
        m = Categorical(prob)
        action = m.sample().item()
        candidate_paths = state['paths']

        if len(candidate_paths) > action:
            routing_path = candidate_paths[action]
        else:
            routing_path = []

        return routing_path, action, prob

    def train_net(self):
        loss_lst = []
        s, s_p, a, r, s_prime, s_p_prime, prob_a, done_mask, mask = self.make_batch()

        r_mean, r_std = r.mean(), r.std()
        r = (r - r_mean) / (r_std + 1e-8)

        for _ in range(ppo_epochs):
            td_target = r + gamma * self.v(s_prime, s_p_prime) * done_mask
            delta = td_target - self.v(s, s_p)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            pi = self.pi(s, s_p, mask=mask, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(prob_a + 1e-8))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, s_p), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            loss_lst.append(loss.detach().numpy())

        return np.mean(loss_lst)

# class PPO(nn.Module):
#     def __init__(self):
#         super(PPO, self).__init__()
#         self.data = []
#
#         # Topology embedding conv
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#
#         # Path embedding
#         self.path_fc1 = nn.Linear(32, 128)
#         self.path_fc2 = nn.Linear(128, 32)
#
#         self.q_proj = nn.Linear(32 * 14 * 14, 128)
#         self.k_proj = nn.Linear(32, 128)
#         self.v_proj = nn.Linear(32, 128)
#
#         self.attn_out = nn.Linear(128, 128)
#
#         self.fc2 = nn.Linear(128, 4)
#         self.fc_v = nn.Linear(128, 1)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#
#     def attention(self, q, k, v):
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         return torch.matmul(attn_weights, v)
#
#     def forward_features(self, x, y):
#         B, P, D_in = y.shape
#
#         # Topology embedding for Q
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)  # [B, C*H*W]
#         q = self.q_proj(x).unsqueeze(1)  # [B, 1, D]
#
#         # Path embedding for K and V
#         y = y.view(B * P, D_in)  # flatten path batch
#         y = F.relu(self.path_fc1(y))  # [B*P, 128]
#         y = F.relu(self.path_fc2(y))  # [B*P, 32]
#         y = y.view(B, P, -1)  # [B, P, 32]
#
#         k = self.k_proj(y)  # [B, P, D]
#         v = self.v_proj(y)  # [B, P, D]
#
#         # Attention: Q = [B, 1, D], K/V = [B, P, D]
#         attn = self.attention(q, k, v).squeeze(1)  # [B, D]
#         fused = F.relu(self.attn_out(attn))  # [B, D]
#
#         return fused
#
#     def pi(self, x, y, mask=None, softmax_dim=0):
#         z = self.forward_features(x, y)
#         logits = self.fc2(z)
#         if mask is not None:
#             return self.masked_softmax(logits, mask, dim=softmax_dim)
#         else:
#             return F.softmax(logits, dim=softmax_dim)
#
#     def v(self, x, y):
#         z = self.forward_features(x, y)
#         return self.fc_v(z)
#
#     def masked_softmax(self, logits, mask, dim=-1, eps=1e-8):
#         mask = mask.to(logits.dtype)
#
#         logits = logits.masked_fill(mask == 0, float('-inf'))
#         probs = F.softmax(logits, dim=dim)
#         probs = probs * mask
#         probs = probs / (probs.sum(dim=dim, keepdim=True) + eps)
#
#         return probs
#
#     def put_data(self, transition):
#         self.data.append(transition)
#
#     def make_batch(self):
#         s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst, mask_lst = [], [], [], [], [], [], [], [], []
#         for transition in self.data:
#             s, a, r, s_prime, prob_a, done_mask = transition
#             s_lst.append(np.squeeze(s['obs'], axis=0))
#             s_p_lst.append(np.squeeze(s['padding_paths'], axis=0))
#             a_lst.append([a])
#             r_lst.append([r])
#             s_prime_lst.append(np.squeeze(s_prime['obs'], axis=0))
#             s_p_prime_lst.append(np.squeeze(s_prime['padding_paths'], axis=0))
#             prob_a_lst.append([prob_a])
#             done_mask_lst.append([done_mask])
#             mask_lst.append(s['valid_mask'])
#
#         s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst, mask_lst = \
#             (np.array(s_lst), np.array(s_p_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst),
#              np.array(s_p_prime_lst), np.array(prob_a_lst), np.array(done_mask_lst), np.array(mask_lst))
#         self.data = []
#
#         return (torch.tensor(s_lst, dtype=torch.float), torch.tensor(s_p_lst, dtype=torch.float),
#                 torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst, dtype=torch.float),
#                 torch.tensor(s_prime_lst, dtype=torch.float),
#                 torch.tensor(s_p_prime_lst, dtype=torch.float), torch.tensor(prob_a_lst), torch.tensor(done_mask_lst),
#                 torch.tensor(mask_lst))
#
#     def sample_action(self, state):
#         obs = torch.from_numpy(state['obs']).float()
#         obs_path = torch.from_numpy(state['padding_paths']).float()
#         valid_mask = torch.tensor(state['valid_mask']).float().unsqueeze(0)
#
#         prob = self.pi(obs, obs_path, mask=valid_mask, softmax_dim=1)
#         m = Categorical(prob)
#         action = m.sample().item()
#         candidate_paths = state['paths']
#
#         if len(candidate_paths) > action:
#             routing_path = candidate_paths[action]
#         else:
#             routing_path = []
#
#         return routing_path, action, prob
#
#     def train_net(self):
#         loss_lst = []
#         s, s_p, a, r, s_prime, s_p_prime, prob_a, done_mask, mask = self.make_batch()
#
#         # Reward normalization
#         r_mean, r_std = r.mean(), r.std()
#         r = (r - r_mean) / (r_std + 1e-8)
#
#         for _ in range(ppo_epochs):
#             td_target = r + gamma * self.v(s_prime, s_p_prime) * done_mask
#             delta = td_target - self.v(s, s_p)
#             delta = delta.detach().numpy()
#
#             advantage_lst = []
#             advantage = 0.0
#             for delta_t in delta[::-1]:
#                 advantage = gamma * lmbda * advantage + delta_t[0]
#                 advantage_lst.append([advantage])
#             advantage_lst.reverse()
#             advantage = torch.tensor(advantage_lst, dtype=torch.float)
#             advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
#
#             pi = self.pi(s, s_p, mask=mask, softmax_dim=1)
#             pi_a = pi.gather(1, a)
#             ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(prob_a + 1e-8))
#
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
#
#             loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, s_p), td_target.detach())
#
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
#             self.optimizer.step()
#
#             loss_lst.append(loss.detach().numpy())
#
#         return np.mean(loss_lst)


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='NSFNET')
    model = PPO()
    episode_memory = []

    max_episode = 100_000
    max_time_step = 100
    print_interval = 20
    score = 0.0
    temp_score = 0.0
    loss = 0.0
    reward = 0.0
    reward_sum = 0.0
    rewards = []
    high_score = 0.0
    scores = []
    losses = []
    avg_losses = []

    for n_epi in range(max_episode):
        s, _ = env.reset(0, max_time_step, True)
        done = False
        reward_sum = 0.0

        while not done:
            a, action, prob = model.sample_action(s)
            prob = prob.squeeze()
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            transition = (s, action, r, s_prime, prob[action].item(), done_mask)
            model.put_data(transition)
            s = s_prime
            reward_sum += r

            if done:
                # print("action: ", action, a, prob)
                # print()
                score += r
                break

        if n_epi % print_interval == 0 and n_epi != 0:
            loss = model.train_net()
            losses.append(loss)
            avg_losses.append(sum(losses[-print_interval:]) / print_interval)

            print("n_episode :{}, score : {:.1f}, loss : {:.5f}".format(
                n_epi, reward_sum, loss))
            score = 0.0

        scores.append(score)
        rewards.append(sum(scores[-print_interval:]) / print_interval)

        if n_epi == max_episode - 1:
            torch.save(model.state_dict(), "model_save\PPO_cost266_highest_model_final")
            print("Final model saved")
            # Generate training result graph
            fig, ax1 = plt.subplots()
            ax1.plot(scores, linestyle='-')
            ax1.plot(rewards, linestyle='-')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Training Reward')
            ax1.set_title('Training results')
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(avg_losses, linestyle='-', color='red')
            ax2.set_ylabel('Loss')
            plt.show()

    # env.close()


if __name__ == '__main__':
    main()