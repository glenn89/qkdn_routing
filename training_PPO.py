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
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
batch_size = 32

# Wandb config
# wandb.init(project="QKD_rl_routing")


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        # Topology embedding conv
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Path embedding
        self.path_fc1 = nn.Linear(128, 128)
        self.path_fc2 = nn.Linear(128, 32)

        self.fc1 = nn.Linear(32 * 14 * 14 + 32, 128)    # NSFNET: 14, COST266: 28
        self.fc2 = nn.Linear(128, 3)  # output class

        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, y, softmax_dim=0):
        # input state['obs']
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # flatten

        # input state['flat_paths']
        y = self.path_fc1(y)
        y = nn.ReLU()(y)
        y = self.path_fc2(y)
        y = nn.ReLU()(y)
        y = y.view(y.size(0), -1)  # flatten

        z = torch.cat((x, y), dim=1)  # Concatenate cnn and fc results
        z = self.fc1(z)
        z = nn.ReLU()(z)
        z = self.fc2(z)
        prob = F.softmax(z, dim=softmax_dim)

        return prob

    def v(self, x, y):
        # input state['obs']
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # flatten

        # input state['flat_paths']
        y = self.path_fc1(y)
        y = nn.ReLU()(y)
        y = self.path_fc2(y)
        y = nn.ReLU()(y)
        y = y.view(y.size(0), -1)  # flatten

        z = torch.cat((x, y), dim=1)  # Concatenate cnn and fc results

        z = self.fc1(z)
        z = nn.ReLU()(z)
        v = self.fc_v(z)

        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done_mask = transition
            s_lst.append(np.squeeze(s['obs'], axis=0))
            s_p_lst.append(np.squeeze(s['flat_paths'], axis=0))
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(np.squeeze(s_prime['obs'], axis=0))
            s_p_prime_lst.append(np.squeeze(s_prime['flat_paths'], axis=0))
            prob_a_lst.append([prob_a])
            done_mask_lst.append([done_mask])

        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, prob_a_lst, done_mask_lst = \
            (np.array(s_lst), np.array(s_p_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst),
             np.array(s_p_prime_lst), np.array(prob_a_lst), np.array(done_mask_lst))
        self.data = []

        return (torch.tensor(s_lst, dtype=torch.float), torch.tensor(s_p_lst, dtype=torch.float),
                torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst),
                torch.tensor(s_prime_lst, dtype=torch.float),
                torch.tensor(s_p_prime_lst, dtype=torch.float), torch.tensor(prob_a_lst), torch.tensor(done_mask_lst))

    def sample_action(self, state):
        obs = torch.from_numpy(state['obs']).float()
        obs_path = torch.from_numpy(state['flat_paths']).float()

        prob = self.pi(obs, obs_path)
        m = Categorical(prob)
        action = m.sample().item()
        candidate_paths = state['paths']

        return candidate_paths[action], action, prob

    def train_net(self):
        loss_lst = []
        s, s_p, a, r, s_prime, s_p_prime, prob_a, done_mask = self.make_batch()

        for i in range(1):
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

            pi = self.pi(s, s_p, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, s_p), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().numpy())

        return np.mean(loss_lst)


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='NSFNET')
    model = PPO()
    episode_memory = []

    max_episode = 10000
    max_time_step = 100
    print_interval = 20
    score = 0.0
    temp_score = 0.0
    loss = 0.0
    reward = 0.0
    rewards = []
    high_score = 0.0
    scores = []
    losses = []
    avg_losses = []

    for n_epi in range(max_episode):
        s, _ = env.reset(0, max_time_step, True)
        done = False

        while not done:
            a, action, prob = model.sample_action(s)
            prob = prob.squeeze()
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            transition = (s, action, r, s_prime, prob[action].item(), done_mask)
            model.put_data(transition)
            s = s_prime

            if done:
                score += r
                break

        loss = model.train_net()

        losses.append(loss)
        avg_losses.append(sum(losses[-print_interval:]) / print_interval)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("n_episode :{}, score : {:.1f}, loss : {:.5f}".format(
                n_epi, score/print_interval, loss))
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