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

from environment import QuantumEnvironment

# Hyperparameters
learning_rate = 0.0005
gamma = 0.97
buffer_limit = 2000
batch_size = 10

# Wandb config
# wandb.init(project="QKD_rl_routing")

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, done_mask_lst = [], [], [], [], [], [], []

        for episode in mini_batch:
            for transition in episode:
                s, a, r, s_prime, done_mask = transition
                s_lst.append(s['obs'])
                s_p_lst.append(s['flat_paths'])
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime['obs'])
                s_p_prime_lst.append(s_prime['flat_paths'])
                done_mask_lst.append([done_mask])

        s_lst, s_p_lst, a_lst, r_lst, s_prime_lst, s_p_prime_lst, done_mask_lst = np.array(s_lst), np.array(s_p_lst), \
        np.array(a_lst), np.array(r_lst), np.array(s_prime_lst), np.array(s_p_prime_lst), np.array(done_mask_lst)

        return (torch.tensor(s_lst, dtype=torch.float), torch.tensor(s_p_lst, dtype=torch.float), \
            torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(s_p_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst))

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.path_fc1 = nn.Linear(64, 128)
        self.path_fc2 = nn.Linear(128, 32)

        self.fc1 = nn.Linear(32 * 28 * 28 + 32, 128)
        self.fc2 = nn.Linear(128, 3)  # output class

    def forward(self, x, y):
        # input state['obs']
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # flatten

        # input state['flat_paths']
        y = self.path_fc1(y)
        y = self.path_fc2(y)
        y = y.view(y.size(0), -1)  # flatten

        z = torch.cat((x, y), dim=1)  # Concatenate cnn and fc results

        z = self.fc1(z)
        z = nn.ReLU()(z)
        z = self.fc2(z)
        z = F.softmax(z, dim=1)

        return z

    def sample_action(self, state, epsilon):
        try:
            obs = torch.from_numpy(state['obs']).float()
            obs_path = torch.from_numpy(state['flat_paths']).float()
            # path 구성하기
            # obs_p
            out = self.forward(obs, obs_path)
            coin = random.random()

            candidate_paths = state['paths']

            if coin < epsilon:
                rand_idx = random.randint(0, 2)
                return candidate_paths[rand_idx], rand_idx
            else:
                return candidate_paths[out.argmax().item()], out.argmax().item()
        except:
            print(candidate_paths)


def train(q, q_target, memory, optimizer):
    loss_lst = []
    for i in range(1):
        s, s_p, a, r, s_prime, s_p_prime, done_mask = memory.sample(batch_size)
        s = s.squeeze(1)    # Reshape: s shape: [32, 1, 2, x, x] --> [32, 2, x, x]
        s_prime = s_prime.squeeze(1)

        q_out = q(s, s_p)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime, s_p_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_lst.append(loss.detach().numpy())

    return np.mean(loss_lst)

def validation(env, q, max_time_step):
    epsilon = 0.0  # Linear annealing from 8% to 1%
    score = 0.0
    s, _ = env.reset(0, max_time_step, True)
    done = False

    while not done:
        a, out = q.sample_action(s, epsilon)
        s_prime, r, done, truncated, info = env.step(a)
        s = s_prime

        if done:
            score = r
            break

    return score

def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='COST266')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    episode_memory = []

    max_episode = 10000
    max_time_step = 20
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
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(max_episode):
        epsilon = max(0.01, 0.9 - 0.025 * (n_epi / 200))  # Linear annealing from 90% to 1%
        s, _ = env.reset(0, max_time_step, True)
        done = False

        while not done:
            a, out = q.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            transition = (s, out, r, s_prime, done_mask)
            episode_memory.append(transition)
            s = s_prime

            if done:
                memory.put(episode_memory)
                score += r
                episode_memory = []
                break

        if memory.size() > 200:
            loss = train(q, q_target, memory, optimizer)
        losses.append(loss)
        avg_losses.append(sum(losses[-print_interval:]) / print_interval)

        # temp_score = validation(env, q, max_time_step)
        score
        if high_score <= score:
            high_score = score
            torch.save(q.state_dict(), "model_save\cost266_highest_model_final")
            print("Best model saved, score: ", score)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, loss : {:.5f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score, loss, memory.size(), epsilon * 100))

        scores.append(score)
        rewards.append(sum(scores[-print_interval:]) / print_interval)
        score = 0.0

        if n_epi == max_episode - 1:
            torch.save(q.state_dict(), "model_save\cost266_highest_model_final")
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