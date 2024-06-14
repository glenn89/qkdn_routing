import gym
import collections
import random
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import QuantumEnvironment

# Hyperparameters
learning_rate = 0.0005
gamma = 0.97
buffer_limit = 5000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s['obs'])
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime['obs'])
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 3)  # output class

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Softmax()(x)

        return x

    def sample_action(self, state, epsilon):
        obs = torch.from_numpy(state['obs']).float()
        obs_p =
        out = self.forward(obs)
        coin = random.random()

        candidate_paths = state['paths']
        if coin < epsilon:
            rand_idx = random.randint(0, 1)
            return candidate_paths[rand_idx], rand_idx
        else:
            return candidate_paths[out.argmax().item()], out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s = s.squeeze(1)    # Reshape: s shape: [32, 1, 2, x, x] --> [32, 2, x, x]
        s_prime = s_prime.squeeze(1)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='NSFNET')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    max_episode = 1500
    print_interval = 20
    score = 0.0
    high_score = 0.0
    scores = []
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(max_episode):
        epsilon = max(0.01, 0.9 - 0.2 * (n_epi / 200))  # Linear annealing from 90% to 1%
        s, _ = env.reset(0, 9)
        done = False

        while not done:
            a, out = q.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, out, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            if high_score <= score:
                high_score = score
                torch.save(q.state_dict(), "model_save\highest_model_best")
                print("Best model saved")
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))

        scores.append(score)
        score = 0.0

        if n_epi == max_episode - 1:
            torch.save(q.state_dict(), "model_save\highest_model_final")
            print("Final model saved")
            # Generate training result graph
            plt.plot(scores, linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel('Training Reward')
            plt.title('Training results')
            plt.grid(True)
            plt.show()

    # env.close()


if __name__ == '__main__':
    main()