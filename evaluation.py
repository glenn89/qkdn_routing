import gym
import collections
import random
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import QuantumEnvironment


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state, epsilon):
        obs = torch.from_numpy(state['obs']).float()
        out = self.forward(obs)
        coin = random.random()

        paths = nx.shortest_simple_paths(state['graph'], source=1, target=3)
        candidate_paths = []
        candidate_paths.append([0, 1, 3])
        candidate_paths.append([0, 2, 3])
        if coin < epsilon:
            rand_idx = random.randint(0, 1)
            return candidate_paths[rand_idx], rand_idx
        else:
            return candidate_paths[out.argmax().item()], out.argmax().item()


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment()
    q = Qnet()
    q.load_state_dict(torch.load('model_save\highest_model'), strict=False)

    for n_epi in range(1):
        epsilon = 0.0  # Linear annealing from 8% to 1%
        s, _ = env.reset(0, 9)
        done = False
        time_step = 0

        while not done:
            a, out = q.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            s = s_prime

            score = r
            time_step += 1
            if done:
                break

            print("time step :{}, score : {:.1f}, action: {}".format(time_step, score, a))
        score = 0.0
    # env.close()


if __name__ == '__main__':
    main()