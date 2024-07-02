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
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 14 * 14 + 32, 128)
        self.fc2 = nn.Linear(128, 3)  # output class

        self.path_fc1 = nn.Linear(64, 128)
        self.path_fc2 = nn.Linear(128, 32)

    def forward(self, x, y):
        # input state['obs']
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # flatten

        y = self.path_fc1(y)
        y = self.path_fc2(y)
        y = y.view(-1, y.size(0))  # flatten

        z = torch.cat((x, y), dim=1)

        z = self.fc1(z)
        z = nn.ReLU()(z)
        z = self.fc2(z)
        z = F.softmax(z, dim=1)

        # input state['flat_paths']

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
                rand_idx = random.randint(0, 1)
                return candidate_paths[rand_idx], rand_idx
            else:
                return candidate_paths[out.argmax().item()], out.argmax().item()
        except:
            print(candidate_paths)


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='NSFNET')
    q = Qnet()
    q.load_state_dict(torch.load('model_save\highest_model_best'), strict=False)

    for n_epi in range(1):
        epsilon = 0.0  # Linear annealing from 8% to 1%
        s, _ = env.reset(0, 20)
        done = False
        time_step = 0

        while not done:
            a, out = q.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            if not done:
                print("candidate routing paths: ", s['paths'])
            s = s_prime

            score = r
            time_step += 1
            if done:
                break

            print("time step :{}, score : {:.1f}, action_index: {}, action: {}".format(time_step, score, out, a))
            print()
        score = 0.0
    # env.close()


if __name__ == '__main__':
    main()