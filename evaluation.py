import collections
import random
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from environment import QuantumEnvironment


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28 + 32, 128)
        self.fc2 = nn.Linear(128, 3)  # output class

        self.path_fc1 = nn.Linear(128, 128)
        self.path_fc2 = nn.Linear(128, 32)

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
                rand_idx = random.randint(0, 1)
                return candidate_paths[rand_idx], rand_idx
            else:
                return candidate_paths[out.argmax().item()], out.argmax().item()
        except:
            print(candidate_paths)


def main():
    seed = 0
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='COST266')
    q = Qnet()
    q.load_state_dict(torch.load('model_save\cost266_highest_model_final'), strict=False)

    (shortest_reward, shortest_average_reward, shortest_average_session_blocking,
     shortest_average_total_generation_keys, shortest_average_remaining_keys, shortest_average_used_keys) = 0, 0, 0, 0, 0, 0

    for n_epi in range(1):
        epsilon = 0.0  # Linear annealing from 8% to 1%
        s, _ = env.reset(seed, 20, True)
        done = False
        time_step = 0

        while not done:
            a, out = q.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            print("candidate routing paths: ", s['paths'])
            s = s_prime

            score = r
            shortest_average_reward = r
            shortest_average_session_blocking = info['session_blocking']
            shortest_average_total_generation_keys = info['total_generation_keys']
            shortest_average_remaining_keys = info['remaining_keys']
            shortest_average_used_keys = info['used_keys']
            print("time step :{}, score : {:.1f}, action_index: {}, action: {}".format(time_step, score, out, a))
            print()

            time_step += 1
            if done:
                break

        score = 0.0
    # env.close()
    print("Simulation information")
    print()
    print("Average Results:")
    print(
        f"{'Metric':<20}{'Success':<10}{'Session Blocking':<20}{'Total generation keys':<25}{'Used keys':<20}{'Used percentage':<10}")
    print(
        f"{'RL-based':<20}{shortest_average_reward:<10}{shortest_average_session_blocking:<20}{shortest_average_total_generation_keys:<25}{shortest_average_used_keys:<20}{(shortest_average_used_keys / shortest_average_total_generation_keys) * 100:<4.2f}%")


if __name__ == '__main__':
    main()