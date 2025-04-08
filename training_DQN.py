import collections
import csv
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
learning_rate = 0.0001  # 0.0005
gamma = 0.98
buffer_limit = 2500
batch_size = 32

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

        self.path_fc1 = nn.Linear(128, 128)
        self.path_fc2 = nn.Linear(128, 32)

        self.fc1 = nn.Linear(32 * 14 * 14 + 32, 128)    # NSFNET: 14, COST266: 28
        self.fc2 = nn.Linear(128, 3)  # output class

        self.fc_v = nn.Linear(128, 1)

    def forward(self, x, y):
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
        out = F.softmax(self.fc2(z), dim=1)
        critic = self.fc_v(z)

        return out, critic

    def sample_action(self, state, epsilon):
        obs = torch.from_numpy(state['obs']).float()
        obs_path = torch.from_numpy(state['flat_paths']).float()
        # path 구성하기
        # obs_p
        out, _ = self.forward(obs, obs_path)
        coin = random.random()

        candidate_paths = state['paths']

        if coin < epsilon:
            rand_idx = random.randint(0, 2)
            return candidate_paths[rand_idx], rand_idx
        else:
            return candidate_paths[out.argmax().item()], out.argmax().item()


def dqn_train(q, q_target, memory, optimizer):
    loss_lst = []
    for i in range(1):
        s, s_p, a, r, s_prime, s_p_prime, done_mask = memory.sample(batch_size)
        r = r / 100
        s = s.squeeze(1)    # Reshape: s shape: [32, 1, 2, x, x] --> [32, 2, x, x]
        s_prime = s_prime.squeeze(1)

        q_out, _ = q(s, s_p)
        q_a = q_out.gather(1, a)
        max_q_prime, _ = q_target(s_prime, s_p_prime)
        max_q_prime = max_q_prime.max(1)[0].unsqueeze(1)
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
    q_valid = Qnet()
    q_valid.load_state_dict(q.state_dict())
    s, _ = env.reset(0, max_time_step, True)
    done = False

    with torch.no_grad():
        while not done:
            a, out = q_valid.sample_action(s, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            # print("Candidate routing paths: ", s['paths'])
            # print("Action: ", a)
            s = s_prime

            if done:
                score = r
                break

    return score


def main():
    # env = gym.make('CartPole-v1')
    env = QuantumEnvironment(topology_type='NSFNET')
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
            loss = dqn_train(q, q_target, memory, optimizer)
            temp_score = validation(env, q, max_time_step)
            # temp_score = score
            # Best training model save
            if high_score <= temp_score and epsilon < 0.1:
                high_score = temp_score
                torch.save(q.state_dict(), "model_save\cost266_highest_model_best")
                print("Best model saved, score: ", temp_score)
                # _ = validation(env, q, max_time_step)
        # losses.append(1/(loss+0.000001))
        losses.append(loss)
        avg_losses.append(sum(losses[-print_interval:]) / print_interval)

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

            # save the results .csv
            data_files = {"scores.csv": scores, "rewards.csv": rewards, "losses.csv": avg_losses}
            for file_name, data in data_files.items():
                with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for value in data:
                        writer.writerow([value])
                print(f"Data saved to {file_name}")

    # env.close()


if __name__ == '__main__':
    main()