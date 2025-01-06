import random
import numpy as np
#
# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # results = np.random.binomial(500, 0.08)
#     # print(results)
#     #
#     # # Set the random seed for reproducibility (optional)
#     # np.random.seed(42)
#     #
#     # # Number of time steps
#     # num_steps = 100
#     #
#     # # Initial mean and standard deviation
#     # initial_mean = 5
#     # initial_std_dev = 3
#     #
#     # # Lists to store qber values and time steps
#     # time_steps = np.arange(num_steps)
#     # qber_values = []
#     #
#     # # Generate qber values over multiple time steps
#     # for step in range(num_steps):
#     #     # Generate a random float from a normal distribution with current mean and standard deviation
#     #     qber = np.random.normal(loc=initial_mean, scale=initial_std_dev)
#     #
#     #     # Clip the value to ensure it falls within a valid range (e.g., between 0 and 10)
#     #     qber = np.clip(qber, 0, 10)
#     #
#     #     # Append the qber value to the list
#     #     qber_values.append(qber)
#     #
#     #     # Adjust mean and standard deviation for the next time step (for demonstration purposes)
#     #     # initial_mean += 0.1
#     #     initial_std_dev -= 0.02
#     #
#     #     if initial_std_dev < 0:
#     #         initial_std_dev = 0.
#     #
#     # # Plot the qber values over time
#     # plt.plot(time_steps, qber_values, label='QBER')
#     # plt.xlabel('Time Steps')
#     # plt.ylabel('QBER')
#     # plt.title('QBER Over Time')
#     # plt.legend()
#     # plt.show()
#
#     def generate_packet_counts(size, alpha):
#         # 파레토 분포를 따르는 패킷 수 생성
#         pareto_counts = 30 - np.random.pareto(alpha, size) * 30
#         print(pareto_counts)
#         print(np.mean(pareto_counts), max(pareto_counts), min(pareto_counts))
#         for i in range(len(pareto_counts)):
#             if pareto_counts[i] < 0:
#                 pareto_counts[i] = 0
#
#         return pareto_counts.astype(int)
#
#
#     def simulate_packet_generation(steps, alpha):
#         # 각 스텝에서 생성되는 패킷 수 생성
#         packet_counts = generate_packet_counts(steps, alpha)
#
#         # 패킷 수 시각화
#         plt.plot(range(1, steps + 1), packet_counts, marker='o', linestyle='-')
#         plt.xlabel('Step', fontsize=15)
#         plt.ylabel('The number of quantum key', fontsize=15)
#         plt.title('Simulation of Quantum key Generation', fontsize=15)
#         plt.xticks(fontsize=15)
#         plt.yticks(fontsize=15)
#         plt.grid(True)
#         plt.show()
#
#
#     # 시뮬레이션 파라미터 설정
#     steps = 100  # 시뮬레이션 스텝 수
#     alpha = 12     # 파레토 분포의 모수
#
#     # 패킷 생성 시뮬레이션
#     simulate_packet_generation(steps, alpha)

#
# cost266_desc = {
#     'NAME': "COST266",
#
#     'QKD_NODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
#                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#
#     'QKD_TOPOLOGY': [
#         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
#     ],
#
#     'NUM_QKD_NODE': 28,
#     'NUM_QKD_LINK': 41,
#
#     'NUM_DIRECT_KEY_POOL': 82,
#     'NUM_INDIRECT_KEY_POOL': 674,
#
#
#     # QKD setting
#     'KEY_AVERAGE_RATE': 5,  # 개수
#     'INITIAL_LIFE_TIME': 5,  # step
#     'LIFE_TIME': 7,  # step
#
#     # User setting
#     'PATH_SEQUENCE': 'KEY',  #
#     'NUM_INITIAL_DIRECT_KEY': 15,
#     'NUM_INITIAL_INDIRECT_KEY': 3,
#     'NUM_DIRECT_KEY_THRESHOLD': 3,
#
#     # ENV setting
#     'MAX_DIRECT_KEY': 5,
#     'MAX_INDIRECT_KEY': 5,
#     'REWARD_INDICATOR': 1.0,
#     'VERBOSE': 1,
#     'MODE': 'REPLACE',  # REPLACE(default), LIMIT
#
# }
#
#
# def check_topology(config):
#     import numpy as np
#
#     qkd_topo = config["QKD_TOPOLOGY"]
#     qkd_node = config["NUM_QKD_NODE"]
#     qkd_link = config["NUM_QKD_LINK"]
#
#     user_topology = np.ones(shape=(qkd_node, qkd_node))
#     user_topo = user_topology - np.diag(np.diag(user_topology))
#
#     flat_qkd_topo = sum(qkd_topo, [])
#     flat_user_topo = sum(user_topo.tolist(), [])
#
#     direct_key_pool = config["NUM_DIRECT_KEY_POOL"]
#     indirect_key_pool = config["NUM_INDIRECT_KEY_POOL"]
#
#     check_list = {}
#
#     # check number of qkd node
#     if qkd_node == len(qkd_topo):
#         check_list['NUM_QKD_NODE'] = True
#     else:
#         check_list['NUM_QKD_NODE'] = False
#
#     # check number of qkd link
#     if qkd_link == int(sum(flat_qkd_topo)/2):
#         check_list['NUM_QKD_LINK'] = True
#     else:
#         check_list['NUM_QKD_LINK'] = False
#
#     # check number of direct key pool
#     if direct_key_pool == int(sum(flat_qkd_topo)):
#         check_list['NUM_DIRECT_KEY_POOL'] = True
#     else:
#         check_list['NUM_DIRECT_KEY_POOL'] = False
#
#     # check number of indirect key pool
#     if indirect_key_pool == int(sum([1 for user_link, qkd_link in zip(flat_user_topo, flat_qkd_topo) if (user_link - qkd_link) > 0])):
#         check_list['NUM_INDIRECT_KEY_POOL'] = True
#     else:
#         check_list['NUM_INDIRECT_KEY_POOL'] = False
#
#     return (check_list, False in check_list.values())
#
#
# if __name__ == "__main__":
#     # False is not error
#     check_list, check_error = check_topology(cost266_desc)
#     if check_error:
#         print("!"*10 + "Find Error" + "!"*10)
#         print("check_list: ", check_list)
#     else:
#         print("@"*10 + "No Error" + "@"*10)
#
#         import networkx as nx
#         import numpy as np
#         import matplotlib.pyplot as plt
#
#         graph = nx.Graph()
#         graph.add_nodes_from(cost266_desc['QKD_NODES'])
#
#         edges = []
#
#         node_1 = np.where(np.array(cost266_desc['QKD_TOPOLOGY']) == 1)[0]
#         node_2 = np.where(np.array(cost266_desc['QKD_TOPOLOGY']) == 1)[1]
#
#         for (i, j) in zip(node_1, node_2):
#             edges.append((i, j))
#
#         graph.add_edges_from(edges)
#
#         pos = {0:[1, 18], 1:[0, 16], 2:[2, 13], 3:[5, 15], 4:[9, 16], 5:[10, 20], 6:[16, 19], 7:[11, 17], 8:[12, 14],
#                9:[19, 14], 10:[5, 12], 11:[7, 11], 12:[13, 11], 13:[3, 10], 14:[6, 10], 15:[10, 9], 16:[15, 9],
#                17:[17, 8], 18:[7, 7], 19:[2, 4], 20:[5, 6], 21:[8, 5], 22:[14, 6], 23:[18, 4],
#                24:[1, 2], 25:[4, 3], 26:[12, 3], 27:[17, 1]}
#
#         nx.draw(graph, pos)
#         plt.show()


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()