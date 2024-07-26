import random
import numpy as np

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # results = np.random.binomial(500, 0.08)
    # print(results)
    #
    # # Set the random seed for reproducibility (optional)
    # np.random.seed(42)
    #
    # # Number of time steps
    # num_steps = 100
    #
    # # Initial mean and standard deviation
    # initial_mean = 5
    # initial_std_dev = 3
    #
    # # Lists to store qber values and time steps
    # time_steps = np.arange(num_steps)
    # qber_values = []
    #
    # # Generate qber values over multiple time steps
    # for step in range(num_steps):
    #     # Generate a random float from a normal distribution with current mean and standard deviation
    #     qber = np.random.normal(loc=initial_mean, scale=initial_std_dev)
    #
    #     # Clip the value to ensure it falls within a valid range (e.g., between 0 and 10)
    #     qber = np.clip(qber, 0, 10)
    #
    #     # Append the qber value to the list
    #     qber_values.append(qber)
    #
    #     # Adjust mean and standard deviation for the next time step (for demonstration purposes)
    #     # initial_mean += 0.1
    #     initial_std_dev -= 0.02
    #
    #     if initial_std_dev < 0:
    #         initial_std_dev = 0.
    #
    # # Plot the qber values over time
    # plt.plot(time_steps, qber_values, label='QBER')
    # plt.xlabel('Time Steps')
    # plt.ylabel('QBER')
    # plt.title('QBER Over Time')
    # plt.legend()
    # plt.show()

    def generate_packet_counts(size, alpha):
        # 파레토 분포를 따르는 패킷 수 생성
        pareto_counts = 30 - np.random.pareto(alpha, size) * 30
        print(pareto_counts)
        print(np.mean(pareto_counts), max(pareto_counts), min(pareto_counts))
        for i in range(len(pareto_counts)):
            if pareto_counts[i] < 0:
                pareto_counts[i] = 0

        return pareto_counts.astype(int)


    def simulate_packet_generation(steps, alpha):
        # 각 스텝에서 생성되는 패킷 수 생성
        packet_counts = generate_packet_counts(steps, alpha)

        # 패킷 수 시각화
        plt.plot(range(1, steps + 1), packet_counts, marker='o', linestyle='-')
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('The number of quantum key', fontsize=15)
        plt.title('Simulation of Quantum key Generation', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True)
        plt.show()


    # 시뮬레이션 파라미터 설정
    steps = 100  # 시뮬레이션 스텝 수
    alpha = 12     # 파레토 분포의 모수

    # 패킷 생성 시뮬레이션
    simulate_packet_generation(steps, alpha)

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
