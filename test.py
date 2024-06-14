# import random
# import numpy as np
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
#         pareto_counts = np.random.pareto(alpha, size) * 30
#         print(pareto_counts)
#         print(np.mean(pareto_counts), max(pareto_counts), min(pareto_counts))
#         return pareto_counts.astype(int)
#
#
#     def simulate_packet_generation(steps, alpha):
#         # 각 스텝에서 생성되는 패킷 수 생성
#         packet_counts = generate_packet_counts(steps, alpha)
#
#         # 패킷 수 시각화
#         plt.plot(range(1, steps + 1), packet_counts, marker='o', linestyle='-')
#         plt.xlabel('Step')
#         plt.ylabel('Consume size')
#         plt.title('Simulation of Request Generation')
#         plt.grid(True)
#         plt.show()
#
#
#     # 시뮬레이션 파라미터 설정
#     steps = 1000  # 시뮬레이션 스텝 수
#     alpha = 2     # 파레토 분포의 모수
#
#     # 패킷 생성 시뮬레이션
#     simulate_packet_generation(steps, alpha)

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 그래프 생성
G = nx.Graph()
edges = [
    ('A', 'B', 0.1), ('A', 'C', 0.2), ('B', 'C', 0.3),
    ('B', 'D', 0.4), ('C', 'D', 0.5), ('C', 'E', 0.6), ('D', 'E', 0.7)
]
G.add_weighted_edges_from(edges)

# 1채널: 인접 행렬 생성 (NumPy ndarray 형식)
adj_matrix_np = nx.to_numpy_array(G)
adj_matrix_np = np.array(adj_matrix_np)

# 2채널: 가중치 행렬 생성
weight_matrix_np = np.zeros_like(adj_matrix_np)
for u, v, data in G.edges(data=True):
    i, j = list(G.nodes).index(u), list(G.nodes).index(v)
    weight_matrix_np[i, j] = data['weight']
    weight_matrix_np[j, i] = data['weight']  # 무방향 그래프일 경우 대칭적으로 채우기

# 2채널 상태 구성
state_np = np.stack([adj_matrix_np, weight_matrix_np], axis=0)  # (2, H, W) 형식으로 변환
state_np = state_np[np.newaxis, :]  # 배치 차원을 추가하여 (1, 2, H, W) 형식으로 변환

print(state_np.shape)

# PyTorch tensor로 변환
state_tensor = torch.tensor(state_np, dtype=torch.float32)

print(f"State tensor shape: {state_tensor.shape}")
print(state_tensor.size(), state_tensor)
