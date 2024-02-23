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
        pareto_counts = np.random.pareto(alpha, size) + 2
        print(pareto_counts)
        return pareto_counts.astype(int)


    def simulate_packet_generation(steps, alpha):
        # 각 스텝에서 생성되는 패킷 수 생성
        packet_counts = generate_packet_counts(steps, alpha)

        # 패킷 수 시각화
        plt.plot(range(1, steps + 1), packet_counts, marker='o', linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('Packet Count')
        plt.title('Simulation of Packet Generation')
        plt.grid(True)
        plt.show()


    # 시뮬레이션 파라미터 설정
    steps = 100  # 시뮬레이션 스텝 수
    alpha = 2  # 파레토 분포의 모수

    # 패킷 생성 시뮬레이션
    simulate_packet_generation(steps, alpha)