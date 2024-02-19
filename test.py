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

    list = [1, 2, 3, 4, 5, 6, 7, 8]
    list -= 1
    print(list)