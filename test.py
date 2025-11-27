import matplotlib.pyplot as plt
import numpy as np

algos = ["RL (Proposed)", "Min-path", "FIFO", "Random"]
avg_reward = [4.0, 3.3, 3.68, 3.04]
std_reward = [6.18, 7.73, 7.65, 7.60]
expired_keys = [695, 926, 842, 574]

x = np.arange(len(algos))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))

# 평균 보상 막대 그래프
ax1.bar(x, avg_reward, width, capsize=5, label="Avg allocation", alpha=0.8)
ax1.set_ylabel("Average allocation", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(algos, fontsize=12)
ax1.set_ylim(0.0, 4.5)
ax1.legend(loc="upper left", fontsize=12)

# 만료된 키 수 선 그래프
ax2 = ax1.twinx()
ax2.plot(x, expired_keys, color="red", marker="o", label="Expired Keys")
ax2.set_ylabel("Total Expired Keys", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

plt.tight_layout()
plt.show()