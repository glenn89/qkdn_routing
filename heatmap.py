import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. CSV file path
# ==============================
# csv_path = "results/COST266/RL00/COST266_RL00_pair_queueing_delay.csv"
# csv_path = "results/COST266/RL05/COST266_RL05_pair_queueing_delay.csv"
# csv_path = "results/COST266/RL10/COST266_RL10_pair_queueing_delay.csv"
# csv_path = "results/COST266/FIFO/COST266_FIFO_pair_queueing_delay.csv"
csv_path = "results/COST266/MIN/COST266_MIN_pair_queueing_delay.csv"

# ==============================
# 2. Algorithm name
# ==============================
algorithm_name = "RL (α=0.0)"
# FIFO, SPF, RL (α=0.0), RL (α=0.5), RL (α=1.0) 등으로 변경 가능

# ==============================
# 3. Load dataset
# ==============================
df = pd.read_csv(csv_path)

print(df.head())
print(df.columns)
print(df.describe())

# ==============================
# 4. Create pair-wise delay matrix
# ==============================
num_nodes = 28  # COST266 topology

delay_matrix = np.full((num_nodes, num_nodes), np.nan)

for _, row in df.iterrows():
    src = int(row["src"])
    dst = int(row["dst"])
    avg_delay = float(row["avg_delay"])

    delay_matrix[src, dst] = avg_delay
    delay_matrix[dst, src] = avg_delay  # symmetric matrix

# Remove self-pair values
np.fill_diagonal(delay_matrix, np.nan)

# ==============================
# 5. Calculate overall mean
# ==============================
overall_mean = df["avg_delay"].mean()

# ==============================
# 6. Plot heatmap
# ==============================
plt.figure(figsize=(7.2, 6.2))

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="white")  # NaN values, including diagonal

im = plt.imshow(
    delay_matrix,
    cmap=cmap,
    vmin=0.0,
    vmax=5.0,
    interpolation="nearest"
)

# plt.title(
#     f"COST266 pair-wise queueing delay heatmap: {algorithm_name}\n"
#     f"waiting time limit = 5 time steps, mean = {overall_mean:.3f}",
#     fontsize=13
# )

plt.xlabel("Destination node", fontsize=12)
plt.ylabel("Source node", fontsize=12)

ticks = [0, 5, 10, 15, 20, 25, 27]
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)

cbar = plt.colorbar(im)
cbar.set_label("Average queueing delay", fontsize=12)
cbar.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

plt.tight_layout()
plt.show()