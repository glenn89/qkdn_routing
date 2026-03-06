import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== 파일 목록 (seed 5개) ======
groups = {
    "SetTransformer (α=0.0)": [
        "results/NSFNET/RL_00/RL_00_reward_log_1.csv",
        "results/NSFNET/RL_00/RL_00_reward_log_2.csv",
        "results/NSFNET/RL_00/RL_00_reward_log_3.csv",
    ],
    "SetTransformer (α=0.5)": [
        "results/NSFNET/RL_05/RL_05_reward_log_1.csv",
        "results/NSFNET/RL_05/RL_05_reward_log_2.csv",
        "results/NSFNET/RL_05/RL_05_reward_log_3.csv",
    ],
    "SetTransformer (α=1.0)": [
        "results/NSFNET/RL_10/RL_10_reward_log_1.csv",
        "results/NSFNET/RL_10/RL_10_reward_log_2.csv",
        "results/NSFNET/RL_10/RL_10_reward_log_3.csv",
    ],
    "Minimize path": [
        "results/NSFNET/Minpath/Minpath_reward_log_1.csv",
        "results/NSFNET/Minpath/Minpath_reward_log_2.csv",
        "results/NSFNET/Minpath/Minpath_reward_log_3.csv",
    ],
    "FIFO": [
        "results/NSFNET/FIFO/FIFO_reward_log_1.csv",
        "results/NSFNET/FIFO/FIFO_reward_log_2.csv",
        "results/NSFNET/FIFO/FIFO_reward_log_3.csv",
    ],
}

def load_reward_series(fp: str) -> np.ndarray:
    """
    CSV가
    - 1열(헤더 유/무) 이거나
    - 'reward' 컬럼이 있는 경우
    모두 처리 + 숫자 강제 변환
    """
    try:
        df = pd.read_csv(fp)
        if "reward" in df.columns:
            s = df["reward"]
        else:
            s = df.iloc[:, 0]
    except Exception:
        s = pd.read_csv(fp, header=None).iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna().reset_index(drop=True)
    return s.to_numpy()

def stack_cumsums(file_list):
    runs = [np.cumsum(load_reward_series(p)) for p in file_list]
    # seed별 길이가 다르면 shortest 길이에 맞춰서 truncate
    T = min(len(r) for r in runs)
    mat = np.vstack([r[:T] for r in runs])  # (n_seeds, T)
    return mat

# ====== 95% CI 계수 (Student-t, n=5 -> df=4) ======
t_crit = 2.776  # t_{0.975, df=4}

# plt.figure(figsize=(8.5, 5))

fig, axes = plt.subplots(1, 1, figsize=(8, 5))
for name, files in groups.items():
    mat = stack_cumsums(files)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)  # sample std
    n = mat.shape[0]
    se = std / np.sqrt(n)
    ci = t_crit * se

    # Overall Graph
    x = np.arange(len(mean))
    # axes.fill_between(x, mean - ci, mean + ci, alpha=0.20, linewidth=0)  # 음영
    # axes.plot(x, mean, linewidth=2.2, label=name)  # 평균 선

    # # Zoomed Graph(Top 20%)
    start = int(len(mean) * 0.8)
    axes.fill_between(
        x[start:],
        mean[start:] - ci[start:],
        mean[start:] + ci[start:],
        alpha=0.2
    )
    axes.plot(x[start:], mean[start:], label=name)

# axes.set_xlabel("Time slot", fontsize=15)
# axes.set_ylabel("Total Served Requests", fontsize=15)
# axes.tick_params(labelsize=15)
# axes.legend(fontsize=14)
# axes.grid(alpha=0.25)

axes.set_xlabel("Time slot", fontsize=15)
axes.set_ylabel("Total Served Requests", fontsize=15)
axes.tick_params(labelsize=15)
axes.legend(fontsize=14)
axes.grid(alpha=0.25)

plt.tight_layout()
plt.show()