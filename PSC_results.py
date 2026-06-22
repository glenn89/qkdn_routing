import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
methods = ["FIFO", "SPF", "RL"]

served_ratio = np.array([47.84, 68.47, 70.21])
overflow_ratio = np.array([43.28, 3.83, 4.34])
expiration_ratio = np.array([8.88, 27.71, 25.45])

jfi = np.array([0.9965, 0.7680, 0.8544])

# =========================
# Plot setting
# =========================
x = np.arange(len(methods))
bar_width = 0.24

# width_ratios=[2.2, 1.0]
# 왼쪽 Ratio subplot을 오른쪽 JFI subplot보다 약 2.2배 넓게 설정
fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.5, 3.6),
    dpi=300,
    gridspec_kw={"width_ratios": [2.2, 1.0]}
)

# ==================================================
# (a) Ratio comparison
# ==================================================
ax = axes[0]

bars1 = ax.bar(
    x - bar_width,
    served_ratio,
    bar_width,
    label="Served Ratio"
)

bars2 = ax.bar(
    x,
    overflow_ratio,
    bar_width,
    label="Overflow Ratio"
)

bars3 = ax.bar(
    x + bar_width,
    expiration_ratio,
    bar_width,
    label="Expiration Ratio"
)

# ax.set_xlabel("Scheduling Method", fontsize=10)
ax.set_ylabel("Ratio (%)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 90)
ax.set_yticks(np.arange(0, 91, 10))

ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)

# Value labels
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.legend(
    loc="best",
    fontsize=10,
    frameon=True
)

# ==================================================
# (b) JFI comparison
# ==================================================
ax = axes[1]

bars = ax.bar(
    methods,
    jfi,
    width=0.45,
    label="JFI",
    color="#9467bd"
)

# ax.set_xlabel("Scheduling Method", fontsize=10)
ax.set_ylabel("Jain's Fairness Index (JFI)", fontsize=10)
ax.set_ylim(0.70, 1.05)
ax.set_yticks(np.arange(0.50, 1.05, 0.1))

ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)

# Value labels
for bar, value in zip(bars, jfi):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.006,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

# =========================
# Style adjustment
# =========================
for ax in axes:
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

fig.tight_layout(w_pad=1.6)

# =========================
# Save figure
# =========================
# plt.savefig("psc2026_ratio_jfi_subplot_narrow_jfi.png", dpi=300, bbox_inches="tight")
# plt.savefig("psc2026_ratio_jfi_subplot_narrow_jfi.pdf", bbox_inches="tight")
# plt.savefig("psc2026_ratio_jfi_subplot_narrow_jfi.eps", bbox_inches="tight")

plt.show()