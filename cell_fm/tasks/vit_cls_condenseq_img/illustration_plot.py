import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from glob import glob

def compute_gap_with_points(y: np.ndarray):
    """
    返回:
      m_gap, max_idx, min_idx, y_max, y_min
    其中 min_idx 是 max_idx 之后的最小值点索引。
    """
    y = np.asarray(y, dtype=float)
    max_idx = int(np.argmax(y))
    y_max = float(y[max_idx])

    if max_idx == len(y) - 1:
        # max 在最后一个点：没有“之后”的最小值点
        return 0.0, max_idx, max_idx, y_max, y_max

    tail = y[max_idx + 1:]
    rel_min_idx = int(np.argmin(tail))
    min_idx = max_idx + 1 + rel_min_idx
    y_min = float(y[min_idx])

    return (y_max - y_min), max_idx, min_idx, y_max, y_min

def moving_average(y, window: int):
    y = np.asarray(y, dtype=float)
    window = int(max(1, min(window, len(y))))
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

protein_name = "hnRNPA1-2"
data_roots = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_single/{protein_name}/protein_{protein_name}/'))

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
})

for root in data_roots:
    data_root = Path(root)
    csv_path = data_root / "pred_vs_intensity.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    needed = {"protein_intensity_level", "predicted_class"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Bad columns in {csv_path.name}: {df.columns.tolist()}")
        continue

    df = df[["protein_intensity_level", "predicted_class"]].copy()
    df["protein_intensity_level"] = pd.to_numeric(df["protein_intensity_level"], errors="coerce")
    df["predicted_class"] = pd.to_numeric(df["predicted_class"], errors="coerce")
    df = df.dropna()

    if len(df) < 5:
        print(f"[WARN] Too few rows after cleaning: {csv_path}")
        continue

    df = df.sort_values("protein_intensity_level", kind="mergesort")

    x = df["protein_intensity_level"].to_numpy()
    y = df["predicted_class"].to_numpy()

    uniq = np.unique(y)
    if uniq.size > 10 or (uniq.min() < 0) or (uniq.max() > 1):
        print(f"[WARN] y looks non-binary / non-probability in {csv_path}: min={y.min():.3g}, max={y.max():.3g}")

    rng = np.random.default_rng(seed=42)
    
    # uniform jitter
    jitter_strength = 0.01
    y_jittered = y + rng.uniform(-jitter_strength, jitter_strength, size=y.shape)
    plt.figure(figsize=(6, 4))
    plt.scatter(
        x,
        y_jittered,
        alpha=0.08,
        s=20,
        c=y,
        cmap="coolwarm",
        edgecolors="none"
    )

    plt.xlabel("Protein Expression Level")
    plt.ylabel("Predicted Class")
    plt.title("Predicted Class vs Protein Expression Level")

    # class 0: non-condensate, class 1: condensate
    class_labels = {0: "non-condensate", 1: "condensate"}

    unique_classes = np.unique(y)
    plt.yticks(unique_classes, [class_labels.get(c, str(c)) for c in unique_classes], rotation=45)

    # save as svg, 0 padding, high dpi
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(data_root / 'pred_vs_intensity.svg', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    window_size = 512
    y_ma = moving_average(y, window_size)

    # 计算 m_gap 以及对应点
    m_gap, max_idx, min_idx, y_max, y_min = compute_gap_with_points(y_ma)

    fig, ax = plt.subplots(figsize=(10, 6))

    # uniform jitter
    jitter_strength = 0.01
    y_jittered = y + rng.uniform(-jitter_strength, jitter_strength, size=y.shape)

    # 分组：0 = no condensate（蓝），1 = condensate（红）
    mask0 = (y == 0)
    mask1 = (y == 1)

    ax.scatter(
        x[mask0],
        y_jittered[mask0],
        alpha=0.1,
        s=20,
        c="tab:blue",
        edgecolors="none",
        label="No condensate"
    )
    ax.scatter(
        x[mask1],
        y_jittered[mask1],
        alpha=0.1,
        s=20,
        c="tab:red",
        edgecolors="none",
        label="Condensate"
    )

    # moving average 曲线
    # ax.plot(x, y_ma, linewidth=2.5)
    ax.plot(x, y_ma, linewidth=4, color="#2B2B2B", zorder=3)  # charcoal

    ax.set_xlabel("Protein Expression Level", fontsize=20)
    ax.set_ylabel("Predicted Condensate Probability", fontsize=20)
    ax.set_title(f"{protein_name} Condensate Formation Across Expression Levels", fontsize=22)
    # ax.set_title("Condensate Formation Across Expression Levels", fontsize=22)

    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # set the size of x and y ticks
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # 图例（只显示两类点）
    ax.legend(
        frameon=False, 
        loc="lower right", 
        bbox_to_anchor=(1.0, 0.1), 
        fontsize=20, 
        markerscale=2.5,
        )

    legend = ax.get_legend()
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()

    # 保存
    save_path = data_root / "predicted_condensate_probability_ma.svg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # ====== 在同一张图上加标注，再另存一份 ======
    # 标注 max / min 点
    # ax.scatter([x[max_idx]], [y_max], s=90, marker="^", zorder=6, label="max", color="#D81B60")   # 酒红/洋红
    # ax.scatter([x[min_idx]], [y_min], s=90, marker="v", zorder=6, label="min after max", color="#1E88E5")   # 亮蓝

    ax.scatter([x[max_idx]], [y_max], s=200, marker="^", zorder=6,
            label="max", color="#B60000", edgecolors="white", linewidths=2.0)
    ax.scatter([x[min_idx]], [y_min], s=200, marker="v", zorder=6,
            label="min after max", color="#E4E100", edgecolors="white", linewidths=2.0)

    # 连接两点（可选，但直观看 gap 取值来源）
    ax.plot([x[max_idx], x[min_idx]], [y_max, y_min], linestyle="--", zorder=5)

    # 两个点的文字标注（带箭头）
    ax.annotate(
        f"max\n({y_max:.3f})",
        xy=(x[max_idx], y_max),
        xytext=(10, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="right",
        va="bottom",
        zorder=7,
        size=10,
    )
    ax.annotate(
        f"min after max\n({y_min:.3f})",
        xy=(x[min_idx], y_min),
        xytext=(10, -14),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="right",
        va="top",
        zorder=7,
    )

    # m_gap 数值标注（放在图左上角的文本框）
    ax.text(
        0.95, 0.95,
        f"Gap = {m_gap:.4f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.85, facecolor="white"),
        zorder=12,
        size=12,
    )

    ax.legend(loc="lower right", frameon=False, fontsize=12)

    save_path2 = data_root / "predicted_condensate_probability_ma_mgap.png"
    fig.savefig(save_path2, bbox_inches="tight")
    plt.close(fig)

    print(f"Max Gap: {m_gap:.4f}")
