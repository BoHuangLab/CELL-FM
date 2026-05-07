import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# --- 1. 找到所有 pred_vs_intensity.csv ---
protein_name = "NUP98"

all_csv_paths = sorted(
    glob(f"/hpc/reference/opencell/condenseq/seq2img_single/{protein_name}/protein_*/pred_vs_intensity.csv")
)
save_dir = f"output/condenseq/seq2img_single/{protein_name}/plots_all"

tasks = []
for csv_path in all_csv_paths:
    folder_name = os.path.basename(os.path.dirname(csv_path))
    variant_name = folder_name.split("_", 1)[-1]
    tasks.append((variant_name, csv_path))
    print(variant_name)

# --- 2. 生成 show_names（与 tasks 一一对应）---
show_names = []
m_idx = 1
for variant_name, _ in tasks:
    if variant_name == protein_name:
        show_names.append(f"{protein_name}_WT")
    else:
        show_names.append(f"{protein_name}_M{m_idx}")
        m_idx += 1

print(show_names)

# 定义窗口大小
window_size = 512

def calculate_ma(y_series, x_series, window):
    y_ma = (
        pd.Series(y_series)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    mask = ~np.isnan(y_ma)
    return x_series[mask], y_ma[mask]

# ✅ 背景透明：figure/axes facecolor 设为 none
fig, ax = plt.subplots(figsize=(10, 6), facecolor="none")
ax.set_facecolor("none")

cmap = plt.get_cmap("tab10")
nonwt_color_idx = 0

for (variant_name, csv_path), show_label in zip(tasks, show_names):
    df = pd.read_csv(csv_path)
    x = df["protein_intensity_level"].to_numpy()
    y = df["predicted_class"].to_numpy()

    x_ma, y_ma = calculate_ma(y, x, window_size)

    # WT 黑色；Mi 用 tab10
    if variant_name == protein_name:
        color = "black"
    else:
        color = cmap(nonwt_color_idx % 10)
        nonwt_color_idx += 1

    ax.plot(x_ma, y_ma, color=color, linewidth=4, label=show_label)

ax.set_xlabel("Protein Expression Level", fontsize=20)
ax.set_ylabel("Predicted Condensate Probability", fontsize=20)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)

ax.set_title("Condensate Formation Across Expression Levels", fontsize=22)
ax.legend(fontsize=18)

# ✅ grid 如果你也希望“更干净”，可以关掉或调淡；透明背景下 grid 仍会显示
ax.grid(alpha=0.5)

fig.tight_layout()

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "pred_vs_intensity_ma_compare.svg")

# ✅ 保存透明背景
fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True)
plt.close(fig)

print(f"Saved plot to {save_path}")
