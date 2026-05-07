import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# --- 1. 找到所有 pred_vs_intensity.csv ---
# protein_name = "E-K"
# protein_name = "E-K-5"
# protein_name = "DDX4"
# protein_name = "LAF-1"
# protein_name = "LAF-1_2"
# protein_name = "hnRNPA1"
# protein_name = "NUP98"
protein_name = "ELP"

all_csv_paths = sorted(glob(f"/hpc/reference/opencell/condenseq/seq2img_single/{protein_name}/protein_*/pred_vs_intensity.csv"))
save_dir = f"output/condenseq/seq2img_single/{protein_name}/plots_all"

tasks = []

for csv_path in all_csv_paths:
    folder_name = os.path.basename(os.path.dirname(csv_path))
    variant_name = folder_name.split("_", 1)[-1]

    if "LAF-1_2" in variant_name:
        variant_name = variant_name.replace("LAF-1_2", "LAF-1")

    if "WT" in variant_name:
        pass  # skip WT
    else:
        tasks.append((variant_name, csv_path))

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

# plt.figure(figsize=(8, 5), facecolor="none")
# plt.figure(figsize=(10, 5), facecolor="none")
plt.figure(figsize=(10, 10), facecolor="none")
# cmap = plt.get_cmap("tab10")
cmap = plt.get_cmap("tab20")
ax = plt.gca()
ax.set_facecolor("none")

for idx, (variant_name, csv_path) in enumerate(tasks):
    df = pd.read_csv(csv_path)
    x = df["protein_intensity_level"].to_numpy()
    y = df["predicted_class"].to_numpy()

    x_ma, y_ma = calculate_ma(y, x, window_size)

    color = cmap(idx)
    label = variant_name

    ax.plot(x_ma, y_ma, color=color, linewidth=4, label=label)

ax.set_xlabel("Protein Expression Level", fontsize=18)
ax.set_ylabel("Predicted Condensate Probability", fontsize=18)

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

plt.title("Condensate Formation Across Expression Levels", fontsize=20)
# plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0, 0.99))
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1, 0.06))
plt.grid(alpha=0.5)
plt.tight_layout()

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"pred_vs_intensity_ma_compare.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True)
plt.close()

print(f"Saved plot to {save_path}")