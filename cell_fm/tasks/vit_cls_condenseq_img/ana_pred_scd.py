import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy.stats import pearsonr
from tqdm import tqdm

# --- 1. find all pred_vs_intensity.csv ---
all_csv_paths = sorted(glob("/hpc/reference/opencell/condenseq/seq2img_single/DDX4_rand_mt/*/pred_vs_intensity.csv"))
meta_data = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/DDX4/scd_panel.csv")

all_scd = []
all_counts = []

for csv_path in tqdm(all_csv_paths):
    mutant_id = csv_path.split("/")[-2].split("_", 2)[-1]  
    scd = meta_data[meta_data["mutant_id"] == mutant_id]["scd"].values[0]
    all_scd.append(scd)

    df = pd.read_csv(csv_path)
    counts = (df['predicted_class'] == 1).sum() / len(df)
    all_counts.append(counts)

all_scd = np.array(all_scd)
all_counts = np.array(all_counts)

# --- 2. Linear regression ---
slope, intercept = np.polyfit(all_scd, all_counts, 1)
pred_line = slope * all_scd + intercept

# Pearson correlation coefficient
r, _ = pearsonr(all_scd, all_counts)

# --- 3. Beautified plot ---
plt.figure(figsize=(9, 7))

# scatter points
plt.scatter(all_scd, all_counts, 
            s=80, 
            alpha=0.75, 
            edgecolor="black", 
            linewidth=0.8,
            color="#1f77b4")  # matplotlib default blue (seaborn-like)

# regression line
x_sorted = np.linspace(min(all_scd), max(all_scd), 200)
plt.plot(x_sorted, slope * x_sorted + intercept, 
         color="red", linewidth=2.0, label=f"Fit: y = {slope:.2f}x + {intercept:.1f} (Pearson r={r:.2f})")

# axis labels
plt.xlabel("SCD (Sequence Charge Decoration)", fontsize=15)
plt.ylabel("Predicted Condensate Count", fontsize=15)
plt.title("DDX4 Mutants: SCD vs Condensate Predictions", fontsize=18, pad=12)

# zero reference line (optional)
plt.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

# grid
plt.grid(True, alpha=0.25)

# tick size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig("./scd_vs_counts_ddx4.png", dpi=300)
plt.close()
