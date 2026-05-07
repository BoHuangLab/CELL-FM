import pandas as pd
from cell_fm.tasks.cell_fm_cs.generate_mutation_scd import compute_scd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm

meta_data = pd.read_csv('/hpc/reference/opencell/condenseq/cellfm_test_meta_data_large_pool_updated.csv')

scds = []
counts = []

for idx, row in tqdm(meta_data.iterrows(), total=len(meta_data)):
    protein_seq = row['protein_seq']
    scd = compute_scd(protein_seq)
    scds.append(scd)

    index = row['index']

    path = glob(f'/hpc/reference/opencell/condenseq/vit_cls_img/eval_with_exp_level/cellfm_test/*_{index}/pred_vs_intensity.csv')
    assert len(path) == 1, f"Cannot find unique path for index {index}"
    path = path[0]

    count_df = pd.read_csv(path)

    count = (count_df['predicted_class'] == 1).sum() / len(count_df)
    counts.append(count)

all_scd = np.array(scds)
all_counts = np.array(counts)

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
plt.title("Test Data: SCD vs Condensate Predictions", fontsize=18, pad=12)

# zero reference line (optional)
plt.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

# grid
plt.grid(True, alpha=0.25)

# tick size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig("output/condenseq/analysis/charged_domain/scd_vs_counts_raw_test_data.png", dpi=300)
plt.close()