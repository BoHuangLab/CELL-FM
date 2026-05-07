import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# all_protein_path = sorted(glob("/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_selected_test_data/*"))
all_protein_path = sorted(glob("/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_selected_test_v6_data/*"))

diff_blocky_list = []
diff_alternating_list = []

for protein_path in tqdm(all_protein_path):
    protein_path = Path(protein_path)

    wt_path = protein_path / "WT"
    blocky_path = protein_path / "blocky"
    alternating_path = protein_path / "alternating"

    wt_df = pd.read_csv(wt_path / "pred_vs_intensity.csv")
    blocky_df = pd.read_csv(blocky_path / "pred_vs_intensity.csv")
    alternating_df = pd.read_csv(alternating_path / "pred_vs_intensity.csv")

    wt_count = (wt_df['predicted_class'] == 1).sum() / len(wt_df)
    blocky_count = (blocky_df['predicted_class'] == 1).sum() / len(blocky_df)
    alternating_count = (alternating_df['predicted_class'] == 1).sum() / len(alternating_df)

    diff_blocky = blocky_count - wt_count
    diff_alternating = alternating_count - wt_count

    diff_blocky_list.append(diff_blocky)
    diff_alternating_list.append(diff_alternating)

print("median blocky:", np.median(diff_blocky_list), "median alternating:", np.median(diff_alternating_list))
print("mean blocky:", np.mean(diff_blocky_list), "mean alternating:", np.mean(diff_alternating_list))

# -----------------------------
# Plot violin + jitter points
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")

data_raw = [
    np.array(diff_blocky_list, dtype=float),
    np.array(diff_alternating_list, dtype=float),
]
labels = ["blocky", "alternating"]
pos = np.arange(len(labels))

# colors
colors = ["#bebebe", "#bebebe"]

fig, ax = plt.subplots(figsize=(6, 5))

# 1) Violin
parts = ax.violinplot(
    data_raw, positions=pos, widths=0.7,
    showmeans=False, showextrema=False, showmedians=False
)
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(colors[i])
    body.set_edgecolor("black")
    body.set_alpha(1.0)
    body.set_linewidth(2.0)

# 2) Scatter (original data + KDE controlled horizontal width)
rng = np.random.default_rng(0)
max_points   = 2000
width_scale  = 0.35
point_size   = 15
point_alpha  = 0.30
point_color  = "black"

for i, vals in enumerate(data_raw):
    y = vals.copy()
    n = len(y)
    if n == 0:
        continue

    # Limit max points
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        y = y[idx]

    y_std = float(np.std(y))
    if y_std <= 1e-12:
        # All points are almost the same, give a fixed width
        widths = np.full_like(y, fill_value=0.06, dtype=float)
    else:
        y_lo, y_hi = np.percentile(y, [1, 99])
        if y_hi <= y_lo:
            y_lo, y_hi = float(np.min(y)), float(np.max(y))

        if y_hi == y_lo:
            widths = np.full_like(y, fill_value=0.06, dtype=float)
        else:
            # KDE in original space
            y_grid = np.linspace(y_lo, y_hi, 256)
            kde = gaussian_kde(vals)  # Use original full vals to estimate density

            dens = kde(y_grid)
            dens = np.maximum(dens, 1e-12)
            dens = dens / dens.max()
            dens = dens * width_scale

            idx_near = np.abs(y[:, None] - y_grid[None, :]).argmin(axis=1)
            widths = dens[idx_near]

    x = pos[i] + rng.uniform(-1.0, 1.0, size=len(y)) * widths
    ax.scatter(x, y, s=point_size, alpha=point_alpha,
               color=point_color, edgecolors="none", zorder=3)

# 3) Statistical lines (original space) + median numbers (white text)
for i, vals in enumerate(data_raw):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        continue

    q1, q3 = np.percentile(vals, [25, 75])
    med = np.median(vals)
    mean = np.mean(vals)
    vmin, vmax = np.min(vals), np.max(vals)

    # whiskers + IQR
    ax.vlines(pos[i], vmin, vmax, color="black", lw=2.0, alpha=0.75, zorder=2)
    ax.vlines(pos[i], q1, q3, color="black", lw=12.0, alpha=0.75, zorder=2)

    # median line
    ax.hlines(med, pos[i]-0.18, pos[i]+0.18, color="black", lw=2.0, zorder=4)

    # Median numbers (white text)
    ax.text(
        pos[i], med,
        f"{med:.4f}",
        color="white",
        ha="center", va="center",
        fontsize=13, fontweight="bold",
        zorder=5
    )

# 4) Axes and labels
ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # WT reference line
ax.set_xticks(pos)
ax.set_xticklabels(labels)
ax.set_xlabel("mutation type")
ax.set_ylabel("Mutant fraction - WT fraction")
ax.set_title("Difference of mutant fraction vs WT")
sns.despine()

fig.tight_layout()
# fig.savefig(
#     "output/condenseq/analysis/charged_domain/charged_domain_mutation_selected_test_data_diff_violin_plot.png",
#     dpi=300, 
# )
fig.savefig(
    "output/condenseq/analysis/charged_domain/charged_domain_mutation_selected_test_v6_data_diff_violin_plot.png",
    dpi=300, 
)
plt.close(fig)