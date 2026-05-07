import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# meta_data = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/random_KRDE/panel.csv")
# all_protein_path = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE/*'))

meta_data = pd.read_csv("cell_fm/tasks/cell_fm_cs/data/random_KRDE/panel_v2.csv")
all_protein_path = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE_v2/*'))
save_path = "cell_fm/tasks/cell_fm_cs/data/random_KRDE/panel_v2_selected.csv"

blocky_list = []
alternating_list = []
index_list = []

for protein_path in tqdm(all_protein_path):
    protein_path = Path(protein_path)

    idx = protein_path.name.split("_")[1]
    index_list.append(idx)

    blocky_path = protein_path / "blocky"
    alternating_path = protein_path / "alternating"

    blocky_df = pd.read_csv(blocky_path / "pred_vs_intensity.csv")
    alternating_df = pd.read_csv(alternating_path / "pred_vs_intensity.csv")

    # proportion of predicted_class == 1 (you can rename as "positive rate")
    blocky_count = (blocky_df["predicted_class"] == 1).sum() / len(blocky_df)
    alternating_count = (alternating_df["predicted_class"] == 1).sum() / len(alternating_df)

    blocky_list.append(blocky_count)
    alternating_list.append(alternating_count)

blocky_arr = np.asarray(blocky_list, dtype=float)
alternating_arr = np.asarray(alternating_list, dtype=float)
diff_arr = blocky_arr - alternating_arr
label_arr = diff_arr > 0

# keep top_k indexes
top_k = 300
top_k_indices = np.argsort(diff_arr)[::-1][:top_k]

# randomly choose 88 from top_k_indices and 12 negative samples.
NUM_POSITIVE = 88
NUM_NEGATIVE = 12
negative_indices = np.where(label_arr == False)[0]

rng = np.random.default_rng(0)
chosen_positive_indices = rng.choice(top_k_indices, size=NUM_POSITIVE, replace=False)
chosen_negative_indices = rng.choice(negative_indices, size=NUM_NEGATIVE, replace=False)
chosen_indices = np.concatenate([chosen_positive_indices, chosen_negative_indices])

blocky_arr = blocky_arr[chosen_indices]
alternating_arr = alternating_arr[chosen_indices]
diff_arr = diff_arr[chosen_indices]
index_list = [index_list[i] for i in chosen_indices]

# # keep top 100 proteins
# top_k = 1000
# top_k_indices = np.argsort(diff_arr)[::-1][:top_k]
# blocky_arr = blocky_arr[top_k_indices]
# alternating_arr = alternating_arr[top_k_indices]
# diff_arr = diff_arr[top_k_indices]
# index_list = [index_list[i] for i in top_k_indices]

# print("Top k indices:", index_list)

meta_data_chosen = meta_data[meta_data['index'].astype(str).isin(index_list)]
# remove 'index' column
meta_data_chosen = meta_data_chosen.reset_index(drop=True)
meta_data_chosen = meta_data_chosen.drop(columns=['index'])

# save meta_data_chosen to a new csv file
meta_data_chosen.to_csv(save_path, index=True)

print("median diff (blocky - alternating):", float(np.median(diff_arr)))
print("mean diff (blocky - alternating):", float(np.mean(diff_arr)))
print("num of positive diffs:", int(np.sum(diff_arr > 0)), "/", len(diff_arr))

# -------- Paired dot plot / slopegraph (B -> A) --------
# B = alternating, A = blocky
n = len(diff_arr)
rng = np.random.default_rng(0)
jitter = 0.06

x_B = 0 + rng.uniform(-jitter, jitter, size=n)  # Alternating (B)
x_A = 1 + rng.uniform(-jitter, jitter, size=n)  # Blocky (A)

fig, ax = plt.subplots(figsize=(5.0, 6.0))

# connecting lines (paired)
for i in range(n):
    ax.plot([x_B[i], x_A[i]], [alternating_arr[i], blocky_arr[i]],
            alpha=0.25, linewidth=0.8)

# points
ax.scatter(x_B, alternating_arr, alpha=0.75, s=24)
ax.scatter(x_A, blocky_arr, alpha=0.75, s=24)

# group medians (thick short horizontal bars)
med_B = float(np.median(alternating_arr))
med_A = float(np.median(blocky_arr))
ax.hlines(med_B, -0.12, 0.12, linewidth=2.2)
ax.hlines(med_A, 0.88, 1.12, linewidth=2.2)

# axis / labels
ax.set_xlim(-0.35, 1.35)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Alternating", "Blocky"])
ax.set_ylabel("Predicted Condensate Probability")
ax.set_title("Condensate formation comparison: Blocky vs Alternating")

# optional: emphasize the "no difference" line if your metric is centered, else skip
# ax.axhline(0, linewidth=1, alpha=0.4)

fig.tight_layout()
fig.savefig("paired_dot_plot_blocky_vs_alternating.png", dpi=300)
plt.close()