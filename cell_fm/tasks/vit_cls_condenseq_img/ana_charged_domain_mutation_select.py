import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

H_WEIGHT = 0.0  # H 是否计入正电：0=不算，1=算（也可 0.1/0.5 等）

def compute_metrics(seq: str):
    seq = str(seq).strip()
    L = len(seq)
    if L == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan  # 多返回一个 aromatic_count

    nK = seq.count("K")
    nR = seq.count("R")
    nH = seq.count("H")
    nD = seq.count("D")
    nE = seq.count("E")

    # Fraction charged residues (FRC): (K+R+D+E)/L
    frc = (nK + nR + nD + nE) / L

    # Net charge per residue (NCPR): (pos - neg)/L
    pos = nK + nR + H_WEIGHT * nH
    neg = nD + nE
    ncpr = (pos - neg) / L

    # Aromatic
    aromatic_count = seq.count("F") + seq.count("W") + seq.count("Y")
    aromatic_frac = aromatic_count / L

    return L, frc, ncpr, aromatic_frac, aromatic_count

meta_data = pd.read_csv("/hpc/reference/opencell/condenseq/cellfm_test_meta_data_large_pool_updated.csv")
all_protein_path = sorted(glob("/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_all_test_data/*"))

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

# keep top 100 proteins
top_k = 100
top_k_indices = np.argsort(diff_arr)[::-1][:top_k]
blocky_arr = blocky_arr[top_k_indices]
alternating_arr = alternating_arr[top_k_indices]
diff_arr = diff_arr[top_k_indices]
index_list = [index_list[i] for i in top_k_indices]

top_k_meta_data = meta_data[meta_data["index"].isin(index_list)]

# 统一 index 的类型，避免 isin 匹配失败
meta_data["index"] = meta_data["index"].astype(str)
index_list = [str(x) for x in index_list]

top_k_meta_data = meta_data.loc[meta_data["index"].isin(index_list)].copy()

# 如果你想立刻确认是否为空：
print("top_k_meta_data rows:", len(top_k_meta_data))

cols = ["Length", "FRC", "NCPR", "AromaticFrac", "AromaticCount"]

# 用 expand 的方式稳定生成 5 列（不依赖 tolist() 的形状）
metrics_df = top_k_meta_data["protein_seq"].apply(
    lambda s: pd.Series(compute_metrics(s), index=cols)
)

top_k_meta_data[cols] = metrics_df
top_k_meta_data["AbsNCPR"] = top_k_meta_data["NCPR"].abs()

print("\nSummary:")
print(top_k_meta_data[["Length", "FRC", "NCPR", "AbsNCPR", "AromaticCount", "AromaticFrac"]].describe())

print("\nExtremes:")
print("Min FRC:", float(top_k_meta_data["FRC"].min()))
print("Max |NCPR|:", float(top_k_meta_data["AbsNCPR"].max()))
print("Max AromaticCount:", int(top_k_meta_data["AromaticCount"].max()))
print("Max AromaticFrac:", float(top_k_meta_data["AromaticFrac"].max()))

print("Top k indices:", index_list)

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
ax.set_xticklabels(["Alternating (B)", "Blocky (A)"])
ax.set_ylabel("Proportion predicted_class==1")
ax.set_title(
    f"Paired dot plot (n={n})\n"
    f"median(A-B)={np.median(diff_arr):.3f},  "
    f"mean(A-B)={np.mean(diff_arr):.3f},  "
    f"#(A>B)={np.sum(diff_arr>0)}/{n}"
)

# optional: emphasize the "no difference" line if your metric is centered, else skip
# ax.axhline(0, linewidth=1, alpha=0.4)

fig.tight_layout()
fig.savefig("paired_dot_plot_blocky_vs_alternating.png", dpi=300)
plt.close()
# plt.show()
