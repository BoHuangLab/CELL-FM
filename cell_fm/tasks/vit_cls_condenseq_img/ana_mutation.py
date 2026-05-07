from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt

def main() -> None:
    output_dir = Path("output/condenseq/analysis/mutation_impact")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = "/hpc/reference/opencell/condenseq/seq2img_exp_level_mutation/PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k/cellfm_test"

    pred_WT_results_files = sorted(glob(os.path.join(base, "WT", "*")))
    pred_D2E_results_files = sorted(glob(os.path.join(base, "D2E", "*")))
    pred_E2D_results_files = sorted(glob(os.path.join(base, "E2D", "*")))
    pred_DE2G_results_files = sorted(glob(os.path.join(base, "DE2G", "*")))
    pred_F2W_results_files = sorted(glob(os.path.join(base, "F2W", "*")))
    pred_K2R_results_files = sorted(glob(os.path.join(base, "K2R", "*")))
    pred_R2K_results_files = sorted(glob(os.path.join(base, "R2K", "*")))

    n_wt, n_d2e, n_e2d, n_de2g, n_f2w, n_k2r, n_r2k = map(len, (pred_WT_results_files, pred_D2E_results_files, pred_E2D_results_files, pred_DE2G_results_files, pred_F2W_results_files, pred_K2R_results_files, pred_R2K_results_files))
    if not (n_wt == n_d2e == n_e2d == n_de2g == n_f2w == n_k2r == n_r2k):
        raise ValueError(f"Folder counts mismatch: WT={n_wt}, D2E={n_d2e}, E2D={n_e2d}, DE2G={n_de2g}, F2W={n_f2w}, K2R={n_k2r}, R2K={n_r2k}")

    diff_D2E = []
    diff_E2D = []
    diff_DE2G = []
    diff_F2W = []
    diff_K2R = []
    diff_R2K = []

    for wt_dir, d2e_dir, e2d_dir, de2g_dir, f2w_dir, k2r_dir, r2k_dir in tqdm(
        zip(pred_WT_results_files, pred_D2E_results_files, pred_E2D_results_files, pred_DE2G_results_files, pred_F2W_results_files, pred_K2R_results_files, pred_R2K_results_files),
        total=n_wt,
        desc="Computing condensate differences"
    ):
        wt_csv = os.path.join(wt_dir, "pred_vs_intensity.csv")
        d2e_csv = os.path.join(d2e_dir, "pred_vs_intensity.csv")
        e2d_csv = os.path.join(e2d_dir, "pred_vs_intensity.csv")
        de2g_csv = os.path.join(de2g_dir, "pred_vs_intensity.csv")
        f2w_csv = os.path.join(f2w_dir, "pred_vs_intensity.csv")
        k2r_csv = os.path.join(k2r_dir, "pred_vs_intensity.csv")
        r2k_csv = os.path.join(r2k_dir, "pred_vs_intensity.csv")

        if not (os.path.exists(wt_csv) and os.path.exists(d2e_csv) and os.path.exists(e2d_csv) and os.path.exists(de2g_csv) and os.path.exists(f2w_csv) and os.path.exists(k2r_csv) and os.path.exists(r2k_csv)):
            continue

        pred_WT_results = pd.read_csv(wt_csv)
        pred_D2E_results = pd.read_csv(d2e_csv)
        pred_E2D_results = pd.read_csv(e2d_csv)
        pred_DE2G_results = pd.read_csv(de2g_csv)
        pred_F2W_results = pd.read_csv(f2w_csv)
        pred_K2R_results = pd.read_csv(k2r_csv)
        pred_R2K_results = pd.read_csv(r2k_csv)

        num_condensate_WT = (pred_WT_results["predicted_class"] == 1).sum()
        num_condensate_D2E = (pred_D2E_results["predicted_class"] == 1).sum()
        num_condensate_E2D = (pred_E2D_results["predicted_class"] == 1).sum()
        num_condensate_DE2G = (pred_DE2G_results["predicted_class"] == 1).sum()
        num_condensate_F2W = (pred_F2W_results["predicted_class"] == 1).sum()
        num_condensate_K2R = (pred_K2R_results["predicted_class"] == 1).sum()
        num_condensate_R2K = (pred_R2K_results["predicted_class"] == 1).sum()

        diff_D2E.append(num_condensate_D2E - num_condensate_WT)
        diff_E2D.append(num_condensate_E2D - num_condensate_WT)
        diff_DE2G.append(num_condensate_DE2G - num_condensate_WT)
        diff_F2W.append(num_condensate_F2W - num_condensate_WT)
        diff_K2R.append(num_condensate_K2R - num_condensate_WT)
        diff_R2K.append(num_condensate_R2K - num_condensate_WT)

    diff_D2E = np.array(diff_D2E, dtype=float)
    diff_E2D = np.array(diff_E2D, dtype=float)
    diff_DE2G = np.array(diff_DE2G, dtype=float)
    diff_F2W = np.array(diff_F2W, dtype=float)
    diff_K2R = np.array(diff_K2R, dtype=float)
    diff_R2K = np.array(diff_R2K, dtype=float)

    # ============= Plot =============
    fig, ax = plt.subplots(figsize=(6, 5))

    parts = ax.violinplot(
        [diff_D2E, diff_E2D, diff_DE2G, diff_F2W, diff_K2R, diff_R2K],
        showmeans=True, 
        showmedians=True, 
        showextrema=True, 
    )

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["D2E - WT", "E2D - WT", "DE2G - WT", "F2W - WT", "K2R - WT", "R2K - WT"], rotation=30)

    x1 = np.full_like(diff_D2E, 1, dtype=float)
    x2 = np.full_like(diff_E2D, 2, dtype=float)
    x3 = np.full_like(diff_DE2G, 3, dtype=float)
    x4 = np.full_like(diff_F2W, 4, dtype=float)
    x5 = np.full_like(diff_K2R, 5, dtype=float)
    x6 = np.full_like(diff_R2K, 6, dtype=float)

    jitter_scale = 0.06
    ax.scatter(x1 + (np.random.rand(len(x1)) - 0.5) * jitter_scale,
               diff_D2E,
               alpha=0.6,
               s=15)
    ax.scatter(x2 + (np.random.rand(len(x2)) - 0.5) * jitter_scale,
               diff_E2D,
               alpha=0.6,
               s=15)
    ax.scatter(x3 + (np.random.rand(len(x3)) - 0.5) * jitter_scale,
               diff_DE2G,
               alpha=0.6,
               s=15)
    ax.scatter(x4 + (np.random.rand(len(x4)) - 0.5) * jitter_scale,
               diff_F2W,
               alpha=0.6,
               s=15)
    ax.scatter(x5 + (np.random.rand(len(x5)) - 0.5) * jitter_scale,
               diff_K2R,
               alpha=0.6,
               s=15)
    ax.scatter(x6 + (np.random.rand(len(x6)) - 0.5) * jitter_scale,
               diff_R2K,
               alpha=0.6,
               s=15)

    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_ylabel("Difference in condensate count (mutation - WT)")
    ax.set_title("Condensate count differences per field of view")

    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    plt.savefig(output_dir / "condensate_count_diff.png", dpi=300)

if __name__ == "__main__":
    main()