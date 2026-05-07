import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

# -----------------------------
# Data loading (your logic)
# -----------------------------
# all_protein_path = sorted(glob(
#     "/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE_panel_v2_selected/*"
# ))
all_protein_path = sorted(glob(
    "/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE_panel_v2_selected_4096_512/*"
))

blocky_list = []
alternating_list = []
index_list = []

for protein_path in tqdm(all_protein_path):
    protein_path = Path(protein_path)

    idx = protein_path.name.split("_")[1]
    index_list.append(idx)

    blocky_df = pd.read_csv(protein_path / "blocky" / "pred_vs_intensity.csv")
    alternating_df = pd.read_csv(protein_path / "alternating" / "pred_vs_intensity.csv")

    blocky_rate = (blocky_df["predicted_class"] == 1).sum() / len(blocky_df)
    alternating_rate = (alternating_df["predicted_class"] == 1).sum() / len(alternating_df)

    blocky_list.append(blocky_rate)
    alternating_list.append(alternating_rate)

A = np.asarray(blocky_list, dtype=float)        # Blocky
B = np.asarray(alternating_list, dtype=float)   # Alternating

print("Median predicted condensate probability (blocky):", float(np.median(A)))
print("Median predicted condensate probability (alternating):", float(np.median(B)))

diff = A - B
n = diff.size

print("median diff (blocky - alternating):", float(np.median(diff)))
print("mean diff (blocky - alternating):", float(np.mean(diff)))
print("num of positive diffs:", int(np.sum(diff > 0)), "/", n)

def paired_dotplot_publish_ready_svg(
    B, A,
    outpath="paired_dot_plot_blocky_vs_alternating.svg",
    figsize=(4.2, 5.0),
    seed=0,
    jitter=0.055,
    y_as_percent=True,     # set False if you want 0..1 labels
):
    B = np.asarray(B, float)
    A = np.asarray(A, float)
    if B.shape != A.shape:
        raise ValueError("A and B must have the same shape.")

    diff = A - B
    n = diff.size

    rng = np.random.default_rng(seed)
    xB = 0 + rng.uniform(-jitter, jitter, size=n)
    xA = 1 + rng.uniform(-jitter, jitter, size=n)

    up = diff > 0

    # with mpl.rc_context(PUBLISH_STYLE):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Grid: subtle, journal-friendly
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.22)
    ax.xaxis.grid(False)

    # Paired lines (very light)
    for i in range(n):
        ax.plot([xB[i], xA[i]], [B[i], A[i]],
                linewidth=1.0, alpha=0.2,
                color=("0.25" if up[i] else "0.60"),
                zorder=1)

    # Points (clean, slightly larger, with white stroke)
    ax.scatter(xB, B, s=26, alpha=0.92, color="0.15",
                edgecolors="white", linewidths=0.35, zorder=3)
    ax.scatter(xA, A, s=26, alpha=0.92, color="0.15",
                edgecolors="white", linewidths=0.35, zorder=3)

    # Medians
    medB = float(np.median(B))
    medA = float(np.median(A))
    ax.hlines(medB, -0.14, 0.14, linewidth=2.5, color="0.10", zorder=4)
    ax.hlines(medA,  0.86, 1.14, linewidth=2.5, color="0.10", zorder=4)

    # # Median labels (offset so they don't collide)
    # ax.annotate(f"{medB:.2f}", (0, medB), xytext=(6, 0), textcoords="offset points",
    #             va="center", ha="left", fontsize=12, color="0.25")
    # ax.annotate(f"{medA:.2f}", (1, medA), xytext=(6, 0), textcoords="offset points",
    #             va="center", ha="left", fontsize=12, color="0.25")

    # Median labels (offset so they don't collide)
    ax.annotate(f"{medB:.2f}", (0, medB),
                xytext=(-22, 0), textcoords="offset points",  # 往左挪
                va="center", ha="right", fontsize=14, color="0.25")

    ax.annotate(f"{medA:.2f}", (1, medA),
                xytext=(22, 0), textcoords="offset points",   # 往右挪
                va="center", ha="left", fontsize=14, color="0.25")

    # Axes
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Alternating", "Blocky"])
    ax.tick_params(axis="x", length=0, labelsize=14)

    ax.set_ylabel("Predicted condensate fraction" + (" (%)" if y_as_percent else ""), fontsize=16)
    ax.tick_params(axis="y", length=0, labelsize=14)

    if y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Y limits with gentle padding (assumes probabilities in [0,1])
    y_all = np.concatenate([A, B])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    span = max(1e-9, y_max - y_min)
    pad = 0.06 * span
    lo = max(0.0, y_min - pad)
    hi = min(1.0, y_max + pad)
    if (hi - lo) < 0.02:
        mid = 0.5 * (hi + lo)
        lo = max(0.0, mid - 0.01)
        hi = min(1.0, mid + 0.01)
    ax.set_ylim(lo, hi)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # # Stats box
    # stat_text = (
    #     f"n = {n}\n"
    #     f"median(A−B) = {np.median(diff):.3f}\n"
    #     f"mean(A−B) = {np.mean(diff):.3f}\n"
    #     f"A > B = {np.sum(diff > 0)}/{n}"
    # )
    # ax.text(0.02, 0.98, stat_text,
    #         transform=ax.transAxes,
    #         va="top", ha="left",
    #         fontsize=9,
    #         bbox=dict(boxstyle="round,pad=0.28",
    #                   facecolor="white", edgecolor="0.85", alpha=0.95),
    #         zorder=10)

    ax.set_title("Blocky vs Alternating patterning", pad=0.0, fontsize=16)

    # Save as SVG (vector)
    fig.savefig(outpath, format="svg", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

paired_dotplot_publish_ready_svg(B, A, "paired_dot_plot_blocky_vs_alternating.svg")
