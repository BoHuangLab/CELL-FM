# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import gaussian_kde

def stats(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def main() -> None:

    # Load KID results
    PUPS = np.load('output/hpa/embed/PUPS_results/kids.npy')
    CD1  = np.load('output/hpa/embed/CELL-Diff_results/kids.npy')
    CD2  = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k_results/kids.npy')

    # only keep finite and positive KID values
    PUPS_pos = PUPS[np.isfinite(PUPS) & (PUPS > 0)]
    CD1_pos  = CD1 [np.isfinite(CD1 ) & (CD1  > 0)]
    CD2_pos  = CD2 [np.isfinite(CD2 ) & (CD2  > 0)]

    def describe(x):
        return {
            "n": len(x),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    # Statistics in the original scale, used for legend text
    d1, d2, d3 = describe(PUPS_pos), describe(CD1_pos), describe(CD2_pos)

    # --------- CHANGED: Log-transform with log10 before plotting histogram ----------
    eps = 0.0  # Since we've filtered out <=0 values, no need to add eps
    PUPS_log = np.log10(PUPS_pos + eps)
    CD1_log  = np.log10(CD1_pos  + eps)
    CD2_log  = np.log10(CD2_pos  + eps)

    # Create linearly spaced bins in log space
    xmin_log = min(PUPS_log.min(), CD1_log.min(), CD2_log.min())
    xmax_log = max(PUPS_log.max(), CD1_log.max(), CD2_log.max())
    bins = np.linspace(xmin_log, xmax_log, 80)

    fig, ax = plt.subplots(figsize=(14, 7))

    c1 = "#0072B2"  # blue
    c2 = "#E69F00"  # orange
    c3 = "#009E73"  # green

    # Plot the data in log space
    h1 = ax.hist(PUPS_log, bins=bins, alpha=0.45, density=False,
                label=f"PUPS (Median={d1['median']:.3f})", color=c1)
    h2 = ax.hist(CD1_log,  bins=bins, alpha=0.45, density=False,
                label=f"CELL-Diff (Median={d2['median']:.3f})", color=c2)
    h3 = ax.hist(CD2_log,  bins=bins, alpha=0.45, density=False,
                label=f"CELL-FM (Median={d3['median']:.3f})", color=c3)

    cline1 = cline2 = cline3 = "#000000"

    # ax.set_xscale("log")   # <- Remove

    # Grid can be directly in linear coordinates
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.5)

    # Set axis labels and title
    ax.set_xlabel(r'$\log_{10}(\mathrm{msKID\ value})$', fontsize=30)  # CHANGED: Log-transformed axis label
    ax.set_ylabel("Number of proteins", fontsize=30)
    ax.set_title("msKID Distribution", fontsize=35)

    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)

    # Median/IQR: Map the original statistics to log space for plotting
    ax.axvline(np.log10(d1["median"]), color=cline1, linewidth=1.5, linestyle="--")
    ax.axvline(np.log10(d2["median"]), color=cline2, linewidth=1.5, linestyle="--")
    ax.axvline(np.log10(d3["median"]), color=cline3, linewidth=1.5, linestyle="--")

    ax.axvspan(np.log10(d1["p25"]), np.log10(d1["p75"]), facecolor=cline1, alpha=0.12, edgecolor="none")
    ax.axvspan(np.log10(d2["p25"]), np.log10(d2["p75"]), facecolor=cline2, alpha=0.12, edgecolor="none")
    ax.axvspan(np.log10(d3["p25"]), np.log10(d3["p75"]), facecolor=cline3, alpha=0.12, edgecolor="none")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch    

    # First, construct proxy handles for the style legend
    median_proxy = Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="Median")
    iqr_proxy    = Patch(facecolor="black", alpha=0.12, label="IQR (p25-p75)")

    # Get the existing handles and labels from the histogram (from the ax.hist calls above)
    handles_hist, labels_hist = ax.get_legend_handles_labels()

    # Combine: datasets + style legend
    handles_all = handles_hist + [median_proxy, iqr_proxy]
    labels_all  = labels_hist  + ["Median", "IQR (p25-p75)"]

    # Only call legend once
    ax.legend(handles_all, labels_all,
            loc="upper left", frameon=False, fontsize=25,
            handlelength=2.5, handleheight=1.5, ncol=1)

    plt.tight_layout()
    plt.savefig("output/hpa/embed/kid_histogram.svg", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print("saved: output/hpa/embed/kid_histogram.svg")

    ########################################################################################################

    # Violin plot (log-transformed)
    # c1 = "#8cbfdc"; c2 = "#f3d38c"; c3 = "#8cd3bf"
    c1 = "#b0b0b0"; c2 = "#b0b0b0"; c3 = "#b0b0b0"

    def clean(x):
        return np.asarray(x)[np.isfinite(x)]

    PUPS_c = clean(PUPS)
    CD1_c  = clean(CD1)
    CD2_c  = clean(CD2)

    # ---------- CHANGED: Log-transform the data (KDE is also estimated in log space) ----------

    eps = 1e-12  # Prevent log(0)
    PUPS_log = np.log10(PUPS_c + eps)
    CD1_log  = np.log10(CD1_c  + eps)
    CD2_log  = np.log10(CD2_c  + eps)

    data_log = [PUPS_log, CD1_log, CD2_log]     # Log-transformed data for plotting/statistics
    labels   = ["PUPS", "CELL-Diff", "CELL-FM"]
    colors   = [c1, c2, c3]
    pos      = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)

    # 1) Violin (in log space)
    parts = ax.violinplot(
        data_log, positions=pos, widths=0.7,
        showmeans=False, showextrema=False, showmedians=False
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(1.0)
        body.set_linewidth(2.0)

    # 2) Scatter plot (also using log-transformed data; KDE is also in log space)
    rng = np.random.default_rng(0)
    max_points   = 2000
    width_scale  = 0.35
    point_size   = 15
    point_alpha  = 0.20
    point_color  = "black"

    for i, vals_log in enumerate(data_log):
        y = vals_log
        n = len(y)
        if n > max_points:
            idx = rng.choice(n, max_points, replace=False)
            y = y[idx]

        y_std = float(np.std(y))
        if y_std <= 1e-12:
            widths = np.full_like(y, fill_value=0.06, dtype=float)
        else:
            y_lo, y_hi = np.percentile(y, [1, 99])
            if y_hi <= y_lo:
                y_lo, y_hi = np.min(y), np.max(y)
            y_grid = np.linspace(y_lo, y_hi, 256)
            kde = gaussian_kde(vals_log)     # CHANGED: Use log-transformed data for KDE

            dens = kde(y_grid)
            dens = np.maximum(dens, 1e-12)
            dens = dens / dens.max()
            dens = dens * width_scale

            idx_near = np.abs(y[:, None] - y_grid[None, :]).argmin(axis=1)
            widths   = dens[idx_near]

        x = pos[i] + rng.uniform(-1.0, 1.0, size=len(y)) * widths
        ax.scatter(x, y, s=point_size, alpha=point_alpha,
                color=point_color, edgecolors="none", zorder=3)

    # 3) Statistical lines (computed and displayed in log space)
    for i, vals_log in enumerate(data_log):
        q1, q3 = np.percentile(vals_log, [25, 75])
        med    = np.median(vals_log)
        vmin, vmax = np.min(vals_log), np.max(vals_log)
        ax.vlines(pos[i], vmin, vmax, color="black", lw=2.0, alpha=0.75, zorder=2)
        ax.vlines(pos[i], q1, q3, color="black", lw=12.0, alpha=0.75, zorder=2)
        ax.hlines(med, pos[i]-0.18, pos[i]+0.18, color="black", lw=2.0, zorder=4)

        # Display the original-space value on the median line (in white)
        ax.text(
            pos[i], med,
            f"{10**med:.3f}",
            color="white",
            ha="center", va="center",
            fontsize=24, fontweight="bold",
            zorder=5
        )
    ax.tick_params(axis="y", labelsize=30)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'$\log_{10}(\mathrm{msKID\ value})$', fontsize=30)                    # CHANGED
    ax.set_title("msKID Distribution", fontsize=35)
    # ax.set_yscale("log")                               # CHANGED: Do not set log scale after plotting

    # Legend (still using original-scale medians; for geometric median, use 10**np.median(data_log[i]))
    from matplotlib.patches import Patch
    dataset_handles = [
        Patch(facecolor=colors[0], edgecolor="black", alpha=1.0,
            label=f"PUPS (Median={np.median(PUPS_c):.3f})"),
        Patch(facecolor=colors[1], edgecolor="black", alpha=1.0,
            label=f"CELL-Diff (Median={np.median(CD1_c):.3f})"),
        Patch(facecolor=colors[2], edgecolor="black", alpha=1.0,
            label=f"CELL-FM (Median={np.median(CD2_c):.3f})"),
    ]
    # ax.legend(handles=dataset_handles, loc="upper right", frameon=False, fontsize=12, handlelength=2.5, handleheight=1.5)

    plt.tight_layout()
    plt.savefig("output/hpa/embed/kid_violin.svg", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print("saved: output/hpa/embed/kid_violin.svg")


if __name__ == "__main__":
    main()