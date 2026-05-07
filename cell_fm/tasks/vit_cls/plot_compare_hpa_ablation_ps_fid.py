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

    # Load FID results
    ps2 = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k_results/fids.npy')
    ps4 = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_ps4_results/fids.npy')
    ps8 = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_ps8_results/fids.npy')

    # Violin plot (log10 space)
    c1 = "#b0b0b0"
    c2 = "#b0b0b0"
    c3 = "#b0b0b0"

    def clean(x):
        return np.asarray(x)[np.isfinite(x)]

    ps2_c = clean(ps2)
    ps4_c   = clean(ps4)
    ps8_c   = clean(ps8)

    # Log-transform the data for plotting/statistics
    eps = 1e-12  # Prevent log(0)
    ps2_log = np.log10(ps2_c + eps)
    ps4_log   = np.log10(ps4_c  + eps)
    ps8_log   = np.log10(ps8_c  + eps)

    data_log = [ps2_log, ps4_log, ps8_log]
    data_raw = [ps2_c,   ps4_c,   ps8_c]   # Original-space data for displaying raw median values
    labels   = ["2", "4", "8"]
    colors   = [c1, c2, c3]
    pos      = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Violin (log data)
    parts = ax.violinplot(
        data_log, positions=pos, widths=0.7,
        showmeans=False, showextrema=False, showmedians=False
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(1.0)
        body.set_linewidth(2.0)

    # 2) Scatter plot (log data + KDE)
    rng = np.random.default_rng(0)
    max_points   = 2000
    width_scale  = 0.35
    point_size   = 15
    point_alpha  = 0.20
    point_color  = "black"

    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)

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
            kde = gaussian_kde(vals_log)  # 在 log 空间估计密度

            dens = kde(y_grid)
            dens = np.maximum(dens, 1e-12)
            dens = dens / dens.max()
            dens = dens * width_scale

            idx_near = np.abs(y[:, None] - y_grid[None, :]).argmin(axis=1)
            widths   = dens[idx_near]

        x = pos[i] + rng.uniform(-1.0, 1.0, size=len(y)) * widths
        ax.scatter(x, y, s=point_size, alpha=point_alpha,
                   color=point_color, edgecolors="none", zorder=3)

    # 3) Statistical lines (log data) + Display raw median values on the median line (white text)
    for i, vals_log in enumerate(data_log):
        q1, q3 = np.percentile(vals_log, [25, 75])
        med_log = np.median(vals_log)               # Median in log space (for positioning)
        med_raw = np.median(data_raw[i])            # Median in original space (for displaying value)
        vmin, vmax = np.min(vals_log), np.max(vals_log)

        # whiskers + IQR
        ax.vlines(pos[i], vmin, vmax, color="black", lw=2.0, alpha=0.75, zorder=2)
        ax.vlines(pos[i], q1, q3, color="black", lw=12.0, alpha=0.75, zorder=2)

        # median line
        ax.hlines(med_log, pos[i]-0.18, pos[i]+0.18, color="black", lw=2.0, zorder=4)

        # Display the raw median value on the median line (white text)
        ax.text(
            pos[i], med_log,
            f"{med_raw:.3f}",
            color="white",
            ha="center", va="center",
            fontsize=25, fontweight="bold",
            zorder=5
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Image generator patch size', fontsize=30)
    ax.set_ylabel(r'$\log_{10}(\mathrm{msFID\ value})$', fontsize=30)
    ax.set_title("msFID Distribution", fontsize=35)

    # Do not draw legend — median values are already displayed on the lines

    plt.tight_layout()
    plt.savefig("output/hpa/embed/ablation_ps_fid_violin.svg", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print("saved: output/hpa/embed/ablation_ps_fid_violin.svg")

if __name__ == "__main__":
    main()

