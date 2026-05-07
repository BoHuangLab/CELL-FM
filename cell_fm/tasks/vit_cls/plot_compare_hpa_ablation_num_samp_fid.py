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

    # -------------------------
    # Load FID results
    # -------------------------
    R0d5k = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_0d5k_results/fids.npy')
    R1k   = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_1k_results/fids.npy')
    R3k   = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_3k_results/fids.npy')
    R6k   = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_6k_results/fids.npy')
    Rall  = np.load('output/hpa/embed/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k_results/fids.npy')

    medians = np.array([np.median(R0d5k), np.median(R1k),
                        np.median(R3k), np.median(R6k),
                        np.median(Rall)], dtype=float)
    num_samples = np.array([500, 1000, 3000, 6000, 11694], dtype=float)

    order = np.argsort(num_samples)
    x = num_samples[order]
    y = medians[order]

    def thousands(x, pos):
        # 500 -> "500", 1000 -> "1,000"
        return f"{int(x):,}" if x >= 1000 else f"{int(x)}"

    # -------------------------
    # Plot
    # -------------------------

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)  # good 1-col paper size

    ax.plot(
        x, y,
        marker="o",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.8,
        markersize=18,
        linewidth=4.0,
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels([thousands(v, None) for v in x], rotation=45, fontsize=30)

    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)

    # Labels / title
    ax.set_xlabel("Number of samples", fontsize=30)
    ax.set_ylabel("Median msFID", fontsize=30)
    ax.set_title("Median msFID vs. Number of Samples", pad=0, fontsize=35)

    # Nice y-limits with margin (instead of hard-coding)
    ymin, ymax = np.min(y), np.max(y)
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + 1.8 * pad)

    # Subtle annotation above points
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(20, 10),
            ha="center",
            va="bottom",
            fontsize=25,
            color="0.25",
            zorder=4
        )

    # Lighten remaining spines
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Grid behind data
    ax.set_axisbelow(True)

    # -------------------------
    # Save (publish-ready)
    # -------------------------
    out_base = "output/hpa/embed/median_fid_vs_samples_publish_ready"
    fig.savefig(out_base + ".svg", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"saved: {out_base}.svg")

    # Violin plot (log10 space)
    c1 = "#b0b0b0"
    c2 = "#b0b0b0"
    c3 = "#b0b0b0"
    c4 = "#b0b0b0"
    c5 = "#b0b0b0"

    def clean(x):
        return np.asarray(x)[np.isfinite(x)]

    R0d5k_c = clean(R0d5k)
    R1k_c   = clean(R1k)
    R3k_c   = clean(R3k)
    R6k_c   = clean(R6k)
    Rall_c  = clean(Rall)

    # log10 变换（用于绘图/统计）
    eps = 1e-12  # 防止 log(0)
    R0d5k_log = np.log10(R0d5k_c + eps)
    R1k_log   = np.log10(R1k_c  + eps)
    R3k_log   = np.log10(R3k_c  + eps)
    R6k_log   = np.log10(R6k_c  + eps)
    Rall_log  = np.log10(Rall_c + eps)

    data_log = [R0d5k_log, R1k_log, R3k_log, R6k_log, Rall_log]
    data_raw = [R0d5k_c,   R1k_c,   R3k_c,   R6k_c,   Rall_c]   # 原始空间数据，用来显示原始 median 数值
    labels   = ["500", "1,000", "3,000", "6,000", "11,694"]
    colors   = [c1, c2, c3, c4, c5]
    pos      = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(15, 8))

    # 1) Violin（log 数据）
    parts = ax.violinplot(
        data_log, positions=pos, widths=0.7,
        showmeans=False, showextrema=False, showmedians=False
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(1.0)
        body.set_linewidth(2.0)

    # 2) 散点（log 数据 + KDE）
    rng = np.random.default_rng(0)
    max_points   = 2000
    width_scale  = 0.35
    point_size   = 15
    point_alpha  = 0.20
    point_color  = "black"

    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)

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

    # 3) 统计线（log 空间）+ 在中位数线上写数值（白字）
    for i, vals_log in enumerate(data_log):
        q1, q3 = np.percentile(vals_log, [25, 75])
        med_log = np.median(vals_log)               # log 空间的中位数（用于位置）
        med_raw = np.median(data_raw[i])            # 原始空间中位数（用于显示的数值）
        vmin, vmax = np.min(vals_log), np.max(vals_log)

        # whiskers + IQR
        ax.vlines(pos[i], vmin, vmax, color="black", lw=2.0, alpha=0.75, zorder=2)
        ax.vlines(pos[i], q1, q3, color="black", lw=12.0, alpha=0.75, zorder=2)

        # median line
        ax.hlines(med_log, pos[i]-0.18, pos[i]+0.18, color="black", lw=2.0, zorder=4)

        # 在中位数线上写原始空间的数字（白色）
        ax.text(
            pos[i], med_log,
            f"{med_raw:.3f}",
            color="white",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            zorder=5
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel('number of training samples', fontsize=25)
    ax.set_ylabel(r'$\log_{10}(\mathrm{msFID\ value})$', fontsize=25)
    ax.set_title("msFID Distribution", fontsize=30)

    plt.tight_layout()
    plt.savefig("output/hpa/embed/ablation_num_samp_fid_violin.svg", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    print("saved: output/hpa/embed/ablation_num_samp_fid_violin.svg")

if __name__ == "__main__":
    main()