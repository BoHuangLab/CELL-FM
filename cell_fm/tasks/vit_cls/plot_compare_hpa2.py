# -*- coding: utf-8 -*-

from glob import glob
import pandas as pd
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

    PUPS = np.load('output/hpa/embed/PUPS_fids.npy')
    CD1 = np.load('output/hpa/embed/CELL-Diff_fids.npy')
    CD2 = np.load('output/hpa/embed/CELL-Diff2_fids.npy')
    CD2_no_mae = np.load('output/hpa/embed/CELL-Diff2_no_mae_fids.npy')

    summary = pd.DataFrame(
        {"PUPS": stats(PUPS), "CELL-Diff": stats(CD1), "CELL-Diff2": stats(CD2), "CELL-Diff2 (No MAE)": stats(CD2_no_mae)}
    ).T

    # # Plot 1: Histogram with log-scaled x-axis
    # PUPS_pos = PUPS[np.isfinite(PUPS) & (PUPS > 0)]
    # CD1_pos  = CD1[np.isfinite(CD1) & (CD1 > 0)]
    # CD2_pos  = CD2[np.isfinite(CD2) & (CD2 > 0)]

    # def describe(x):
    #     return {
    #         "n": len(x),
    #         "mean": float(np.mean(x)),
    #         "median": float(np.median(x)),
    #         "p25": float(np.percentile(x, 25)),
    #         "p75": float(np.percentile(x, 75)),
    #         "min": float(np.min(x)),
    #         "max": float(np.max(x)),
    #     }

    # d1, d2, d3 = describe(PUPS_pos), describe(CD1_pos), describe(CD2_pos)

    # # 使用 log 等距的 bins 与 log x 轴匹配
    # xmin = max(1e-6, min(d1["min"], d2["min"], d3["min"]))
    # xmax = max(d1["max"], d2["max"], d3["max"])
    # bins = np.logspace(np.log10(xmin), np.log10(xmax), 80)

    # fig, ax = plt.subplots(figsize=(12, 7))

    # c1 = "#0072B2"  # blue
    # c2 = "#E69F00"  # orange
    # c3 = "#009E73"  # green

    # # --- 绘制直方图，并读取每个数据集的颜色（与 rcParams palette 一致） ---
    # h1 = ax.hist(
    #     PUPS_pos, bins=bins, alpha=0.45, density=True, label="PUPS (Median={:.2f})".format(d1["median"]), color=c1
    # )
    # h2 = ax.hist(
    #     CD1_pos,  bins=bins, alpha=0.45, density=True, label="CELL-Diff (Median={:.2f})".format(d2["median"]), color=c2
    # )
    # h3 = ax.hist(
    #     CD2_pos,  bins=bins, alpha=0.45, density=True, label="CELL-Diff2 (Median={:.2f})".format(d3["median"]), color=c3
    # )

    # # 从每个直方图的第一个 patch 读取颜色（RGBA）
    # cline1 = "#000000"  # blue
    # cline2 = "#000000"  # orange
    # cline3 = "#000000"  # green

    # ax.set_xscale("log")

    # from matplotlib.ticker import LogLocator, LogFormatter
    # ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=12))
    # ax.xaxis.set_major_formatter(LogFormatter(base=10.0))
    # ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.5)
    # ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # ax.set_xlabel("FID value (log scale)")
    # ax.set_ylabel("Density")
    # ax.set_title("FID Distribution (log-scale histogram)")

    # # --- 用与直方图相同的颜色标注 median 与 IQR ---
    # # median 虚线
    # ax.axvline(d1["median"], color=cline1, linewidth=1.5, linestyle="--")
    # ax.axvline(d2["median"], color=cline2, linewidth=1.5, linestyle="--")
    # ax.axvline(d3["median"], color=cline3, linewidth=1.5, linestyle="--")

    # # IQR 阴影（使用相同颜色但更低透明度）
    # ax.axvspan(d1["p25"], d1["p75"], facecolor=cline1, alpha=0.12, edgecolor="none")
    # ax.axvspan(d2["p25"], d2["p75"], facecolor=cline2, alpha=0.12, edgecolor="none")
    # ax.axvspan(d3["p25"], d3["p75"], facecolor=cline3, alpha=0.12, edgecolor="none")

    # # --- 图例：数据集图例 + 说明 median/IQR 的样式图例 ---
    # # 第一层：数据集（来自 hist 的 label）
    # leg1 = ax.legend(
    #     loc="upper right",
    #     frameon=False,
    #     fontsize=12,       # 调整文字大小
    #     handlelength=2.5,  # 控制线条长度
    #     handleheight=1.5   # 控制色块高度
    # )
    
    # # 第二层：样式说明（不区分颜色，仅说明含义）
    # from matplotlib.lines import Line2D
    # from matplotlib.patches import Patch
    # median_proxy = Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="Median")
    # iqr_proxy = Patch(facecolor="black", alpha=0.12, label="IQR (p25-p75)")
    # # leg2 = ax.legend(handles=[median_proxy, iqr_proxy], loc="upper left", frameon=False)
    # leg2 = ax.legend(
    #     handles=[median_proxy, iqr_proxy], 
    #     loc="upper left", 
    #     frameon=False, 
    #     fontsize=12,       # 调整文字大小
    #     handlelength=2.5,  # 控制线条长度
    #     handleheight=1.5   # 控制色块高度
    # )
    # ax.add_artist(leg1)  # 保留第一层图例

    # plt.tight_layout()
    # hist_path = "output/hpa/embed/fid_histogram_log.png"
    # plt.savefig(hist_path, dpi=240)


    # # Plot 2: Violin
    # plt.figure(figsize=(8, 6))
    # plt.violinplot([PUPS, CD1, CD2], showmeans=True, showextrema=True, showmedians=True)
    # plt.xticks([1, 2, 3], ["PUPS", "CELL-Diff", "CELL-Diff2"])
    # plt.ylabel("FID value")
    # plt.title("FID Distribution (Violin Plot)")
    # vio_path = "output/hpa/embed/fid_violin.png"
    # plt.tight_layout()
    # plt.savefig(vio_path, dpi=200)


    # # Plot 2: Violin + strip (jitter) points, log y-axis
    # c1 = "#8cbfdc"  # blue
    # c2 = "#f3d38c"  # orange
    # c3 = "#8cd3bf"  # green

    # def clean(x):
    #     return np.asarray(x)[np.isfinite(x)]

    # PUPS_c = clean(PUPS)
    # CD1_c  = clean(CD1)
    # CD2_c  = clean(CD2)
    # data   = [PUPS_c, CD1_c, CD2_c]
    # labels = ["PUPS", "CELL-Diff", "CELL-Diff2"]
    # colors = [c1, c2, c3]
    # pos    = [1, 2, 3]

    # fig, ax = plt.subplots(figsize=(10, 6))

    # # 1) Violin，边框改为黑色
    # parts = ax.violinplot(
    #     data, positions=pos, widths=0.7,
    #     showmeans=False, showextrema=False, showmedians=False
    # )
    # for i, body in enumerate(parts['bodies']):
    #     body.set_facecolor(colors[i])
    #     body.set_edgecolor("black")   # 👈 边框改成黑色
    #     body.set_alpha(1.0)
    #     body.set_linewidth(2.0)

    # # 2) 散点：改为灰色，仍然放在 violin 内部
    # rng = np.random.default_rng(0)
    # max_points   = 2000
    # width_scale  = 0.35
    # point_size   = 15
    # point_alpha  = 0.20
    # point_color  = "black"  # 👈 散点颜色改成黑色

    # for i, vals in enumerate(data):
    #     y = vals
    #     n = len(y)
    #     if n > max_points:
    #         idx = rng.choice(n, max_points, replace=False)
    #         y = y[idx]

    #     y_std = float(np.std(y))
    #     if y_std <= 1e-12:
    #         widths = np.full_like(y, fill_value=0.06, dtype=float)
    #     else:
    #         y_lo, y_hi = np.percentile(y, [1, 99])
    #         if y_hi <= y_lo:
    #             y_lo, y_hi = np.min(y), np.max(y)
    #         y_grid = np.linspace(y_lo, y_hi, 256)
    #         kde = gaussian_kde(y)

    #         dens = kde(y_grid)
    #         dens = np.maximum(dens, 1e-12)
    #         dens = dens / dens.max()
    #         dens = dens * width_scale

    #         idx_near = np.abs(y[:, None] - y_grid[None, :]).argmin(axis=1)
    #         widths   = dens[idx_near]

    #     x = pos[i] + rng.uniform(-1.0, 1.0, size=len(y)) * widths
    #     ax.scatter(x, y, s=point_size, alpha=point_alpha,
    #             color=point_color, edgecolors="none", zorder=3)

    # # 3) 自定义统计线（IQR + median）
    # for i, vals in enumerate(data):
    #     q1, q3 = np.percentile(vals, [25, 75])
    #     med    = np.median(vals)
    #     vmin, vmax = np.min(vals), np.max(vals)
    #     ax.vlines(pos[i], vmin, vmax, color="black", lw=2.0, alpha=0.75, zorder=2)
    #     ax.vlines(pos[i], q1, q3, color="black", lw=12.0, alpha=0.75, zorder=2)
    #     ax.hlines(med, pos[i]-0.18, pos[i]+0.18, color="black", lw=2.0, zorder=4)

    # ax.set_xticks(pos)
    # ax.set_xticklabels(labels)
    # ax.set_ylabel("FID value (log scale)")
    # ax.set_title("FID Distribution (log-scale violin plot)")
    # ax.set_yscale("log")

    # # 图例只保留数据集色块
    # from matplotlib.patches import Patch
    # dataset_handles = [Patch(facecolor=colors[i], edgecolor="black", alpha=1.0, label=labels[i]) for i in range(3)]
    # ax.legend(handles=dataset_handles, loc="upper right", frameon=False, fontsize=12,
    #         handlelength=2.5, handleheight=1.5)

    # plt.tight_layout()
    # plt.savefig("output/hpa/embed/fid_violin_scatter_gray_logy.png", dpi=200)



    # # Plot 3: ECDF
    # plt.figure(figsize=(10, 6))
    # def ecdf_values(x):
    #     xs = np.sort(x)
    #     ys = np.linspace(0, 1, len(xs), endpoint=True)
    #     return xs, ys

    # for vals, name in [(PUPS, "PUPS"), (CD1, "CELL-Diff"), (CD2, "CELL-Diff2")]:
    #     xs, ys = ecdf_values(vals)
    #     plt.plot(xs, ys, label=name)

    # plt.xlabel("FID value")
    # plt.ylabel("Empirical CDF")
    # plt.title("FID Empirical CDF Comparison")
    # plt.legend()
    # cdf_path = "output/hpa/embed/fid_ecdf.png"
    # plt.tight_layout()
    # plt.savefig(cdf_path, dpi=200)

    # # Plot 4: Per-index line comparison (each sample index vs FID)
    # plt.figure(figsize=(12, 6))
    # idx = np.arange(len(PUPS))
    # plt.plot(idx, PUPS, label="PUPS", linewidth=1.0, alpha=0.8)
    # plt.plot(idx, CD1, label="CELL-Diff", linewidth=1.0, alpha=0.8)
    # plt.plot(idx, CD2, label="CELL-Diff2", linewidth=1.0, alpha=0.8)

    # plt.xlabel("Sample Index")
    # plt.ylabel("FID value")
    # plt.title("FID Per-Index Comparison (Line Plot)")
    # plt.legend(ncol=3, frameon=False)
    # plt.tight_layout()
    # line_path = "output/hpa/embed/fid_line_per_index.png"
    # plt.savefig(line_path, dpi=200)

    # # 可选：增加滚动中位数平滑一条辅助线，便于观察趋势（不会覆盖上面的图）
    # window = max(5, len(PUPS)//100)  # 自适应窗口大小
    # if len(PUPS) >= window:
    #     import pandas as _pd
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(idx, _pd.Series(PUPS).rolling(window, center=True).median(),
    #              label=f"PUPS (rolling median, w={window})", linewidth=2, alpha=0.9)
    #     plt.plot(idx, _pd.Series(CD1).rolling(window, center=True).median(),
    #              label=f"CELL-Diff (rolling median, w={window})", linewidth=2, alpha=0.9)
    #     plt.plot(idx, _pd.Series(CD2).rolling(window, center=True).median(),
    #              label=f"CELL-Diff2 (rolling median, w={window})", linewidth=2, alpha=0.9)
    #     plt.xlabel("Sample Index")
    #     plt.ylabel("FID value")
    #     plt.title("FID Per-Index Rolling-Median Trend")
    #     plt.legend(ncol=1, frameon=False)
    #     plt.tight_layout()
    #     plt.savefig("output/hpa/embed/fid_line_per_index_trend.png", dpi=200)

    combined_path = None
    if len(PUPS) == len(CD1) == len(CD2):
        df = pd.DataFrame({"PUPS": PUPS, "CELL-Diff": CD1, "CELL-Diff2": CD2, "CELL-Diff2 (No MAE)": CD2_no_mae})
        combined_path = "output/hpa/embed/fid_per_index_values.csv"
        df.to_csv(combined_path, index_label="index")


if __name__ == "__main__":
    main()