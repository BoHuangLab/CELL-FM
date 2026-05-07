import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import json

def compute_gap_with_points(y: np.ndarray):
    """
    返回:
      m_gap, max_idx, min_idx, y_max, y_min
    其中 min_idx 是 max_idx 之后的最小值点索引。
    """
    y = np.asarray(y, dtype=float)
    max_idx = int(np.argmax(y))
    y_max = float(y[max_idx])

    if max_idx == len(y) - 1:
        return 0.0, max_idx, max_idx, y_max, y_max

    tail = y[max_idx + 1:]
    rel_min_idx = int(np.argmin(tail))
    min_idx = max_idx + 1 + rel_min_idx
    y_min = float(y[min_idx])

    return (y_max - y_min), max_idx, min_idx, y_max, y_min

def moving_average(y, window: int):
    y = np.asarray(y, dtype=float)
    window = int(max(1, min(window, len(y))))
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

def _infer_max_workers():
    """
    HPC/SLURM 常用：优先用 SLURM_CPUS_PER_TASK，否则用 os.cpu_count().
    """
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v is not None:
        try:
            return max(1, int(v))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)

def process_one_protein(data_path: str, window_size: int = 32):
    # 进程内使用无界面后端，避免并行保存图时报错
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
    })

    data_root = Path(data_path)
    csv_path = data_root / "pred_vs_intensity.csv"

    if not csv_path.exists():
        # 缺文件就跳过
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    needed = {"protein_intensity_level", "predicted_class"}
    if not needed.issubset(df.columns):
        return None

    df = df[["protein_intensity_level", "predicted_class"]].copy()
    df["protein_intensity_level"] = pd.to_numeric(df["protein_intensity_level"], errors="coerce")
    df["predicted_class"] = pd.to_numeric(df["predicted_class"], errors="coerce")
    df = df.dropna()

    if len(df) < 5:
        return None

    df = df.sort_values("protein_intensity_level", kind="mergesort")
    x = df["protein_intensity_level"].to_numpy()
    y = df["predicted_class"].to_numpy()

    y_ma = moving_average(y, window_size)
    m_gap, max_idx, min_idx, y_max, y_min = compute_gap_with_points(y_ma)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    step = max(1, len(x) // 3000)
    ax.scatter(x[::step], y[::step], s=6, alpha=0.08, linewidths=0)
    ax.plot(x, y_ma, linewidth=2.5)

    ax.set_xlabel("Protein Intensity Level")
    ax.set_ylabel("Predicted Condensate Probability (MA)")

    img_title = '_'.join(data_path.split('/')[-2:])
    ax.set_title(f"{img_title}: Condensate probability vs intensity")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    # 原图
    save_path = data_root / "predicted_condensate_probability_ma.png"
    try:
        fig.savefig(save_path, bbox_inches="tight")
    except Exception:
        plt.close(fig)
        return None

    # 标注图
    ax.scatter([x[max_idx]], [y_max], s=200, marker="^", zorder=6,
               label="max", color="#B60000", edgecolors="white", linewidths=2.0)
    ax.scatter([x[min_idx]], [y_min], s=200, marker="v", zorder=6,
               label="min after max", color="#E4E100", edgecolors="white", linewidths=2.0)
    ax.plot([x[max_idx], x[min_idx]], [y_max, y_min], linestyle="--", zorder=5)

    ax.annotate(
        f"max\n({y_max:.3f})",
        xy=(x[max_idx], y_max),
        xytext=(10, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="right",
        va="bottom",
        zorder=7,
        size=10,
    )
    ax.annotate(
        f"min after max\n({y_min:.3f})",
        xy=(x[min_idx], y_min),
        xytext=(10, -14),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="right",
        va="top",
        zorder=7,
    )

    ax.text(
        0.95, 0.95,
        f"Gap = {m_gap:.4f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.85, facecolor="white"),
        zorder=12,
        size=12,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    save_path2 = data_root / "predicted_condensate_probability_ma_mgap.png"
    try:
        fig.savefig(save_path2, bbox_inches="tight")
    except Exception:
        plt.close(fig)
        return None
    finally:
        plt.close(fig)

    # save results in json to data_root
    json_path = data_root / "predicted_condensate_probability_ma_mgap.json"
    try:
        with open(json_path, "w") as f:
            json.dump({
                "max_gap": float(m_gap),
                "y_max": y_max,
                "y_min": y_min,
            }, f, indent=2)
    except Exception:
        pass

    return (float(m_gap), y_max, y_min)

def main():
    data_paths = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_selected_test_data/*/*/'))

    # 更快的遍历方式：itertuples
    tasks = data_paths

    max_workers = _infer_max_workers()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_protein, data_path, 32)
            for data_path in tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel processing ({max_workers} workers)"):
            try:
                res = fut.result()
            except Exception:
                continue
            if res is not None:
                results.append(res)

if __name__ == "__main__":
    main()