# -*- coding: utf-8 -*-
"""
HPA virtual-staining dict intensity analysis.

For each mutant subfolder, loads all generated PNGs and computes per-image:
    score = mean(inside nucleus) - mean(inside cell but outside nucleus)

where:
    nucleus mask  = chosen_nucleus_masks.png  (any non-zero pixel)
    cell mask     = chosen_ER_masks.png       (any non-zero pixel)
    cytoplasm     = cell_mask & ~nucleus_mask

Plots all 32 per-image scores as a scatter at each window position,
with a mean line connecting subfolder means.

Folder naming conventions (auto-detected from DATA_DIR name):
  PRRSV-section-<W>  subfolders like "22-41"
  PRRSV-win-<W>      subfolders like "del1-20"
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR     = Path("/hpc/projects/group.huang/dihan.zheng/CELL-FM/output/hpa/virtual_staining_dict")
NUCL_MASK_PATH = BASE_DIR / "chosen_nucleus_masks.png"
CELL_MASK_PATH = BASE_DIR / "chosen_ER_masks.png"

DATA_DIR       = BASE_DIR / "PRRSV-section-25"
OUT_DIFF       = BASE_DIR / "PRRSV-section-25_intensity_diff.png"
OUT_RATIO      = BASE_DIR / "PRRSV-section-25_intensity_ratio.png"
OUT_RATIO_INV  = BASE_DIR / "PRRSV-section-25_intensity_ratio_inv.png"

# DATA_DIR       = BASE_DIR / "PRRSV-section-20-GSx3"
# OUT_DIFF       = BASE_DIR / "PRRSV-section-20-GSx3_intensity_diff.png"
# OUT_RATIO      = BASE_DIR / "PRRSV-section-20-GSx3_intensity_ratio.png"
# OUT_RATIO_INV  = BASE_DIR / "PRRSV-section-20-GSx3_intensity_ratio_inv.png"

# ── infer experiment type ─────────────────────────────────────────────────────
_parts       = DATA_DIR.name.split("-")
EXP_TYPE     = _parts[1]       # "section" or "win"
PROTEIN_NAME = _parts[0]
WINDOW_SIZE  = int(_parts[2])


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return arr[:, :, 0] > 0    # bool (H, W)


def image_to_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0


def per_image_scores(img_path: Path, nucl_mask: np.ndarray, cyto_mask: np.ndarray) -> dict | None:
    """Returns both diff and ratio of nucleus vs. cytoplasm mean intensity."""
    gray = image_to_gray(img_path)
    inside  = gray[nucl_mask]
    outside = gray[cyto_mask]
    if len(inside) == 0 or len(outside) == 0:
        return None
    m_in  = float(inside.mean())
    m_out = float(outside.mean())
    return {"diff": m_in - m_out, "ratio": m_in / (m_out + 1e-8),
            "inv_ratio": m_out / (m_in + 1e-8)}


def parse_start(name: str, exp_type: str) -> int:
    if exp_type == "section":
        return int(name.split("-")[0])
    else:
        return int(name.removeprefix("del").split("-")[0])


def main():
    nucl_mask = load_binary_mask(NUCL_MASK_PATH)
    cell_mask = load_binary_mask(CELL_MASK_PATH)
    cyto_mask = cell_mask & ~nucl_mask       # cell body excluding nucleus

    print(f"Experiment : {EXP_TYPE}  window={WINDOW_SIZE}")
    print(f"Nucleus    : {nucl_mask.sum()} px")
    print(f"Cytoplasm  : {cyto_mask.sum()} px")

    results = []
    for subfolder in DATA_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        name  = subfolder.name
        imgs  = sorted(subfolder.glob("*.png"))
        raw   = [per_image_scores(p, nucl_mask, cyto_mask) for p in imgs]
        raw   = [s for s in raw if s is not None]
        if not raw:
            continue
        diffs      = [s["diff"]      for s in raw]
        ratios     = [s["ratio"]     for s in raw]
        inv_ratios = [s["inv_ratio"] for s in raw]
        start  = parse_start(name, EXP_TYPE)
        results.append({
            "name": name, "start": start,
            "diffs":      diffs,      "median_diff":      float(np.median(diffs)),
            "ratios":     ratios,     "median_ratio":     float(np.median(ratios)),
            "inv_ratios": inv_ratios, "median_inv_ratio": float(np.median(inv_ratios)),
        })
        print(f"  {name:>10}  n={len(diffs)}  diff={np.median(diffs):+.4f}  ratio={np.median(ratios):.4f}  inv_ratio={np.median(inv_ratios):.4f}")

    results.sort(key=lambda r: r["start"])

    if EXP_TYPE == "section":
        xlabel = f"Section window (start-end, size={WINDOW_SIZE})"
        title  = f"{PROTEIN_NAME} sliding-window section: nucleus vs. cytoplasm intensity"
        color  = "#4CAF50"
    else:
        xlabel = f"Deletion window (start-end, size={WINDOW_SIZE})"
        title  = f"{PROTEIN_NAME} sliding-window deletion: nucleus vs. cytoplasm intensity"
        color  = "#2196F3"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 11,
    })

    starts = [r["start"] for r in results]
    names  = [r["name"]  for r in results]
    rng    = np.random.default_rng(0)

    def save_scatter_plot(scores_key: str, means_key: str, ylabel: str, hline: float,
                          out_path: Path, ylim: tuple | None = None):
        fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
        means_vals = [r[means_key] for r in results]
        for r in results:
            jitter = rng.uniform(-0.3, 0.3, size=len(r[scores_key]))
            ax.scatter(r["start"] + jitter, r[scores_key],
                       s=10, color=color, alpha=0.35, zorder=2, linewidths=0)
        ax.plot(starts, means_vals, color=color, linewidth=1.2, zorder=3)
        ax.scatter(starts, means_vals, s=30, color=color, zorder=4, label="median per window")
        ax.axhline(hline, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(starts)
        ax.set_xticklabels(names, rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)
        ax.legend(frameon=False)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Plot saved: {out_path}")

    save_scatter_plot("diffs",      "median_diff",        "Median intensity\n(nucleus − cytoplasm)",    0,   OUT_DIFF)
    save_scatter_plot("ratios",     "median_ratio",     "Median intensity ratio\n(nucleus / cytoplasm)", 1.0, OUT_RATIO,
                      ylim=(0, 10))
    save_scatter_plot("inv_ratios", "median_inv_ratio", "Median intensity ratio\n(cytoplasm / nucleus)", 1.0, OUT_RATIO_INV,
                      ylim=(0, 5))


if __name__ == "__main__":
    main()
