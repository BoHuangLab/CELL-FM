# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.hpa_data.dataset import HPAImageOnlyDataset
from cell_fm.models.vit_cls.config import ViTConfig
from cell_fm.models.vit_cls.model import ViTModel
from cell_fm.utils.cli_utils import cli

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm

from glob import glob
from PIL import Image

location_colors = {
    'Experiment':      "#0072B2",  # blue
    'Predicted': '#D55E00',  # orange
}


def _trace_sqrt_product(cov1: np.ndarray, cov2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Return Tr( (sqrt(cov1) @ cov2 @ sqrt(cov1))^{1/2} ).
    Uses symmetric eigendecompositions; no SciPy needed.
    """
    # Regularize to avoid singularities
    d = cov1.shape[0]
    I = np.eye(d, dtype=cov1.dtype)
    c1 = cov1 + eps * I
    c2 = cov2 + eps * I

    # Eigen-decomp of cov1: cov1 = U diag(s1) U^T
    s1, U = np.linalg.eigh(c1)
    s1 = np.clip(s1, a_min=0.0, a_max=None)

    # sqrt(cov1) = U diag(sqrt(s1)) U^T
    sqrt_s1 = np.sqrt(s1)
    # Form A = sqrt(cov1) @ cov2 @ sqrt(cov1)  (still symmetric PSD)
    # Compute via change of basis to improve stability:
    # A = U diag(sqrt_s1) U^T @ c2 @ U diag(sqrt_s1) U^T
    #   = U [ diag(sqrt_s1) (U^T c2 U) diag(sqrt_s1) ] U^T
    B = U.T @ c2 @ U
    B = (B + B.T) * 0.5  # symmetrize
    A_tilde = (sqrt_s1[:, None] * B) * sqrt_s1[None, :]

    # sqrt(A) trace = sum(sqrt(eigvals(A)))
    evals, _ = np.linalg.eigh(A_tilde)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    return float(np.sum(np.sqrt(evals)))

def fid_from_embeddings(real: np.ndarray, gen: np.ndarray, eps: float = 1e-6) -> float:
    """
    real, gen: [N, D] embeddings (rows=samples)
    Returns scalar FID.
    """
    assert real.ndim == 2 and gen.ndim == 2, "Embeddings must be 2D (N x D)."

    mu1 = real.mean(axis=0)
    mu2 = gen.mean(axis=0)
    cov1 = np.cov(real, rowvar=False)
    cov2 = np.cov(gen, rowvar=False)

    mean_diff = mu1 - mu2
    m2 = float(mean_diff @ mean_diff)

    tr_cov = float(np.trace(cov1 + cov2))
    tr_sqrt = _trace_sqrt_product(cov1, cov2, eps=eps)

    fid = m2 + tr_cov - 2.0 * tr_sqrt
    # Guard against tiny negative due to numerical error
    return float(max(fid, 0.0))


@cli(ViTConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    trainset = HPAImageOnlyDataset(args, split_key='cellfm_train')
    config = ViTConfig(**vars(args))
    config.num_classes = len(trainset.antibody)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "umap_embedding_data.npz").exists():
        print(f"UMAP embedding data already exists in {output_dir}, loading it.")
        # load saved data
        saved_data = np.load(output_dir / "umap_embedding_data.npz")
        embedding_2d = saved_data['embeddings']
        labels_arr = saved_data['labels']
        print("Data loaded. Proceeding to plotting.")
    else:
        model = ViTModel(config=config)

        model.to(device)
        model.eval()

        cell_image_paths = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/cell_imgs/0002_H3C13/*.png'))
        real_image_paths = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k/cellfm_test/0002_H3C13/real/*.png'))
        pred_image_paths = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k/cellfm_test/0002_H3C13/generated/*.png'))
        # pred_image_paths = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_PUPS/cellfm_test/0002_H3C13/generated/*.png'))

        all_embeddings = []
        all_labels = []

        real_img_embeddings = []
        pred_img_embeddings = []

        for cell_path, real_path, pred_path in tqdm(zip(cell_image_paths, real_image_paths, pred_image_paths), total=len(cell_image_paths), desc="Processing images"):
            cell_img = np.array(Image.open(cell_path)).astype(np.float32) / 255.0
            real_img = np.array(Image.open(real_path)).astype(np.float32) / 255.0
            pred_img = np.array(Image.open(pred_path)).astype(np.float32) / 255.0

            real_img = real_img.mean(axis=2, keepdims=True)
            pred_img = pred_img.mean(axis=2, keepdims=True)

            # normalize to [0, 1]
            real_img = (real_img - real_img.min()) / max((real_img.max() - real_img.min()), 1e-6)
            pred_img = (pred_img - pred_img.min()) / max((pred_img.max() - pred_img.min()), 1e-6)

            real_img = real_img.clip(0, 1)
            pred_img = pred_img.clip(0, 1)

            real_input_img = np.concatenate([cell_img, real_img], axis=2)  # [H, W, 4]
            real_input_img = torch.from_numpy(real_input_img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 4, H, W]

            real_img_embedding = model.embed(real_input_img)
            real_img_embedding = real_img_embedding.detach().cpu().flatten(start_dim=1, end_dim=-1)
            all_embeddings.append(real_img_embedding.numpy())
            all_labels.append('Experiment')

            pred_input_img = np.concatenate([cell_img, pred_img], axis=2)  # [H, W, 4]
            pred_input_img = torch.from_numpy(pred_input_img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 4, H, W]

            pred_img_embedding = model.embed(pred_input_img)
            pred_img_embedding = pred_img_embedding.detach().cpu().flatten(start_dim=1, end_dim=-1)
            all_embeddings.append(pred_img_embedding.numpy())
            all_labels.append('Predicted')

            # real_img_embedding = real_img_embedding / (real_img_embedding.norm(dim=-1, keepdim=True) + 1e-6)
            # pred_img_embedding = pred_img_embedding / (pred_img_embedding.norm(dim=-1, keepdim=True) + 1e-6)

            real_img_embeddings.append(real_img_embedding.numpy())
            pred_img_embeddings.append(pred_img_embedding.numpy())

        real_img_embeddings = np.concatenate(real_img_embeddings, axis=0)
        pred_img_embeddings = np.concatenate(pred_img_embeddings, axis=0)

        fid = fid_from_embeddings(real_img_embeddings, pred_img_embeddings)
        print(f"FID between Experiment and Predicted embeddings: {fid:.4f}")

        embedding_matrix = np.concatenate(all_embeddings, axis=0)   # [N, D]
        labels_arr = np.array(all_labels)                            # ['Experiment'/'Predicted'] * N

        assert embedding_matrix.shape[0] == len(all_labels)

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embedding_matrix)  # [N,2]

        # save embedding_2d and labels
        np.savez_compressed(
            output_dir / "umap_embedding_data.npz",
            embeddings=embedding_2d,
            labels=labels_arr,
        )
    
    # -------------------- plot (refined) --------------------
    palette = {
        "Experiment":location_colors["Experiment"],
        "Predicted": location_colors["Predicted"],
    }
    markers = {"Experiment": "v", "Predicted": "^"}
    sizes   = {"Experiment": 150, "Predicted": 150}

    unique_labels = [lbl for lbl in ["Experiment", "Predicted"] if lbl in set(labels_arr)]
    if not unique_labels:
        raise RuntimeError("No labels to plot. Check data generation above.")

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    # fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    # draw points (pred first, Experiment on top)
    for lbl in ["Predicted", "Experiment"]:
        if lbl not in unique_labels:
            continue
        m = (labels_arr == lbl)
        if not np.any(m):
            continue
        ax.scatter(
            embedding_2d[m, 0],
            embedding_2d[m, 1],
            s=sizes[lbl],
            c=palette[lbl],
            marker=markers[lbl],
            alpha=0.85 if lbl == "Predicted" else 0.95,
            linewidths=1.0,
            edgecolors="white",
            zorder=3 if lbl == "Experiment" else 2,
            rasterized=True,
        )

    # bounds
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    x_pad = 0.04 * (x_max - x_min + 1e-9)
    y_pad = 0.04 * (y_max - y_min + 1e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)

    # optional title (comment out if you want the same style as your example)
    # ax.set_title("UMAP of Image Embeddings")

    # ---- custom L indicator (labels INSIDE) ----
    origin = (0.06, 0.08)   # L corner, axes fraction
    length = 0.14
    lw = 3.8

    # L lines
    ax.plot([origin[0], origin[0]],
            [origin[1], origin[1] + length],
            transform=ax.transAxes, color="black", lw=lw,
            solid_capstyle="butt", clip_on=False)
    ax.plot([origin[0], origin[0] + length],
            [origin[1], origin[1]],
            transform=ax.transAxes, color="black", lw=lw,
            solid_capstyle="butt", clip_on=False)

    # INSIDE text placement:
    # - UMAP2: 放在竖线右侧（内侧）
    ax.text(origin[0] + 0.03,
            origin[1] + length / 2 + 0.03,
            "UMAP2",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            color="black")

    # - UMAP1: 放在横线的上侧（内侧）
    ax.text(origin[0] + length / 2 + 0.03,
            origin[1] + 0.03,
            "UMAP1",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=14,
            color="black")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0],
               marker=markers[lbl], linestyle="",
               markerfacecolor=palette[lbl],
               markeredgecolor="white",
               markeredgewidth=1.0,
               markersize=14,
               label=lbl)
        for lbl in unique_labels
    ]
    ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="0.85", loc="upper right", fontsize=20)

    fig.tight_layout()

    output_path = output_dir / "umap_embedding.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"UMAP plot saved to: {output_path}")

    # save to svg
    output_path_svg = output_dir / "umap_embedding.svg"
    fig.savefig(output_path_svg, dpi=800, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    main()