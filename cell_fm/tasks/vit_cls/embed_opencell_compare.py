# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path
from cell_fm.models.vit_cls.config import ViTConfig
from cell_fm.models.vit_cls.model import ViTModel
from cell_fm.utils.cli_utils import cli

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm

from glob import glob
from PIL import Image

import tifffile as tiff
from torchvision.transforms.functional import to_tensor


location_colors = {
    'Real':      "#0072B2",  # blue
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


def prepare_img(img: np.ndarray, clip=False) -> torch.Tensor:
    # img is a 2D array with shape (2, H, W) where the first channel is nucleus and the second channel is protein
    nucleus_img = img[0]
    protein_img = img[1]

    if clip:
        # clip the images to 1% - 99% range through np.percentile
        nucleus_img = np.clip(nucleus_img, np.percentile(nucleus_img, 1), np.percentile(nucleus_img, 99))
        protein_img = np.clip(protein_img, np.percentile(protein_img, 1), np.percentile(protein_img, 99))

    # normalize the images to [0, 1]
    nucleus_img = (nucleus_img - nucleus_img.min()) / (nucleus_img.max() - nucleus_img.min())
    protein_img = (protein_img - protein_img.min()) / (protein_img.max() - protein_img.min())

    nucleus_img = to_tensor(nucleus_img)
    protein_img = to_tensor(protein_img)

    img = torch.cat([protein_img, nucleus_img], dim=0)

    return img


@cli(ViTConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = ViTConfig(**vars(args))
    config.num_classes = 1311

    model = ViTModel(config=config)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_image_dirs = sorted(glob('/hpc/reference/opencell/opencell/2d_proj_256_crop_dataset/*/'))
    pred_image_dirs = sorted(glob('/hpc/reference/opencell/opencell/2d_proj_256_crop_dataset_virtual_staining_same_nucl/*/'))

    fids = []

    for pred_dir in pred_image_dirs:
        gene_name = os.path.basename(os.path.dirname(pred_dir)).split('_')[0]
        matching_real_dirs = [d for d in real_image_dirs if os.path.basename(os.path.dirname(d)).split('_')[0] == gene_name]
        if not matching_real_dirs:
            print(f"No matching real image directory found for gene: {gene_name}")
            continue
        real_dir = matching_real_dirs[0]

        pred_img_paths = sorted(glob(os.path.join(pred_dir, '*.tif')))
        real_img_paths = sorted(glob(os.path.join(real_dir, '*.tif')))

        real_img_embeddings = []
        pred_img_embeddings = []

        for pred_img_path, real_img_path in zip(pred_img_paths, real_img_paths):
            pred_img = tiff.imread(pred_img_path).astype(np.float32)
            real_img = tiff.imread(real_img_path).astype(np.float32)

            pred_img = prepare_img(pred_img, clip=False).unsqueeze(0).to(device)
            real_img = prepare_img(real_img, clip=True).unsqueeze(0).to(device)

            pred_img_embedding = model.embed(pred_img)
            pred_img_embedding = pred_img_embedding.detach().cpu().flatten(start_dim=1, end_dim=-1)
            pred_img_embeddings.append(pred_img_embedding.numpy())

            real_img_embedding = model.embed(real_img)
            real_img_embedding = real_img_embedding.detach().cpu().flatten(start_dim=1, end_dim=-1)
            real_img_embeddings.append(real_img_embedding.numpy())

        real_img_embeddings = np.concatenate(real_img_embeddings, axis=0)
        pred_img_embeddings = np.concatenate(pred_img_embeddings, axis=0)

        fid = fid_from_embeddings(real_img_embeddings, pred_img_embeddings)
        print(f"Gene: {gene_name}, FID between Real and Predicted embeddings: {fid:.4f}")
        fids.append((gene_name, fid))

    # save fids to csv
    with open(output_dir / 'fids.txt', 'w') as f:
        f.write("All FIDs:\n")
        for fid in fids:
            gene_name, fid = fid
            f.write(f"{gene_name}: {fid:.4f}\n")

if __name__ == "__main__":
    main()
