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

import numpy as np
from tqdm import tqdm

from glob import glob
from PIL import Image

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


@cli(ViTConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = HPAImageOnlyDataset(args, split_key='cellfm_train')
    config = ViTConfig(**vars(args))
    config.num_classes = len(trainset.antibody)

    model = ViTModel(config=config)

    model.to(device)
    model.eval()

    # config.output_dir = "./output/hpa/embed/CELL-Diff2_S2_R2_50k/"
    # config.output_dir = "./output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_0d5k/"
    # config.output_dir = "./output/hpa/embed/CELL-Diff2_S2_R1_Rerun_50k/"
    # config.output_dir = "./output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_ps8/"
    config.output_dir = "./output/hpa/embed/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k/"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/cell_imgs/*/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_PUPS/cellfm_test/*/real/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_PUPS/cellfm_test/*/generated/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_CELL-Diff/cellfm_test/*/generated/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_Rep_Dev_NH8/cellfm_test/*/generated/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_cd2split_0d5k/cellfm_test/*/generated/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_Rerun_50k/cellfm_test/*/generated/'))
    # prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_cd2split_ps8/cellfm_test/*/generated/'))
    prot_image_dirs = sorted(glob('/hpc/reference/opencell/human_protein_atlas/seq2img/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k/cellfm_test/*/generated/'))

    for cdir, pdir in tqdm(zip(cell_image_dirs, prot_image_dirs), total=len(cell_image_dirs), desc="Processing"):
        prot_name = os.path.basename(os.path.dirname(cdir))

        cell_image_paths = sorted(glob(os.path.join(cdir, '*.png')))
        prot_image_paths = sorted(glob(os.path.join(pdir, '*.png')))

        embeddings = []

        for cell_path, prot_path in zip(cell_image_paths, prot_image_paths):
            cell_img = np.array(Image.open(cell_path)).astype(np.float32) / 255.0
            prot_img = np.array(Image.open(prot_path)).astype(np.float32) / 255.0

            prot_img = prot_img.mean(axis=2, keepdims=True)  # to grayscale

            # normalize to [0, 1]
            prot_img = (prot_img - prot_img.min()) / max((prot_img.max() - prot_img.min()), 1e-6)
            prot_img = prot_img.clip(0, 1)

            input_img = np.concatenate([cell_img, prot_img], axis=2)  # [H, W, 4]
            input_img = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 4, H, W]

            embedding = model.embed(input_img)
            embedding = embedding.detach().cpu().flatten(start_dim=1, end_dim=-1)
            embeddings.append(embedding.numpy())
        
        embedding_matrix = np.concatenate(embeddings, axis=0)  # [N, D]

        # save embeddings
        np.save(output_dir / f"{prot_name}_embeddings.npy", embedding_matrix)

if __name__ == "__main__":
    main()