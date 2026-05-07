# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import numpy as np
from tqdm import tqdm
from glob import glob


# -----------------------------
# FID
# -----------------------------
def _trace_sqrt_product(cov1: np.ndarray, cov2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Return Tr( (sqrt(cov1) @ cov2 @ sqrt(cov1))^{1/2} ).
    Uses symmetric eigendecompositions; no SciPy needed.
    """
    d = cov1.shape[0]
    I = np.eye(d, dtype=cov1.dtype)
    c1 = cov1 + eps * I
    c2 = cov2 + eps * I

    s1, U = np.linalg.eigh(c1)
    s1 = np.clip(s1, a_min=0.0, a_max=None)

    sqrt_s1 = np.sqrt(s1)
    B = U.T @ c2 @ U
    B = (B + B.T) * 0.5
    A_tilde = (sqrt_s1[:, None] * B) * sqrt_s1[None, :]

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
    return float(max(fid, 0.0))


# -----------------------------
# MMD (RBF kernel)
# -----------------------------
def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    ab = A @ B.T
    return np.maximum(aa + bb - 2.0 * ab, 0.0)

def mmd_rbf_full(X: np.ndarray,
                 Y: np.ndarray,
                 sigma: float | None = None,
                 unbiased: bool = True) -> float:
    """
    FULL-sample MMD^2 with RBF kernel. No subsampling.
    If sigma is None, use median heuristic on ALL pooled data.
    """
    X = np.asarray(X); Y = np.asarray(Y)

    if sigma is None:
        Z = np.vstack([X, Y])
        D2 = _pairwise_sq_dists(Z, Z)
        iu = np.triu_indices_from(D2, k=1)
        med = np.median(np.sqrt(D2[iu]))
        sigma = max(float(med), 1e-6)
    gamma = 1.0 / (2.0 * sigma * sigma)

    Kxx = np.exp(-gamma * _pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * _pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _pairwise_sq_dists(X, Y))

    if unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        m = len(X); n = len(Y)
        term_xx = Kxx.sum() / (m * (m - 1))
        term_yy = Kyy.sum() / (n * (n - 1))
        term_xy = 2.0 * Kxy.mean()
    else:
        term_xx = Kxx.mean()
        term_yy = Kyy.mean()
        term_xy = 2.0 * Kxy.mean()

    return float(term_xx + term_yy - term_xy)  # MMD^2


# # -----------------------------
# # KID (Polynomial kernel, unbiased) - FULL (no subsampling)
# # k(x,y) = ((x·y)/d + 1)^3
# # -----------------------------
# def _kid_poly_kernel(A: np.ndarray, B: np.ndarray, degree: int = 3) -> np.ndarray:
#     d = A.shape[1]
#     return ((A @ B.T) / d + 1.0) ** degree

# def kid_polynomial_full_unbiased(X: np.ndarray,
#                                  Y: np.ndarray,
#                                  degree: int = 3) -> tuple[float, float]:
#     """
#     FULL-sample, single-shot unbiased KID estimate (no subsets).
#     Returns (kid_mean, kid_std) where kid_std=0.0 by definition.
#     """
#     X = np.asarray(X); Y = np.asarray(Y)
#     Kxx = _kid_poly_kernel(X, X, degree=degree)
#     Kyy = _kid_poly_kernel(Y, Y, degree=degree)
#     Kxy = _kid_poly_kernel(X, Y, degree=degree)

#     np.fill_diagonal(Kxx, 0.0)
#     np.fill_diagonal(Kyy, 0.0)
#     m = len(X); n = len(Y)

#     kid = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean()
#     return float(kid), 0.0

def kid_poly_biased(X, Y, degree=3):
    X = np.asarray(X); Y = np.asarray(Y)
    d = X.shape[1]
    Kxx = ((X @ X.T) / d + 1.0) ** degree
    Kyy = ((Y @ Y.T) / d + 1.0) ** degree
    Kxy = ((X @ Y.T) / d + 1.0) ** degree
    kid = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(max(kid, 0.0))  # numerical: clip ≥ 0


def main() -> None:

    real_image_embeddings = sorted(glob('output/hpa/embed/real/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/PUPS/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/CELL-Diff/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/CELL-Diff2/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/CELL-Diff2_no_mae/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_6k/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/CELL-Diff2_S2_R1_Rerun_50k/*.npy'))
    # pred_image_embeddings = sorted(glob('output/hpa/embed/PT_HPA_CELL-Diff2_cd2split_ps8/*.npy'))
    pred_image_embeddings = sorted(glob('output/hpa/embed/PT_HPA_CELL-Diff2_Rep_Dev_NH8_S2_R1_50k/*.npy'))

    fids = []
    mmds = []        # MMD^2
    kids = []        # KID mean over subsets

    rng = np.random.default_rng(0)

    for real_image_embedding, pred_image_embedding in tqdm(
        zip(real_image_embeddings, pred_image_embeddings),
        total=len(real_image_embeddings), desc="Processing"
    ):
        prot_name = os.path.basename(real_image_embedding).split('_')[1]

        real_embeddings = np.load(real_image_embedding)  # [N, D]
        pred_embeddings = np.load(pred_image_embedding)  # [M, D]

        # FID
        fid = fid_from_embeddings(real_embeddings, pred_embeddings)
        fids.append(fid)

        # MMD
        mmd2 = mmd_rbf_full(real_embeddings, pred_embeddings,
                            sigma=None, unbiased=True)
        mmds.append(mmd2)

        # KID
        kid = kid_poly_biased(
            real_embeddings, pred_embeddings, degree=3
        )
        kids.append(kid)

    fids = np.asarray(fids, dtype=np.float64)
    mmds = np.asarray(mmds, dtype=np.float64)
    kids = np.asarray(kids, dtype=np.float64)

    # summary
    print(f"\n=== Summary over {len(fids)} proteins ===")
    print(f"FID        : mean={fids.mean():.4f} ± {fids.std(ddof=1):.4f} | median={np.median(fids):.4f}")
    print(f"MMD_RBF^2  : mean={mmds.mean():.6f} ± {mmds.std(ddof=1):.6f} | median={np.median(mmds):.6f}")
    print(f"KID        : mean={kids.mean():.6f} ± {kids.std(ddof=1):.6f} | median={np.median(kids):.6f}")

    # tag
    pred_dir = os.path.dirname(pred_image_embeddings[0]) if pred_image_embeddings else "output/hpa/embed/unknown"
    tag = os.path.basename(pred_dir)

    # save output dir
    os.makedirs(f'output/hpa/embed/{tag}_results/', exist_ok=True)

    # save
    np.save(f'output/hpa/embed/{tag}_results/fids.npy', fids)
    np.save(f'output/hpa/embed/{tag}_results/mmd2.npy', mmds)
    np.save(f'output/hpa/embed/{tag}_results/kids.npy', kids)

if __name__ == "__main__":
    main()