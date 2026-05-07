# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.opencell_crop_data.dataset import OpenCellCropImageEmbedDataset
from cell_fm.models.vit_cls.config import ViTConfig
from cell_fm.models.vit_cls.model import ViTModel
from cell_fm.utils.cli_utils import cli

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score, v_measure_score, fowlkes_mallows_score,
    normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, rand_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


location_colors = {
    "Multilocalizing": "#999999",
    "big_aggregates": "#D50000",
    "cell_contact": "#018600",
    "centrosome": "#b400fe",
    "chromatin": "#04abc5",
    "cytoplasmic": "#96fe00",
    "cytoskeleton": "#fea42e",
    "er": "#fe8dc7",
    "focal_adhesions": "#78525d",
    "golgi": "#00fcce",
    "membrane": "#aea4fe",
    "mitochondria": "#92ab82",
    "negative": "#996900",
    "nuclear_membrane": "#356961",
    "nuclear_punctae": "#d2008b",
    "nucleolus_fc_dfc": "#fcf490",
    "nucleolus_gc": "#c76d66",
    "nucleoplasm": "#9de1fe",
    "vesicles": "#00c746",
    "NA": "#999999"
}


# ----- helpers (external metrics) -----
def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)  # rows: true, cols: pred
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

def b3_scores(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred).astype(float)
    a = cm.sum(axis=1)       # |C_i|
    b = cm.sum(axis=0)       # |K_j|
    n = cm.sum()
    # precision: for each (i,j), contribution n_ij / |K_j|
    # recall   : for each (i,j), contribution n_ij / |C_i|
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_num = (cm * (cm / b)).sum()
        rec_num  = (cm * (cm / a[:, None])).sum()
    prec = prec_num / n
    rec  = rec_num  / n
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)

def hungarian_matched_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    r_ind, c_ind = linear_sum_assignment(-cm)  # maximize trace
    return cm[r_ind, c_ind].sum() / cm.sum()

def variation_of_information(y_true, y_pred, *, base=np.e):
    cm = contingency_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    pi = cm.sum(axis=1) / n  # P(C)
    pj = cm.sum(axis=0) / n  # P(K)
    pij = cm / n             # P(C,K)

    def H(p):
        p = p[p > 0]
        return float(-np.sum(p * (np.log(p) / np.log(base))))
    H_C  = H(pi)
    H_K  = H(pj)
    H_CK = H(pij.flatten())
    # VI = H(C|K) + H(K|C) = 2*H(C,K) - H(C) - H(K)
    return (H_CK - H_K) + (H_CK - H_C)  # smaller is better

def _cluster_labels(X, method="kmeans", n_clusters=None, random_state=42, **kw):
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kw).fit_predict(X)
    if method == "spectral":
        return SpectralClustering(n_clusters=n_clusters, affinity=kw.pop("affinity", "nearest_neighbors"),
                                  random_state=random_state, **kw).fit_predict(X)

# ----- main scoring -----
def get_scores(embedding, labels, method="spectral"):
    # 1) Convert labels to an array and filter out "Multilocalizing"
    labels = np.array(labels, dtype=object)
    mask = (labels != "Multilocalizing") & (labels != "NA")

    X = embedding[mask]          # (N_filtered, D)
    y = labels[mask]             # (N_filtered,)

    # Basic sanity checks
    unique_y = np.unique(y)
    if X.shape[0] < 2 or unique_y.size < 2:
        raise ValueError(
            f"Not enough samples/classes after filtering. "
            f"n_samples={X.shape[0]}, n_classes={unique_y.size}"
        )

    # 2) Standardize (zero mean, unit variance) to help KMeans
    X_std = StandardScaler().fit_transform(X)

    # 3) Set number of clusters = number of unique ground-truth classes
    n_clusters = unique_y.size
    y_pred = _cluster_labels(X_std, method=method, n_clusters=n_clusters)

    # 4) External metrics (true labels vs clusters)
    ari = adjusted_rand_score(y, y_pred)
    v   = v_measure_score(y, y_pred)
    fmi = fowlkes_mallows_score(y, y_pred)

    nmi = normalized_mutual_info_score(y, y_pred, average_method="arithmetic")
    ami = adjusted_mutual_info_score(y, y_pred)
    h   = homogeneity_score(y, y_pred)
    c   = completeness_score(y, y_pred)
    ri  = rand_score(y, y_pred)  # unadjusted RI

    purity = purity_score(y, y_pred)
    b3_p, b3_r, b3_f1 = b3_scores(y, y_pred)
    hung_acc = hungarian_matched_accuracy(y, y_pred)
    vi = variation_of_information(y, y_pred, base=np.e)

    print(f"[{method}] n_samples={X.shape[0]}, n_clusters={n_clusters}")
    print(f"Adjusted Rand Index     : {ari:.4f}")
    print(f"V-measure               : {v:.4f}")
    print(f"Fowlkes-Mallows Index   : {fmi:.4f}")
    print(f"Normalized MI (NMI)     : {nmi:.4f}")
    print(f"Adjusted MI (AMI)       : {ami:.4f}")
    print(f"Homogeneity             : {h:.4f}")
    print(f"Completeness            : {c:.4f}")
    print(f"Rand Index (unadjusted) : {ri:.4f}")
    print(f"Purity                  : {purity:.4f}")
    print(f"B3 Precision/Recall/F1  : {b3_p:.4f} / {b3_r:.4f} / {b3_f1:.4f}")
    print(f"Hungarian Accuracy      : {hung_acc:.4f}")
    print(f"Variation of Information: {vi:.4f}  (lower is better)")

    return {
        "ARI": ari, "V": v, "FMI": fmi,
        "NMI": nmi, "AMI": ami,
        "Homogeneity": h, "Completeness": c, "RandIndex": ri,
        "Purity": purity, "B3_P": b3_p, "B3_R": b3_r, "B3_F1": b3_f1,
        "Hungarian_Acc": hung_acc, "VI": vi,
        "n_samples": int(X.shape[0]), "n_clusters": int(n_clusters),
    }


@cli(ViTConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # subfolder_name = '2d_proj_256_crop_dataset'
    # subfolder_name = '2d_proj_256_crop_dataset_virtual_staining'
    # subfolder_name = '2d_proj_256_crop_dataset_virtual_staining_same_nucl'
    subfolder_name = '2d_proj_256_crop_dataset_virtual_staining_same_nucl_20k'

    args.max_protein_sequence_len = 2048
    valset = OpenCellCropImageEmbedDataset(args, subfolder_name=subfolder_name)

    print(f"Validation set size: {len(valset)}")

    config = ViTConfig(**vars(args))
    config.num_classes = len(valset.gene_names)

    val_loader = DataLoader(
        valset, batch_size=256, shuffle=False,
        num_workers=32, collate_fn=valset.collate, 
    )

    model = ViTModel(config=config)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / subfolder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_labels = []

    for data in tqdm(val_loader, desc="Processing"):
        data = data["batched_data"]

        locations = data['locations']
        for j, loc in enumerate(locations):
            if len(loc) > 1:
                locations[j] = 'Multilocalizing'
            else:
                locations[j] = loc[0]

        protein_img = data['protein_img'].to(device)  # [B, 1, H, W]
        nucleus_img = data['nucleus_img'].to(device)  # [B, 1, H, W]

        input_img = torch.cat([nucleus_img, protein_img], dim=1)  # [B, 2, H, W]
        embedding = model.embed(input_img)

        all_embeddings.append(embedding.detach().cpu().flatten(start_dim=1, end_dim=-1).numpy())
        all_labels.extend(locations)

    embedding_matrix = np.concatenate(all_embeddings, axis=0)

    assert embedding_matrix.shape[0] == len(all_labels)

    # save embeddings and labels
    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=embedding_matrix,
        labels=np.array(all_labels),
    )

    get_scores(embedding_matrix, all_labels, method="kmeans")
    get_scores(embedding_matrix, all_labels, method="spectral")

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embedding_matrix)

    # ---- put 'Multilocalizing' first and 'NA' last ----
    labels_set = set(all_labels)

    middle_labels = sorted(
        [lbl for lbl in labels_set if lbl not in ("Multilocalizing", "NA")]
    )

    unique_labels = []
    if "Multilocalizing" in labels_set:
        unique_labels.append("Multilocalizing")
    unique_labels.extend(middle_labels)
    if "NA" in labels_set:
        unique_labels.append("NA")
    # -----------------------------------------------

    default_color = "#999999"

    labels_arr = np.array(all_labels)
    mask_multi = (labels_arr == "Multilocalizing")
    mask_other = ~mask_multi

    coords_multi = embedding_2d[mask_multi]
    coords_other = embedding_2d[mask_other]

    color_multi = mcolors.to_rgba(location_colors.get("Multilocalizing", default_color))
    colors_other = [
        mcolors.to_rgba(location_colors.get(lbl, default_color))
        for lbl in labels_arr[mask_other]
    ]

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.22], wspace=0.02)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis('off')

    if coords_multi.size > 0:
        ax.scatter(
            coords_multi[:, 0], coords_multi[:, 1],
            c=[color_multi], s=2, marker='o',
            edgecolors='white', linewidths=0.2,
            alpha=1.0, zorder=0,
        )

    ax.scatter(
        coords_other[:, 0], coords_other[:, 1],
        c=colors_other, s=4, marker='o',
        edgecolors='white', linewidths=0.2,
        alpha=1.0, zorder=2,
    )

    ax.set_title("UMAP Projection of Protein Embeddings", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.15, linewidth=0.5)

    legend = ax_leg.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w',
                    label=label,
                    markerfacecolor=location_colors.get(label, default_color),
                    markersize=6)
            for label in unique_labels
        ],
        title="Locations",
        loc='center left',
        bbox_to_anchor=(0.0, 0.5),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
        fontsize='small',
        title_fontsize='medium',
    )

    fig.tight_layout()
    output_path = output_dir / "umap_embedding.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"UMAP plot saved to: {output_path}")

if __name__ == "__main__":
    main()
