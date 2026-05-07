# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])


import matplotlib.pyplot as plt
import numpy as np
import umap

import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score, v_measure_score, fowlkes_mallows_score,
    normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, rand_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def main() -> None:
    # saved_embeddings = np.load('output/opencell/embedding/2d_proj_256_crop_dataset/embeddings.npz')
    saved_embeddings = np.load('output/opencell/embedding/2d_proj_256_crop_dataset_virtual_staining_same_nucl/embeddings.npz')

    embedding_matrix = saved_embeddings['embeddings']
    all_labels = saved_embeddings['labels'].tolist()

    # get_scores(embedding_matrix, all_labels, method="kmeans")
    # get_scores(embedding_matrix, all_labels, method="spectral")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
    embedding_2d = reducer.fit_transform(embedding_matrix)

    # rotation 180 degrees for embedding_2d
    embedding_2d = embedding_2d * -1.0

    # flip y axis
    embedding_2d[:, 1] = -embedding_2d[:, 1]

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
    # --------------------------------------------

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
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.02)
    ax     = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1]); ax_leg.axis('off')

    ax.set_box_aspect(1)                 # <-- square plotting area

    # —— 关键：让右侧轴与左侧轴同高 ——
    divider = make_axes_locatable(ax)
    ax_leg = divider.append_axes("right", size="5%", pad=0.02)  # size=右侧宽度比例, pad=间距
    ax_leg.axis("off")

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

    # ----- custom axis style -----
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 两段轴线（相对坐标）
    x0, x1 = 0.06, 0.25   # 底部轴起止
    y0, y1 = 0.06, 0.25   # 左侧轴起止
    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color="#000000", lw=2)
    ax.plot([x0, x0], [y0, y1], transform=ax.transAxes, color="#000000", lw=2)

    # 轴标题放到“内侧”
    # UMAP1：放到底部轴线稍上方、居中
    ax.text((x0 + x1) / 2, y0 + 0.01, "UMAP1",
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, color="#000000")

    # UMAP2：放到左侧轴线稍右侧、居中
    ax.text(x0 + 0.01, (y0 + y1) / 2, "UMAP2",
            transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color="#000000", rotation=90)
    # --------------------------------

    handles=[plt.Line2D([0], [0], marker='o', color='w',
                        label=label,
                        markerfacecolor=location_colors.get(label, default_color),
                        markersize=6)
                for label in unique_labels]

    ax_leg.legend(
        handles=handles,
        # title="Locations",
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        ncol=2,
        fontsize="small",
        # title_fontsize="medium",
    )

    fig.tight_layout()
    output_path = "./umap_embedding.png"
    fig.savefig(output_path, dpi=1200, bbox_inches='tight', pad_inches=0)
    # save to svg
    output_path_svg = "./umap_embedding.svg"
    fig.savefig(output_path_svg, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"UMAP plot saved to: {output_path} and {output_path_svg}")

if __name__ == "__main__":
    main()