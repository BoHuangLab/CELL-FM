# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import matplotlib.pyplot as plt
import numpy as np
import umap

import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

location_colors = {
    'Multilocalizing':           '#999999',  # gray for multilocalizing
    'Actin filaments':           '#fea42e',  # cytoskeleton
    'Aggresome':                 "#FF0000",  # protein aggregates
    'Cell Junctions':            '#018600',  # cell-cell contact
    'Centriolar satellite':      '#b400fe',  # centrosome-related structure
    'Centrosome':                "#fe00b2",  # centrosome
    'Cleavage furrow':           "#f9e894",  # cytoskeleton
    'Cytokinetic bridge':        "#c5aa28",  # cytoskeleton
    'Cytoplasmic bodies':        '#96fe00',  # cytoplasmic structures
    'Cytosol':                   "#3bfe00",  # cytoplasm
    'Endoplasmic reticulum':     '#fe8dc7',  # endoplasmic reticulum
    'Endosomes':                 '#00c746',  # vesicles
    'Focal adhesion sites':      '#78525d',  # focal adhesions
    'Golgi apparatus':           '#00fcce',  # Golgi
    'Intermediate filaments':    "#b96900",  # cytoskeleton
    'Kinetochore':               '#04abc5',  # chromatin-associated
    'Lipid droplets':            "#11ca52",  # vesicles (lipid storage)
    'Lysosomes':                 "#00c77e",  # vesicles (lysosomal compartment)
    'Micronucleus':              '#356961',  # nuclear membrane (micronuclei often retain nuclear envelope)
    'Microtubule ends':          "#a66e25",  # cytoskeleton
    'Microtubules':              "#ffd82b",  # cytoskeleton
    'Midbody':                   "#f87911",  # cytoskeleton
    'Midbody ring':              "#dc6313",  # cytoskeleton
    'Mitochondria':              '#92ab82',  # mitochondria
    'Mitotic chromosome':        "#31e4ff",  # chromatin
    'Mitotic spindle':           "#e13e11",  # cytoskeleton
    'Nuclear bodies':            '#d2008b',  # nuclear puncta
    'Nuclear membrane':          "#214741",  # nuclear membrane
    'Nuclear speckles':          "#ff32bb",  # nuclear puncta
    'Nucleoli':                  '#c76d66',  # nucleolus (granular component)
    'Nucleoli fibrillar center': '#fcf490',  # nucleolus (fibrillar center/DFC)
    'Nucleoli rim':              "#93423c",  # nucleolus (rim closer to granular component)
    'Nucleoplasm':               '#9de1fe',  # nucleoplasm
    'Peroxisomes':               "#14c47d",  # vesicles (small peroxisomes)
    'Plasma membrane':           '#aea4fe',  # cell membrane
    'Rods & Rings':              "#00fe00",  # cytoplasmic inclusions
    'Vesicles':                  "#4eff8c",  # vesicles
    'NA':                        '#999999',  # negative / not annotated
}


def main() -> None:

    # saved_embeddings = np.load('output/hpa/embedding/cellfm_test/embeddings.npz')
    # embedding_matrix = saved_embeddings['embeddings']
    # all_labels = saved_embeddings['labels'].tolist()
    # print("embedding shape:", embedding_matrix.shape)

    # reducer = umap.UMAP(n_components=2, random_state=42)
    # embedding_2d = reducer.fit_transform(embedding_matrix)

    # np.savez_compressed(
    #     "output/hpa/embedding/cellfm_test/embedding_2d.npz",
    #     embeddings=embedding_2d,
    #     labels=np.array(all_labels),
    # )

    saved_embeddings = np.load('output/hpa/embedding/cellfm_test/embedding_2d.npz')
    embedding_2d = saved_embeddings['embeddings']
    all_labels = saved_embeddings['labels'].tolist()

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

    # 关键：figure 透明
    fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="none")   # <<<

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.02)
    ax     = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1]); ax_leg.axis('off')

    # 关键：axes 透明
    ax.set_facecolor("none")                                      # <<<
    ax_leg.set_facecolor("none")                                  # <<<

    ax.set_box_aspect(1)                 # <-- square plotting area

    # —— 关键：让右侧轴与左侧轴同高 ——
    divider = make_axes_locatable(ax)
    ax_leg = divider.append_axes("right", size="5%", pad=0.02)  # size=右侧宽度比例, pad=间距
    ax_leg.axis("off")
    ax_leg.set_facecolor("none") 

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
                        markersize=10)
                for label in unique_labels]

    leg = ax_leg.legend(                                            # <<<
        handles=handles,
        # title="Locations",
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,  # 你原本就是 False
        ncol=2,
        fontsize=15,
        # title_fontsize=30,
    )
    # 如果你未来把 frameon=True，也确保透明：
    leg.get_frame().set_facecolor("none")                            # <<<
    leg.get_frame().set_edgecolor("none")                            # <<<

    fig.tight_layout()
    output_path = "./umap_embedding.png"
    fig.savefig(
        output_path, 
        dpi=1200, 
        bbox_inches='tight', 
        pad_inches=0, 
        transparent=True, 
    )
    print(f"UMAP plot saved to: {output_path}")

if __name__ == "__main__":
    main()
