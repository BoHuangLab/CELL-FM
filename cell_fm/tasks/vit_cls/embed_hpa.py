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
import matplotlib.colors as mcolors

from torch.utils.data import DataLoader


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


@cli(ViTConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    trainset = HPAImageOnlyDataset(args, split_key='cellfm_train')
    config = ViTConfig(**vars(args))
    config.num_classes = len(trainset.antibody)

    valset = HPAImageOnlyDataset(config, split_key=config.split_key)

    val_loader = DataLoader(
        valset, batch_size=256, shuffle=False,
        num_workers=32, collate_fn=valset.collate
    )

    model = ViTModel(config=config)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_labels = []

    for data in tqdm(val_loader, desc="Processing valset"):
        data = data["batched_data"]

        locations = data['locations']
        for j, loc in enumerate(locations):
            if len(loc) > 1:
                locations[j] = 'Multilocalizing'
            else:
                locations[j] = loc[0]

        protein_img = data['protein_img'].to(device)  # [B, 1, H, W]
        nucleus_img = data['nucleus_img'].to(device)  # [B, 1, H, W]
        microtubules_img = data['microtubules_img'].to(device)  # [B, 1, H, W]
        ER_img = data['ER_img'].to(device)  # [B, 1, H, W]

        input_img = torch.cat([nucleus_img, ER_img, microtubules_img, protein_img], dim=1)  # [B, 4, H, W]
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

    ax = fig.add_subplot(gs[0, 0])      # 主图轴
    ax_leg = fig.add_subplot(gs[0, 1])  # legend 轴
    ax_leg.axis('off')                   # 隐藏坐标轴/边框

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
        loc='center left',           # 在 legend 轴的垂直中心对齐
        bbox_to_anchor=(0.0, 0.5),   # 从左侧中点开始
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
