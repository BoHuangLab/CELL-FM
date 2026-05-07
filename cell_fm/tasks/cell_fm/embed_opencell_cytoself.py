# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_cytoself_data.dataset import OpenCellCytoselfDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from esm.utils import encoding, decoding

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm

import matplotlib.cm as cm
import matplotlib.colors as mcolors

location_colors = {
    "Multilocalizing": "#999999",
    "big_aggregates": "#FF0000",
    "cell_contact": "#008000",
    "centrosome": "#D000FF",
    "chromatin": "#04abc5",
    "cytoplasmic": "#96fe00",
    "cytoskeleton": "#fea42e",
    "er": "#fe8dc7",
    "focal_adhesions": "#78525d",
    "golgi": "#00FFFF",
    "membrane": "#D8B0FF",
    "mitochondria": "#B0C4A3",
    "negative": "#996515",
    "nuclear_membrane": "#2F4F4F",
    "nuclear_punctae": "#FF00AA",
    "nucleolus_fc_dfc": "#FFFACD",
    "nucleolus_gc": "#CD7054",
    "nucleoplasm": "#ADD8E6",
    "vesicles": "#00C000", 
}

@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    valset = OpenCellCytoselfDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = valset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    num_aa = 400

    all_embeddings = []
    all_labels = []

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):
        # print(data['gene_name'])
        # print(decoding.decode_sequence(data['protein_seq'].squeeze(0), vocab))

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)
        locations = data['locations']

        if len(locations) > 1:
            continue

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        protein_img = data['protein_img'].unsqueeze(0).to(device)
        nucleus_img = data['nucleus_img'].unsqueeze(0).to(device)
        protein_seq = protein_seq.repeat(protein_img.shape[0], 1)

        cell_img = nucleus_img

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        img_embedding, _ = model.embed(
            protein_seq, 
            protein_img, 
            cell_img, 
        )

        protein_seq_masked = "<mask>" * num_aa
        protein_seq_masked = encoding.tokenize_sequence(protein_seq_masked, vocab, True)
        protein_seq_masked = protein_seq_masked.unsqueeze(0).to(device)

        _, seq_embedding = model.embed(
            protein_seq_masked, 
            protein_img, 
            cell_img, 
        )
        embedding = seq_embedding + img_embedding * 0.15
        # embedding = seq_embedding
        # embedding = torch.cat([img_embedding, seq_embedding], dim=-1)

        all_embeddings.append(embedding.detach().cpu().flatten(start_dim=1, end_dim=-1).numpy())
        all_labels.extend([locations[0]] * protein_img.shape[0])

    embedding_matrix = np.concatenate(all_embeddings, axis=0)

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embedding_matrix)

    unique_labels = sorted(set(all_labels))

    default_color = "#999999"
    colors = [mcolors.to_rgba(location_colors.get(label, default_color)) for label in all_labels]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=colors, 
        s=2, 
        alpha=1.0, 
    )

    handles = [
        plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            label=label, 
            markerfacecolor=location_colors.get(label, default_color), 
            markeredgecolor='k', 
            markersize=6, 
        )
        for label in unique_labels
    ]

    plt.legend(
        handles=handles, 
        title="Locations", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        fontsize='small', 
        title_fontsize='medium', 
        ncol=1, 
    )

    plt.title("UMAP Projection of Protein Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path = output_dir / "umap_embedding.png"
    plt.savefig(output_path, dpi=300)
    print(f"UMAP plot saved to: {output_path}")

if __name__ == "__main__":
    main()
