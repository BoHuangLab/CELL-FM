# -*- coding: utf-8 -*-
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from esm.utils import encoding

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm
import matplotlib.colors as mcolors


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, output_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(input_dim*2)

    def forward(self, x):
        x = x.mean(dim=1)  # Global average pooling across sequence
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def embed(self, x):
        x = x.mean(dim=1)  # Global average pooling across sequence
        x = self.fc1(x)
        x = self.bn(x)
        return x

location_colors = {
    "multilocalizing": "#999999",
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
    "vesicles": "#00C000"
}


@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    valset = OpenCellCropAllImageDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = valset.vocab

    model.to(device)
    model.eval()

    classification_model = MLP(config.encoder_hidden_size, 1311).to(device)
    classification_model.load_state_dict(torch.load('output/opencell/embedding/checkpoints/epoch200.pth'))
    classification_model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    num_aa = 10
    batch_size = 120

    all_embeddings = []
    all_labels = []

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):
        # protein_seq = data['protein_seq'].unsqueeze(0)

        protein_seq = "<mask>" * num_aa
        protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        locations = data['locations']

        if len(locations) > 1:
            locations = ["multilocalizing"]

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        # protein_img = torch.stack(data['protein_imgs'])[:batch_size]
        # nucleus_img = torch.stack(data['nucleus_imgs'])[:batch_size]

        protein_img = torch.stack(data['protein_imgs'])
        nucleus_img = torch.stack(data['nucleus_imgs'])
        label = data['protein_idx'].to(device)

        for batch_start in range(0, protein_img.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, protein_img.shape[0])
            protein_img_batch = protein_img[batch_start:batch_end].to(device)
            nucleus_img_batch = nucleus_img[batch_start:batch_end].to(device)
            protein_seq_batch = protein_seq.repeat(protein_img_batch.shape[0], 1).to(device)

            cell_img_batch = nucleus_img_batch

            with torch.no_grad():
                img_feat_embed, seq_feat_embed = model.embed(
                    protein_seq_batch, 
                    protein_img_batch, 
                    cell_img_batch, 
                    img_mask_ratio=0, 
                )
                embedding = classification_model.embed(img_feat_embed)
                outputs = classification_model(img_feat_embed)

                print(outputs.argmax(dim=1))
                print("label:", label)

                assert False

            all_embeddings.append(embedding.detach().cpu().flatten(start_dim=1, end_dim=-1).numpy())
            all_labels.extend([locations[0]] * protein_img_batch.shape[0])

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
        alpha=0.5,
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=label,
            markerfacecolor=location_colors.get(label, default_color),
            markeredgecolor='k',
            markersize=6
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
