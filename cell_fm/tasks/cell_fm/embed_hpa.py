# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.hpa_data.dataset import HPAAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from esm.utils import encoding

import matplotlib.pyplot as plt
import numpy as np
import umap

from tqdm import tqdm
import matplotlib.cm as cm


@cli(CELLFMConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLFMConfig(**vars(args))
    valset = HPAAllImageDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = valset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    num_aa = 10
    batch_size = 32

    all_embeddings = []
    all_labels = []

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):
        # if i > 20:
        #     break

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        protein_seq = "<mask>" * num_aa
        protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        locations = data['locations']

        if len(locations) > 1:
            continue

        if protein_seq.shape[1] > args.max_protein_sequence_len + 2:
            continue

        protein_img = torch.stack(data['protein_imgs'])
        nucleus_img = torch.stack(data['nucleus_imgs'])
        microtubules_img = torch.stack(data['microtubules_imgs'])
        ER_img = torch.stack(data['ER_imgs'])

        for batch_start in range(0, protein_img.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, protein_img.shape[0])
            protein_img_batch = protein_img[batch_start:batch_end].to(device)
            nucleus_img_batch = nucleus_img[batch_start:batch_end].to(device)
            microtubules_img_batch = microtubules_img[batch_start:batch_end].to(device)
            ER_img_batch = ER_img[batch_start:batch_end].to(device)

            cell_img_batch = torch.cat([nucleus_img_batch, ER_img_batch, microtubules_img_batch], dim=1)

            protein_seq_batch = protein_seq.repeat(protein_img_batch.shape[0], 1).to(device)

            img_feat_embed, _ = model.embed(
                protein_seq_batch,
                protein_img_batch,
                cell_img_batch,
                img_mask_ratio=0,
            )

            # Reduce to (B, D) by averaging spatial dimensions
            img_feat_embed = img_feat_embed.mean(dim=1)

            all_embeddings.append(img_feat_embed.detach().cpu().flatten(start_dim=1, end_dim=-1).numpy())
            all_labels.extend([locations[0]] * protein_img_batch.shape[0])

    embedding_matrix = np.concatenate(all_embeddings, axis=0)

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embedding_matrix)

    unique_labels = sorted(set(all_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = [label_to_index[label] for label in all_labels]

    cmap = cm.get_cmap('tab20', len(unique_labels))
    colors = [cmap(i) for i in color_indices]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=4,
        alpha=1, 
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=label,
            markerfacecolor=cmap(i),
            markersize=6
        )
        for i, label in enumerate(unique_labels)
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
