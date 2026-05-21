# -*- coding: utf-8 -*-
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from esm.utils import encoding, decoding
from cell_fm.pipeline.accelerator.trainer import seed_everything
from dataclasses import dataclass

REMAINING_CSV = "/hpc/projects/group.huang/dihan.zheng/CELL-FM/cell_fm/tasks/analysis/output/remaining_human_proteins.csv"

# Fixed cell context: ATG7, image index 1
ANCHOR_GENE  = "ATG7"
ANCHOR_INDEX = 1
MIN_PROTEIN_SEQ_LEN = 100


def save_tif(image, output_path):
    tensor_np = image.cpu().numpy()
    tensor_np = tensor_np.clip(0, 1)
    tensor_np = np.round(tensor_np * 65535).astype(np.uint16)
    tiff.imwrite(output_path, tensor_np, imagej=True)


@dataclass
class SplitConfig:
    shard_id: int = 0
    num_shards: int = 1

@cli(CELLFMConfig, SplitConfig)
def main(args) -> None:
    shard_id, num_shards = args.shard_id, args.num_shards

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))

    # load dataset only to obtain the anchor nucleus image and vocab
    dataset = OpenCellCropAllImageDataset(config, split_key=config.split_key)
    vocab = dataset.vocab

    anchor_meta = dataset.meta_data[dataset.meta_data["gene_name"] == ANCHOR_GENE]
    dataset.meta_data = anchor_meta
    anchor_data = dataset.__getitem__(0)
    chosen_nucleus_img = anchor_data["nucleus_imgs"][ANCHOR_INDEX].unsqueeze(0).to(device)

    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)
    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 32
    batch_size  = 32

    proteins = pd.read_csv(REMAINING_CSV)
    proteins = proteins.iloc[shard_id::num_shards].reset_index(drop=True)
    print(f"Shard {shard_id}/{num_shards}: {len(proteins)} proteins to process")

    for _, row in tqdm(proteins.iterrows(), total=len(proteins), desc="Virtual staining"):

        # Use the same noise seed for each protein to ensure consistency across different runs
        seed_everything(config.seed)

        gene_name = str(row["gene_name"])
        uniprot_id = str(row["uniprot"])
        sequence  = str(row["sequence"])

        protein_seq = encoding.tokenize_sequence(sequence, vocab, True).unsqueeze(0).to(device)

        if protein_seq.shape[1] > args.max_protein_sequence_len + 2 or protein_seq.shape[1] <= MIN_PROTEIN_SEQ_LEN + 2:
            continue

        save_file = output_dir / uniprot_id

        # skip if already fully generated
        if save_file.exists() and len(list(save_file.glob("*.tif"))) >= num_samples:
            continue

        save_file.mkdir(parents=True, exist_ok=True)

        try:
            for batch_start in range(0, num_samples, batch_size):
                batch_end         = min(batch_start + batch_size, num_samples)
                current_batch     = batch_end - batch_start
                cell_imgs_batch   = chosen_nucleus_img.repeat(current_batch, 1, 1, 1)
                protein_seq_batch = protein_seq.repeat(current_batch, 1)

                samples = model.sequence_to_image(
                    protein_seq_batch,
                    cell_imgs_batch,
                    num_steps=config.num_steps,
                )

                cat_img = torch.cat([cell_imgs_batch, samples], dim=1)

                for j in range(current_batch):
                    img_j = (cat_img[j] + 1) / 2
                    img_j = img_j.clamp(0, 1)
                    save_tif(img_j, save_file / f"{batch_start + j + 1:04d}.tif")
        except OSError as e:
            print(f"[WARN] Skipping {gene_name}_{uniprot_id}: {e}")
            continue


if __name__ == "__main__":
    main()
