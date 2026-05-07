# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.condenseq_data.dataset import CondenSeqAllImageDataset
from cell_fm.models.cell_fm.cell_fm_cs_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_cs_model import CELLFMCSModel
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger

from esm.utils import encoding
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm


def save_tif(image, output_path):
    image = image * 65535

    image = image.astype(np.uint16)
    tiff.imwrite(output_path, image, imagej=True)


@dataclass
class ExtendedConfig:
    start_idx: int = 0
    end_idx: int = 1000

@cli(CELLFMConfig, ExtendedConfig)
def main(args) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)

    start_idx = args.start_idx
    end_idx = args.end_idx

    meta_data = pd.read_csv('/home/dihan.zheng/workspace/dihan.zheng/dataset/human_protein/human_idr_regions_plddt_only_chopped_unique_seq.csv')

    # chop the meta_data according to start_index and end_index (not inclusive)
    meta_data = meta_data[start_idx:end_idx]

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(args, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)
    vocab = valset.vocab

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_mask_img = chosen_data['mask_imgs'][index].unsqueeze(0).to(device).round().long()

    chosen_cell_img = chosen_nucleus_img

    # batch_size = 256
    # num_protein_intensity_levels = 256
    # max_protein_intensity_level = 800

    batch_size = 512
    num_protein_intensity_levels = 512
    max_protein_intensity_level = 4096

    for _, row in tqdm(meta_data.iterrows(), total=meta_data.shape[0], desc=f"Processing from {start_idx} to {end_idx} (not inclusive)"):

        from cell_fm.pipeline.accelerator.trainer import seed_everything    
        seed_everything(args.seed)

        protein_index = row['index']

        save_file = output_dir / f'{protein_index:06d}'
        save_file.mkdir(parents=True, exist_ok=True)

        protein_seq_txt = row['sequence']

        protein_seq = encoding.tokenize_sequence(protein_seq_txt, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)

        generated_imgs = []
        max_minus_medians_all = []

        protein_intensity_levels_linspace = torch.linspace(0, max_protein_intensity_level, steps=num_protein_intensity_levels).to(device)

        for batch_start in range(0, num_protein_intensity_levels, batch_size):
            batch_end = min(batch_start + batch_size, num_protein_intensity_levels)

            protein_intensity_level_batch = protein_intensity_levels_linspace[batch_start:batch_end].unsqueeze(-1)

            protein_seq_batch = protein_seq.repeat(protein_intensity_level_batch.shape[0], 1)
            chosen_cell_img_batch = chosen_cell_img.repeat(protein_intensity_level_batch.shape[0], 1, 1, 1)

            sample = model.sequence_to_image(
                protein_seq_batch, 
                chosen_cell_img_batch, 
                protein_intensity_level_batch, 
                num_steps=config.num_steps, 
            )
            generated_imgs.append(sample.squeeze(1))

            sample = ((sample + 1) / 2).clamp(0, 1).squeeze(1)
            mask = chosen_mask_img.squeeze().round().long()

            masked_samples = sample[:, mask == 1] # shape (B, N), where N = number of masked voxels

            max_vals = masked_samples.max(dim=1).values
            median_vals = masked_samples.median(dim=1).values

            max_minus_medians = (max_vals - median_vals).tolist()
            max_minus_medians_all.append(max_minus_medians)

        # Concatenate all images
        generated_imgs = torch.cat(generated_imgs, dim=0)

        # rescale images to [0, 1]
        generated_imgs = (generated_imgs + 1) / 2
        generated_imgs = generated_imgs.clamp(0, 1)

        # save as tif file
        save_tif(generated_imgs.detach().cpu().numpy(), save_file / 'imgs.tif')
        # save protein_intensity_levels_linspace as npy
        np.save(save_file / 'protein_intensity_levels.npy', protein_intensity_levels_linspace.detach().cpu().numpy())

        # Step 0: prepare figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Step 1: plot generated data scatter
        ax.scatter(protein_intensity_levels_linspace.detach().cpu().tolist(), max_minus_medians_all, color='orange', alpha=0.6, s=15, label='Generated Data')

        # Step 2: formatting
        ax.set_xlim(-10, max_protein_intensity_level + 10)
        ax.set_xlabel('Protein Intensity Level')
        ax.set_ylabel('Max - Median')
        ax.set_title('Max - Median vs Protein Intensity Level')
        ax.grid(axis='y')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_file / 'Max_minus_median_normalized_img.png')
        plt.close()

    logger.info("Done!")

if __name__ == "__main__":
    main()