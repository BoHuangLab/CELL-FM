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

from torchvision.utils import save_image
from esm.utils import decoding
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


def colorize_image(tensor, color):
    # Create a zero tensor with the same size as the input tensor but with three channels
    colored_image = torch.zeros((3, tensor.size(0), tensor.size(1)), dtype=tensor.dtype, device=tensor.device)

    if color == 'blue':
        colored_image[2] = tensor  # Set blue channel
    elif color == 'red':
        colored_image[0] = tensor  # Set red channel
    elif color == 'green':
        colored_image[1] = tensor  # Set green channel
    elif color == 'yellow':
        colored_image[0] = tensor  # Set red and green channels to get yellow
        colored_image[1] = tensor
    return colored_image

def save_colored_image(tensor, filename, color):
    colored_tensor = colorize_image(tensor, color)
    save_image(colored_tensor, filename, normalize=True, value_range=(0, 1))

def save_tif(image, output_path):
    image = image * 65535

    image = image.astype(np.uint16)
    tiff.imwrite(output_path, image, imagej=True)


@cli(CELLFMConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLFMConfig(**vars(args))
    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(args, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_mask_img = chosen_data['mask_imgs'][index].unsqueeze(0).to(device).round().long()
    chosen_protein_img = chosen_data['protein_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    batch_size = 64
    num_protein_intensity_levels = 256
    # max_protein_intensity_level = 500

    valset = CondenSeqAllImageDataset(args, split_key=args.split_key)
    vocab = valset.vocab

    for i, data in enumerate(valset):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > config.max_protein_sequence_len + 2:
            continue

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        protein_imgs = torch.stack(data['protein_imgs']).to(device)
        nucleus_imgs = torch.stack(data['nucleus_imgs']).to(device)
        mask_imgs = torch.stack(data['mask_imgs']).to(device).round().long()

        protein_intensity_levels = torch.tensor(data['protein_intensity_levels']).float()
        protein_intensity_levels = protein_intensity_levels.unsqueeze(-1).to(device)
        # max_protein_intensity_level = torch.quantile(protein_intensity_levels, 0.9).item()
        max_protein_intensity_level = protein_intensity_levels.max().item()

        logger.info(data['index'])

        save_file = output_dir / ('{:04d}_'.format(i+1) + str(data['index']))
        save_file.mkdir(parents=True, exist_ok=True)

        real_max_minus_medians = []
        real_intensity_levels = []

        protein_imgs = ((protein_imgs + 1) / 2).clamp(0, 1)

        for i in range(protein_imgs.shape[0]):
            masked_img = protein_imgs[i][mask_imgs[i] == 1]

            max_val = masked_img.max().item()
            median_val = masked_img.median().item()
            diff = max_val - median_val

            real_max_minus_medians.append(diff)
            real_intensity_levels.append(protein_intensity_levels[i].item())

        real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
        logger.info(real_protein_seq)

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

        # # save real protein images
        # protein_imgs = protein_imgs.clamp(0, 1)
        # save_tif(protein_imgs.detach().cpu().numpy(), save_file / 'real_protein_imgs.tif')

        # Step 0: prepare figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Step 1: plot generated data scatter
        ax.scatter(protein_intensity_levels_linspace.detach().cpu().tolist(), max_minus_medians_all, color='orange', alpha=0.6, s=15, label='Generated Data')

        # Step 2: plot scatter on same axis
        ax.scatter(real_intensity_levels, real_max_minus_medians, color='darkblue', alpha=0.6, s=15, label='Real Data')

        # Step 3: formatting
        ax.set_xlim(-10, max_protein_intensity_level + 10)
        # ax.set_ylim(0, 12)
        ax.set_xlabel('Protein Intensity Level')
        ax.set_ylabel('Max - Median')
        ax.set_title('Max - Median vs Protein Intensity Level')
        ax.grid(axis='y')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_file / 'combined_max_minus_median_normalized_img.png')
        plt.close()

    logger.info("Done!")

if __name__ == "__main__":
    main()