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

    num_sample_each_protein_intensity_level = 16
    num_protein_intensity_levels = 20

    valset = CondenSeqAllImageDataset(args, split_key=args.split_key)
    vocab = valset.vocab

    background_value = valset.background_value_gfp

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
        max_protein_intensity_level = torch.quantile(protein_intensity_levels, 0.9).item()
        # max_protein_intensity_level = protein_intensity_levels.max().item()
        # max_protein_intensity_level = max_protein_intensity_level * 0.5

        chosen_cell_img_batch = chosen_cell_img.repeat(num_sample_each_protein_intensity_level, 1, 1, 1).to(device)

        logger.info(data['index'])

        save_file = output_dir / ('{:04d}_'.format(i+1) + str(data['index']))
        save_file.mkdir(parents=True, exist_ok=True)

        real_max_minus_medians = []
        real_intensity_levels = []

        protein_imgs = ((protein_imgs + 1) / 2).clamp(0, 1)

        for i in range(protein_imgs.shape[0]):
            masked_img = protein_imgs[i][mask_imgs[i] == 1]

            # mean_prot_img = masked_img.mean().item()
            # scale = protein_intensity_levels[i].item() / (mean_prot_img + 1e-6)
            # masked_img = masked_img * scale + background_value

            max_val = masked_img.max().item()
            median_val = masked_img.median().item()
            diff = max_val - median_val

            real_max_minus_medians.append(diff)
            real_intensity_levels.append(protein_intensity_levels[i].item())

        real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
        logger.info(real_protein_seq)

        protein_seq_batch = protein_seq.repeat(num_sample_each_protein_intensity_level, 1)  # (B, L)

        generated_imgs = []
        max_minus_medians_all = []

        protein_intensity_levels_linspace = torch.linspace(0, max_protein_intensity_level, steps=num_protein_intensity_levels).to(device)

        for protein_intensity_level in protein_intensity_levels_linspace:
            protein_intensity_level = protein_intensity_level.unsqueeze(0).repeat(num_sample_each_protein_intensity_level, 1).to(device)

            # Generate images
            sample = model.sequence_to_image(
                protein_seq_batch, 
                chosen_cell_img_batch, 
                protein_intensity_level, 
                num_steps=args.num_steps, 
            )
            generated_imgs.append(sample.squeeze(1))

            sample = ((sample + 1) / 2).clamp(0, 1).squeeze(1)
            mask = chosen_mask_img.squeeze().round().long()

            masked_samples = sample[:, mask == 1]              # shape (B, N), where N = number of masked voxels

            # mean_masked_samples = masked_samples.mean(dim=-1, keepdim=True)  # shape (B, 1)
            # scale = protein_intensity_level / (mean_masked_samples + 1e-6)
            # masked_samples = masked_samples * scale + background_value

            max_vals = masked_samples.max(dim=1).values
            median_vals = masked_samples.median(dim=1).values

            max_minus_medians = (max_vals - median_vals).tolist()
            max_minus_medians_all.append(max_minus_medians)

        # Concatenate all images
        generated_imgs = torch.stack(generated_imgs, dim=0)

        # rescale images to [0, 1]
        generated_imgs = (generated_imgs + 1) / 2
        generated_imgs = generated_imgs.clamp(0, 1)

        # save as tif file
        save_tif(generated_imgs.detach().cpu().numpy(), save_file / 'imgs.tif')

        # save real protein images
        protein_imgs = protein_imgs.clamp(0, 1)
        save_tif(protein_imgs.detach().cpu().numpy(), save_file / 'real_protein_imgs.tif')

        # Step 0: prepare figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Step 1: plot the boxplot
        x_labels = []
        y_data = []
        for i, level in enumerate(protein_intensity_levels_linspace.cpu().numpy()):
            values = max_minus_medians_all[i]
            if len(values) == 0:
                continue
            y_data.append(values)
            x_labels.append(round(level))

        # Draw boxplot
        box = ax.boxplot(
            y_data,
            positions=x_labels,
            widths=(max_protein_intensity_level / num_protein_intensity_levels) * 0.2,
            patch_artist=True,
            showfliers=True
        )

        # Set box color
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('black')
            patch.set_alpha(0.3)

        # Step 2: plot scatter on same axis
        ax.scatter(real_intensity_levels, real_max_minus_medians, color='darkblue', alpha=0.6, s=15, label='Real Data')

        # Step 3: formatting
        ax.set_xlim(-50, max_protein_intensity_level + 50)
        # ax.set_ylim(0, 12)
        ax.set_xlabel('Protein Intensity Level')
        ax.set_ylabel('Max - Median')
        ax.set_title('Max - Median vs Protein Intensity Level')
        ax.grid(axis='y')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_file / 'combined_max_minus_median_normalized_img.png')
        plt.close()

        # # Step 1: Prepare data
        # x_labels = []  # real x-axis tick labels
        # y_data = []    # list of lists: each inner list is a group for one box

        # for i, level in enumerate(protein_intensity_levels_linspace.cpu().numpy()):
        #     values = c_sat[i]
        #     if len(values) == 0:
        #         continue  # skip if no valid values
        #     y_data.append(values)
        #     x_labels.append(round(level, 2))  # you can keep more/less precision if needed

        # # Step 2: Create boxplot
        # plt.figure(figsize=(12, 6))

        # # boxplot expects list of lists; positions specify where each box is drawn
        # box = plt.boxplot(
        #     y_data,
        #     positions=x_labels,
        #     widths=(max_protein_intensity_level / num_protein_intensity_levels) * 0.8,
        #     patch_artist=True,
        #     showfliers=True  # optional: show outliers
        # )

        # # Optional: customize appearance
        # for patch in box['boxes']:
        #     patch.set_facecolor('lightblue')

        # plt.xlim(-100, max_protein_intensity_level + 100)
        # plt.ylim(0, 12)
        # plt.xticks(rotation=45)
        # plt.xlabel('Protein Intensity Level')
        # plt.ylabel('Max / Median')
        # plt.title('Boxplot of Max/Median vs Protein Intensity Level')
        # plt.grid(axis='y')
        # plt.tight_layout()
        # plt.savefig(save_file / 'c_sat.png')
        # plt.close()

        # # save c_sat as plot
        # plt.figure(figsize=(10, 5))
        # plt.xlim(-100, max_protein_intensity_level + 100)
        # # plt.scatter(all_intensity_levels, all_max_over_medians)
        # plt.title('Max over Median vs Protein Intensity Level')
        # plt.xlabel('Protein Intensity Level')
        # plt.ylabel('Max over Median')
        # plt.grid()
        # plt.savefig(save_file / 'c_sat.png')
        # plt.close()

    logger.info("Done!")

if __name__ == "__main__":
    main()