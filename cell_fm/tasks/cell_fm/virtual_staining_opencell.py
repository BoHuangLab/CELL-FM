# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image
import numpy as np
import tifffile as tiff
from tqdm import tqdm

def save_tif(image, output_path):
    tensor_np = image.cpu().numpy()
    tensor_np = tensor_np.clip(0, 1)
    tensor_np = np.round(tensor_np * 65535)

    tensor_np = tensor_np.astype(np.uint16)    
    tiff.imwrite(output_path, tensor_np, imagej=True)

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


@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    valset = OpenCellCropAllImageDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    # chosen images
    gene_name = 'ATG7'
    index = 1

    valset.meta_data = valset.meta_data[valset.meta_data['gene_name'] == gene_name]
    chosen_data = valset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_protein_img = chosen_data['protein_imgs'][index].unsqueeze(0).to(device)

    chosen_cell_img = chosen_nucleus_img

    # number of samples
    num_samples = 100

    # batch size
    batch_size = 10

    # chosen gene
    gene_name = 'MRTO4'
    valset = OpenCellCropAllImageDataset(config, split_key=config.split_key)
    valset.meta_data = valset.meta_data[valset.meta_data['gene_name'] == gene_name]

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        # seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
        # print(seq)
        # chosen_cell_img = data['nucleus_imgs'][index].unsqueeze(0).to(device)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            cell_imgs_batch = chosen_cell_img.repeat(batch_end - batch_start, 1, 1, 1)
            protein_seq_batch = protein_seq.repeat(cell_imgs_batch.shape[0], 1)

            # Generate images
            sample = model.sequence_to_image(
                protein_seq_batch, 
                cell_imgs_batch, 
                num_steps=args.num_steps, 
            )

            cat_img = torch.cat([torch.full_like(cell_imgs_batch, -1), sample, cell_imgs_batch], dim=1)

            save_file = output_dir / ('{:04d}_'.format(i + 1) + data['gene_name'])
            save_file.mkdir(parents=True, exist_ok=True)

            # Save sample images
            for j in range(batch_start, batch_end):
                # Save the generated image
                save_image(cat_img[j - batch_start], save_file / f'image_{j}.png', normalize=True, value_range=(-1, 1))
                save_image(sample[j - batch_start], save_file / f'sample_{j}.png', normalize=True, value_range=(-1, 1))

                # # Save the generated image
                # cat_img_j = cat_img[j - batch_start]
                # cat_img_j = cat_img_j[1:]  # remove the black image
                # cat_img_j = torch.cat([cat_img_j[1:2, :, :], cat_img_j[0:1, :, :]], dim=0)  # nucleus, protein
                # # convert to [0, 1]
                # cat_img_j = (cat_img_j + 1) / 2
                # cat_img_j = cat_img_j.clamp(0, 1)

                # save_tif(cat_img_j, save_file / ('{:04d}.tif'.format(j + 1)))

                # # Save the chosen images
                # save_colored_image(chosen_nucleus_img, save_file / 'chosen_nucleus_img.png', 'blue')
                # save_colored_image(chosen_protein_img, save_file / 'chosen_protein_img.png', 'red')


if __name__ == "__main__":
    main()