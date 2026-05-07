# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.opencell_crop_data.dataset import OpenCellCropDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger

from cell_fm.metrics.iou import compute_iou, binarize_img
from torchvision.utils import save_image
from esm.utils import decoding


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
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    config = CELLFMConfig(**vars(args))

    valset = OpenCellCropDataset(args, split_key=args.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    vocab = valset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    ious = []

    for i, data in enumerate(valset):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > config.max_protein_sequence_len + 2:
            continue

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)
        protein_img = data['protein_img'].unsqueeze(0).to(device)
        nucleus_img = data['nucleus_img'].unsqueeze(0).to(device)

        cell_img = nucleus_img
        
        logger.info(data['gene_name'])

        save_file = output_dir / ('{:04d}_'.format(i+1) + data['gene_name'])
        save_file.mkdir(parents=True, exist_ok=True)

        real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
        logger.info(real_protein_seq)

        real_img = torch.cat([torch.full_like(protein_img, -1), protein_img, nucleus_img], dim=1)
        save_image(real_img, save_file / 'real_img.png', normalize=True, value_range=(-1, 1))
        save_image(nucleus_img, save_file / 'real_nucleus_img.png', normalize=True, value_range=(-1, 1))
        save_image(protein_img, save_file / 'real_protein_img.png', normalize=True, value_range=(-1, 1))

        sample = model.sequence_to_image(
            protein_seq, 
            cell_img, 
            num_steps=config.num_steps, 
        )

        pred_img = torch.cat([torch.full_like(protein_img, -1), sample, nucleus_img], dim=1)

        save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
        save_image(sample, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))

        iou = compute_iou(binarize_img(sample, threshold_mode="quantile", quantile_q=0.5), binarize_img(protein_img)).item()
        ious.append(iou)

        generated_threshold_img = binarize_img(sample, threshold_mode="quantile", quantile_q=0.5)
        real_threshold_img = binarize_img(protein_img)

        real_threshold_img = 2 * (real_threshold_img.float() * 0.5) - 1
        real_threshold_img = torch.cat([torch.full_like(protein_img, -1), real_threshold_img, nucleus_img], dim=1)
        save_image(real_threshold_img, save_file / 'real_threshold_img.png', normalize=True, value_range=(-1, 1))

        generated_threshold_img = 2 * (generated_threshold_img.float() * 0.5) - 1
        generated_threshold_img = torch.cat([torch.full_like(protein_img, -1), generated_threshold_img, nucleus_img], dim=1)
        save_image(generated_threshold_img, save_file / 'generated_threshold_img.png', normalize=True, value_range=(-1, 1))

        save_colored_image((nucleus_img.squeeze() + 1) / 2, save_file / 'real_nucleus_img_blue.png', 'blue')
        save_colored_image((protein_img.squeeze() + 1) / 2, save_file / 'real_protein_img_green.png', 'green')
        save_colored_image((sample.squeeze() + 1) / 2, save_file / 'generated_protein_img_green.png', 'green')

        logger.info(f"iou: {iou}")

    logger.info(args.loadcheck_path)
    logger.info(f"Avg iou: {sum(ious)/len(ious)}")

if __name__ == "__main__":
    main()
