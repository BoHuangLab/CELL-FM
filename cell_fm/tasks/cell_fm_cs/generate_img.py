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

from tqdm import tqdm


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
    # image = np.transpose(image, (1, 0, 2, 3))
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

    valset = CondenSeqAllImageDataset(args, split_key=args.split_key)
    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    vocab = valset.vocab

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    batch_size = 16

    for i, data in tqdm(enumerate(valset)):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > config.max_protein_sequence_len + 2:
            continue

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        protein_imgs = torch.stack(data['protein_imgs']).to(device)
        nucleus_imgs = torch.stack(data['nucleus_imgs']).to(device)
        protein_intensity_levels = torch.tensor(data['protein_intensity_levels']).float()
        protein_intensity_levels = protein_intensity_levels.unsqueeze(-1).to(device)

        cell_imgs = nucleus_imgs

        logger.info(data['index'])

        save_file = output_dir / ('{:04d}_'.format(i+1) + str(data['index']))
        save_file.mkdir(parents=True, exist_ok=True)

        real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
        logger.info(real_protein_seq)

        protein_seq = protein_seq.repeat(cell_imgs.shape[0], 1)  # (B, L)

        real_imgs = []
        generated_imgs = []

        for batch_start in range(0, cell_imgs.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, cell_imgs.shape[0])
            batch_cell_imgs = cell_imgs[batch_start:batch_end].to(device)
            batch_protein_seq = protein_seq[batch_start:batch_end].to(device)
            batch_prot_imgs = protein_imgs[batch_start:batch_end].to(device)
            batch_protein_intensity_levels = protein_intensity_levels[batch_start:batch_end].to(device)

            # Generate images
            sample = model.sequence_to_image(
                batch_protein_seq, 
                batch_cell_imgs, 
                batch_protein_intensity_levels, 
                num_steps=args.num_steps, 
            )

            # generated_save_dir = save_file / 'generated'
            # real_save_dir = save_file / 'real'

            # generated_save_dir.mkdir(parents=True, exist_ok=True)
            # real_save_dir.mkdir(parents=True, exist_ok=True)

            # # save images
            # for j in range(batch_start, batch_end):
            #     # save samples
            #     save_image(sample[j - batch_start], generated_save_dir / f'img_{j}.png', normalize=True, value_range=(-1, 1))
            #     # save protein images
            #     save_image(batch_prot_imgs[j - batch_start], real_save_dir / f'img_{j}.png', normalize=True, value_range=(-1, 1))
            
            real_imgs.append(batch_prot_imgs)
            generated_imgs.append(sample)

        # Concatenate all images
        real_imgs = torch.cat(real_imgs, dim=0)
        generated_imgs = torch.cat(generated_imgs, dim=0)

        # rescale images to [0, 1]
        real_imgs = (real_imgs + 1) / 2
        real_imgs = real_imgs.clamp(0, 1)

        generated_imgs = (generated_imgs + 1) / 2
        generated_imgs = generated_imgs.clamp(0, 1)

        # save as tif file
        save_tif(real_imgs.detach().cpu().numpy(), save_file / 'real_imgs.tif')
        save_tif(generated_imgs.detach().cpu().numpy(), save_file / 'generated_imgs.tif')

        # rescale images to [0, 1]
        # imgs = (imgs + 1) / 2
        # imgs = imgs.clamp(0, 1)

        # # save as tif file
        # save_tif(imgs.detach().cpu().numpy(), save_file / 'imgs.tif')

    logger.info("Done!")

if __name__ == "__main__":
    main()