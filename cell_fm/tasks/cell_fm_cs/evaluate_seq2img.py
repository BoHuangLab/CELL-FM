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
import tifffile as tiff
import numpy as np

from tqdm import tqdm
from esm.utils import encoding



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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))

    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    # hnRNPA1
    protein_name = 'hnRNPA1'
    protein_seq = 'NGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGG'

    output_dir = Path(config.output_dir)
    output_dir = output_dir / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(args, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    vocab = valset.vocab

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_cell_img = chosen_nucleus_img

    max_protein_intensity_level = 3000
    num_protein_intensity_levels = 20
    num_sample_each_protein_intensity_level = 16

    protein_intensity_levels_linspace = torch.linspace(0, max_protein_intensity_level, steps=num_protein_intensity_levels).to(device)

    print(protein_seq)

    protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
    protein_seq = protein_seq.unsqueeze(0).to(device)

    generated_imgs = []

    for protein_intensity_level in tqdm(protein_intensity_levels_linspace):
        protein_intensity_level = protein_intensity_level.unsqueeze(0).repeat(num_sample_each_protein_intensity_level, 1).to(device)
        protein_seq_batch = protein_seq.repeat(num_sample_each_protein_intensity_level, 1).to(device)
        chosen_cell_img_batch = chosen_cell_img.repeat(num_sample_each_protein_intensity_level, 1, 1, 1).to(device)

        # Generate images
        sample = model.sequence_to_image(
            protein_seq_batch, 
            chosen_cell_img_batch, 
            protein_intensity_level, 
            num_steps=args.num_steps, 
        )
        generated_imgs.append(sample.squeeze(1))

    generated_imgs = torch.stack(generated_imgs, dim=0)

    # rescale images to [0, 1]
    generated_imgs = (generated_imgs + 1) / 2
    generated_imgs = generated_imgs.clamp(0, 1)

    # save as tif file
    save_tif(generated_imgs.detach().cpu().numpy(), output_dir / 'imgs.tif')
    # save nucleus img as tif file
    # chosen_cell_img = (chosen_cell_img + 1) / 2
    # chosen_cell_img = chosen_cell_img.clamp(0, 1)
    # save_tif(chosen_cell_img.detach().cpu().numpy(), output_dir / 'nucleus_imgs.tif')

    logger.info("Done!")

if __name__ == "__main__":
    main()