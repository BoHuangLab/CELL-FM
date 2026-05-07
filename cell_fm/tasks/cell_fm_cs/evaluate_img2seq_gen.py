# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from cell_fm.models.cell_fm.cell_fm_cs_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_cs_model import CELLFMCSModel
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger

from torchvision.utils import save_image
import tifffile as tiff
import numpy as np

from tqdm import tqdm
from esm.utils import encoding, decoding
import csv



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

    model = CELLFMCSModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    vocab = EsmSequenceTokenizer()

    num_aa = 66

    expression_level_range = [100, 200]
    num_gen = 20

    config.output_dir = f'output/condenseq/img2seq/el_{expression_level_range[0]}_{expression_level_range[1]}'

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_signals = []

    # Image without condensate
    # chosen_data_root = 'output/condenseq/seq2img_single/test/cat_img.tif'
    # Image with condensate
    chosen_data_root = 'output/condenseq/seq2img_single/test/img_with_condensate.tif'

    data = tiff.imread(chosen_data_root).astype(np.float32)
    chosen_cell_img = torch.from_numpy(data[0]).unsqueeze(0).unsqueeze(0).to(device)
    chosen_protein_img = torch.from_numpy(data[1]).unsqueeze(0).unsqueeze(0).to(device)

    # rescale to [0, 1]
    chosen_cell_img = (chosen_cell_img - chosen_cell_img.min()) / (chosen_cell_img.max() - chosen_cell_img.min())
    chosen_protein_img = (chosen_protein_img - chosen_protein_img.min()) / (chosen_protein_img.max() - chosen_protein_img.min())

    # normalize to [-1, 1]
    chosen_cell_img = chosen_cell_img * 2 - 1
    chosen_protein_img = chosen_protein_img * 2 - 1

    for expression_level in tqdm(range(expression_level_range[0], expression_level_range[1])):

        expression_level = torch.tensor([expression_level], dtype=torch.float32, device=device).unsqueeze(0)

        masked_seq = "<mask>" * num_aa

        protein_seq = encoding.tokenize_sequence(masked_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)
        protein_seq_mask = (protein_seq == vocab.mask_token_id)

        expression_level = expression_level.repeat(num_gen, 1)
        protein_seq = protein_seq.repeat(num_gen, 1)
        protein_seq_mask = protein_seq_mask.repeat(num_gen, 1)

        for i in range(num_gen):
            sample = model.image_to_sequence(
                protein_seq, protein_seq_mask, chosen_protein_img, 
                chosen_cell_img, expression_level, order='random', temperature=1.0, 
                progress=False, 
            )
            gen_seq = decoding.decode_sequence(sample.squeeze(), vocab)
            
            gen_sig = gen_seq

            print(gen_sig)

            if gen_sig not in gen_signals:
                gen_signals.append(gen_sig)

    # save gen_signals
    with open(output_dir / 'gen_signals.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for signal in gen_signals:
            writer.writerow([signal])

if __name__ == "__main__":
    main()