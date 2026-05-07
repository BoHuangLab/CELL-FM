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

    config.output_dir = 'output/condenseq/img2seq/el_2000'

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_aas = [66]

    expression_level = 100
    num_gen = 20
    num_val = 1
    gen_pos = "end"

    gen_signals = []

    # 4401 (with condensate)
    # seq = 'GSRRGGRGSFRGARGGFGLGSPNNDLDPDEGFNFGMNNFGRRGGRGSFRGARGGFGLGSPNNDLDP'
    # 2315 (no condensate)
    # seq = 'EPLLQTPPAHQQQQQQQQPLLCSSPTSMQSSGTSVTGSSIASGTGAATSSSGVGLGPTTGLDSGAN'

    # append 20
    # seq = 'LCSSPTSMQSSGTSVTGSSIASGTGAATSSSGVGLGPTTGLDSGAN'
    # append 30
    # seq = 'EPLLQTPPAHQQQQQQQQPLLCSSPTSMQSSGTSVT'
    # append 60
    # seq = 'EPLLQT'

    seq = ''

    # Image without condensate
    # chosen_data_root = 'output/condenseq/seq2img_single/test/img_wo_condensate.tif'
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

    expression_level = torch.tensor([expression_level], dtype=torch.float32, device=device).unsqueeze(0)

    for num_aa in num_aas:
        if gen_pos == 'start':
            masked_seq = "M" + "<mask>" * num_aa + seq[1:]
        elif gen_pos == 'end':
            masked_seq = seq + "<mask>" * num_aa

        print(masked_seq)

        protein_seq = encoding.tokenize_sequence(masked_seq, vocab, True)
        protein_seq = protein_seq.unsqueeze(0).to(device)
        protein_seq_mask = (protein_seq == vocab.mask_token_id)

        for i in range(num_gen):
            sample = model.image_to_sequence(
                protein_seq, protein_seq_mask, chosen_protein_img, 
                chosen_cell_img, expression_level, order='random', temperature=10, 
                progress=False, 
            )
            gen_seq = decoding.decode_sequence(sample.squeeze(), vocab)

            if gen_pos == 'start':
                gen_sig = gen_seq[1:num_aa+1]
            elif gen_pos == 'end':
                gen_sig = gen_seq[-num_aa:]

            if gen_sig not in gen_signals:
                gen_signals.append(gen_sig)

                seq_save = [gen_seq, gen_sig]
                save_dir = output_dir / '{:04d}_{:02d}'.format(num_aa, i+1)
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_dir / 'seq.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for item in seq_save:
                        writer.writerow([item])

                for j in range(num_val):
                    protein_image_gen = model.sequence_to_image(
                        sample, chosen_cell_img, expression_level, config.num_steps, 
                    )

                    save_image(
                        protein_image_gen, 
                        save_dir / 'protein_image_gen_{:02d}.png'.format(j+1), 
                        normalize=True, 
                        value_range=(-1, 1)
                    )

    # save gen_signals
    with open(output_dir / 'gen_signals.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for signal in gen_signals:
            writer.writerow([signal])

if __name__ == "__main__":
    main()