# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.criterions.cell_fm.unidiffuser import UniDiffCriterions
from cell_fm.data.hpa_data.dataset import HPAAllImageDataset
from cell_fm.models.cell_fm.cell_fm_config import CELLFMConfig
from cell_fm.models.cell_fm.cell_fm_model import CELLFMModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image
from esm.utils import encoding, decoding


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
    dataset = HPAAllImageDataset(config, split_key='all')
    vocab = dataset.vocab

    dataset.meta_data = dataset.meta_data[dataset.meta_data['gene_name'] == 'H3C13']
    chosen_data = dataset.__getitem__(0)
    index = 33

    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)
    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)
    chosen_microtubules_img = chosen_data['microtubules_imgs'][index].unsqueeze(0).to(device)
    chosen_ER_img = chosen_data['ER_imgs'][index].unsqueeze(0).to(device)

    num_samples = 128
    batch_size = 32

    with torch.no_grad():
        if config.cell_image == 'nucl':
            chosen_cell_img = chosen_nucleus_img
        elif config.cell_image == 'nucl,er':
            if config.test_cell_image == 'nucl':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_ER_img], dim=1)
        elif config.cell_image == 'nucl,mt':
            if config.test_cell_image == 'nucl':
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_microtubules_img], dim=1)
        elif config.cell_image == 'nucl,er,mt':
            if config.test_cell_image == 'nucl':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            elif config.test_cell_image == 'nucl,er':
                chosen_microtubules_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            elif config.test_cell_image == 'nucl,mt':
                chosen_ER_img = torch.full_like(chosen_nucleus_img, fill_value=-2.0)
            chosen_cell_img = torch.cat([chosen_nucleus_img, chosen_ER_img, chosen_microtubules_img], dim=1)
        else:
            raise ValueError(f"Cell image type: {config.cell_image} is not supported")

    # H3C13
    # gene_name = 'H3C13'
    # protein_seq = 'MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAYLVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA'

    # gene_name = 'H3C13_rm_VKKP'
    # protein_seq = 'MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAYLVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA'

    # gene_name = 'H3C13_VKKP_to_VRRP'
    # protein_seq = 'MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVRRPHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAYLVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA'

    gene_name = 'H3C13_VKKP_to_GGGG'
    protein_seq = 'MARTKQTARKSTGGKAPRKQLATKAARKSAPATGGGGGGHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAYLVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA'

    protein_seq = encoding.tokenize_sequence(protein_seq, vocab, True)
    protein_seq = protein_seq.unsqueeze(0).to(device)
    
    save_file = output_dir / gene_name
    save_file.mkdir(parents=True, exist_ok=True)

    real_protein_seq = decoding.decode_sequence(protein_seq.squeeze(), vocab)
    print(real_protein_seq)
        
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        samples = model.sequence_to_image(
            protein_seq.repeat(current_batch_size, 1), 
            chosen_cell_img.repeat(current_batch_size, 1, 1, 1), 
            num_steps=config.num_steps, 
        )
        for j in range(current_batch_size):
            sample = samples[j]
            pred_img = sample

            save_image(pred_img, save_file / f'generated_img_{batch_start + j}.png', normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()



