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
from tqdm import tqdm


@cli(CELLFMConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMConfig(**vars(args))
    valset = HPAAllImageDataset(config, split_key=config.split_key)
    model = CELLFMModel(config=config, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir)
    output_dir = output_dir / config.split_key

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        locations = data['locations']

        if len(locations) > 1:
            continue

        if protein_seq.shape[1] > args.max_protein_sequence_len+2:
            continue

        for j, (protein_img, nucleus_img, microtubules_img, ER_img) in enumerate(zip(data['protein_imgs'], data['nucleus_imgs'], data['microtubules_imgs'], data['ER_imgs'])):
            protein_img = protein_img.unsqueeze(0).to(device)
            nucleus_img = nucleus_img.unsqueeze(0).to(device)
            microtubules_img = microtubules_img.unsqueeze(0).to(device)
            ER_img = ER_img.unsqueeze(0).to(device)

            with torch.no_grad():
                if config.cell_image == 'nucl':
                    cell_img = nucleus_img
                elif config.cell_image == 'nucl,er':
                    if config.test_cell_image == 'nucl':
                        ER_img = torch.full_like(nucleus_img, fill_value=-2)
                    cell_img = torch.cat([nucleus_img, ER_img], dim=1)
                elif config.cell_image == 'nucl,mt':
                    if config.test_cell_image == 'nucl':
                        microtubules_img = torch.full_like(nucleus_img, fill_value=-2)
                    cell_img = torch.cat([nucleus_img, microtubules_img], dim=1)
                elif config.cell_image == 'nucl,er,mt':
                    if config.test_cell_image == 'nucl':
                        ER_img = torch.full_like(nucleus_img, fill_value=-2)
                        microtubules_img = torch.full_like(nucleus_img, fill_value=-2)
                    elif config.test_cell_image == 'nucl,er':
                        microtubules_img = torch.full_like(nucleus_img, fill_value=-2)
                    elif config.test_cell_image == 'nucl,mt':
                        ER_img = torch.full_like(nucleus_img, fill_value=-2)
                    cell_img = torch.cat([nucleus_img, ER_img, microtubules_img], dim=1)
                else:
                    raise ValueError(f"Cell image type: {config.cell_image} is not supported")

            recon_protein_img = model.recon(
                protein_seq, 
                protein_img, 
                cell_img, 
            )
            save_file = output_dir / ('{:04d}_{:04d}'.format(i+1, j+1) + data['gene_name'])
            save_file.mkdir(parents=True, exist_ok=True)

            save_image(recon_protein_img, save_file / 'reconstruct_protein_img.png', normalize=True, value_range=(-1, 1))
            save_image(protein_img, save_file / 'original_protein_img.png', normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
