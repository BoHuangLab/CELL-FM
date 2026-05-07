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

from tqdm import tqdm


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

    for i, data in enumerate(tqdm(valset, desc="Processing valset")):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        for j, (protein_img, nucleus_img) in enumerate(zip(data['protein_imgs'], data['nucleus_imgs'])):
            protein_img = protein_img.unsqueeze(0).to(device)
            nucleus_img = nucleus_img.unsqueeze(0).to(device)

            cell_img = nucleus_img

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