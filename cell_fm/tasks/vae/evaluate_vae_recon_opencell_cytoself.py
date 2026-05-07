# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.opencell_cytoself_data.dataset import OpenCellCytoselfDataset
from cell_fm.models.vae.vae_config import VAEConfig
from cell_fm.models.vae.vae_model import VAEModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image

@cli(VAEConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    valset = OpenCellCytoselfDataset(args, split_key=args.split_key)

    model = VAEModel(config=VAEConfig(**vars(args)))

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)

    for i, data in enumerate(valset):
        print(f"Processing {i+1}/{len(valset)}")
        protein_img = data['protein_img'].unsqueeze(0).to(device)

        protein_img_recon = model.reconstruct(protein_img)

        save_file = output_dir / ('{:04d}'.format(i+1))
        save_file.mkdir(parents=True, exist_ok=True)

        save_image(protein_img_recon, save_file / 'protein_img_recon.png', normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()