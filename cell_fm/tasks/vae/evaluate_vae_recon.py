# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.hpa_data.dataset import HPAImageOnlyDataset
from cell_fm.models.vae.vae_config import VAEConfig
from cell_fm.models.vae.vae_model import VAEModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image

@cli(VAEConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    antibody = "HPA039247" # TUBA1A antibody
    valset = HPAImageOnlyDataset(args, split_key=args.split_key)
    valset.meta_data = valset.meta_data[valset.meta_data['antibody'] == antibody]

    model = VAEModel(config=VAEConfig(**vars(args)))

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)

    for i, data in enumerate(valset):
        protein_img = data['protein_img'].unsqueeze(0).to(device)

        protein_img_recon = model.reconstruct(protein_img)

        save_file = output_dir / ('{:04d}_'.format(i+1) + data['antibody'])
        save_file.mkdir(parents=True, exist_ok=True)

        print(data['antibody'])
        save_image(protein_img_recon, save_file / 'protein_img_recon.png', normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()