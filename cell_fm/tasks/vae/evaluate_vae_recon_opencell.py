# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.opencell_crop_data.dataset import OpenCellCropImageOnlyDataset
from cell_fm.models.vae.vae_config import VAEConfig
from cell_fm.models.vae.vae_model import VAEModel
from cell_fm.utils.cli_utils import cli

from torchvision.utils import save_image

@cli(VAEConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gene_name = 'TUBB4B'  # Microtubules
    valset = OpenCellCropImageOnlyDataset(args, split_key="all")
    valset.meta_data = valset.meta_data[valset.meta_data['gene_name'] == gene_name]
    valset.img_paths = valset.gather_image_paths()

    model = VAEModel(config=VAEConfig(**vars(args)))

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)

    for i, data in enumerate(valset):
        print(f"Processing {i+1}/{len(valset)}")
        protein_img = data['protein_img'].unsqueeze(0).to(device)

        protein_img_recon, latent_dist = model.reconstruct(protein_img)

        latents = latent_dist.mean
        latents = latents.norm(dim=1, keepdim=True)
        latents = (latents - latents.min()) / (latents.max() - latents.min())

        save_file = output_dir / ('{:04d}'.format(i+1))
        save_file.mkdir(parents=True, exist_ok=True)

        save_image(protein_img_recon, save_file / 'protein_img_recon.png', normalize=True, value_range=(-1, 1))
        save_image(protein_img, save_file / 'protein_img.png', normalize=True, value_range=(-1, 1))
        save_image(latents, save_file / 'latents_img.png', normalize=True, value_range=(0, 1))

if __name__ == "__main__":
    main()