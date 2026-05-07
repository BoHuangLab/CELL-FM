# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from stable_diffusion.models.stable_diffusion.vae_config import VAEConfig
from stable_diffusion.models.stable_diffusion.vae_model import VAEModel
from stable_diffusion.utils.cli_utils import cli

from torchvision.utils import save_image

@cli(VAEConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    model = VAEModel(config=VAEConfig(**vars(args)))

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = 64

    sample = model.sample(num_samples, 64, device)

    save_image(sample, output_dir / 'sample.png', nrow=8, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()