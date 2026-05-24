# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

import torch
from torchvision.utils import save_image

from cell_fm.criterions.cell_fmc.img_diff import ImgDiffCriterion
from cell_fm.data.hpa_data.dataset_with_context import HPAAllImageDatasetWithContext
from cell_fm.models.cell_fmc.cell_fmc_config import CELLFMCConfig
from cell_fm.models.cell_fmc.cell_fmc_model import CELLFMCModel
from cell_fm.utils.cli_utils import cli
from cell_fm.logging import logger


@cli(CELLFMCConfig)
def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = CELLFMCConfig(**vars(args))
    valset = HPAAllImageDatasetWithContext(args, split_key=config.split_key)
    model = CELLFMCModel(config=config, loss_fn=ImgDiffCriterion)
    model.to(device)
    model.eval()

    output_dir = Path(config.output_dir) / config.split_key
    batch_size = 32

    for i, data in enumerate(valset):
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)
        if protein_seq.shape[1] > config.max_protein_sequence_len + 2:
            continue

        gene_name = data['gene_name']
        logger.info(gene_name)

        protein_imgs = torch.stack(data['protein_imgs']).to(device)
        nucleus_imgs = torch.stack(data['nucleus_imgs']).to(device)
        microtubules_imgs = torch.stack(data['microtubules_imgs']).to(device)
        ER_imgs = torch.stack(data['ER_imgs']).to(device)
        cell_imgs = torch.cat([nucleus_imgs, ER_imgs, microtubules_imgs], dim=1)

        interaction_context = data['interaction_context'].unsqueeze(0).to(device)  # [1, n_context, D]
        interaction_context = interaction_context.repeat(cell_imgs.shape[0], 1, 1)  # [B, n_context, D]
        protein_seq = protein_seq.repeat(cell_imgs.shape[0], 1)  # [B, L]

        save_dir = output_dir / '{:04d}_{}'.format(i + 1, gene_name)
        generated_dir = save_dir / 'generated'
        real_dir = save_dir / 'real'
        generated_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)

        for batch_start in range(0, cell_imgs.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, cell_imgs.shape[0])
            batch_cell_imgs = cell_imgs[batch_start:batch_end]
            batch_protein_seq = protein_seq[batch_start:batch_end]
            batch_context = interaction_context[batch_start:batch_end]
            batch_prot_imgs = protein_imgs[batch_start:batch_end]

            sample = model.sequence_to_image(
                batch_protein_seq,
                batch_cell_imgs,
                batch_context,
                num_steps=config.num_steps,
            )

            for j in range(batch_start, batch_end):
                k = j - batch_start
                save_image(sample[k], generated_dir / f'img_{j}.png', normalize=True, value_range=(-1, 1))
                save_image(batch_prot_imgs[k], real_dir / f'img_{j}.png', normalize=True, value_range=(-1, 1))

    logger.info("Done!")


if __name__ == "__main__":
    main()
