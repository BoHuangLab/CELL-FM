# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.data.condenseq_data.config import CondenSeqDatasetConfig
from cell_fm.data.condenseq_data.dataset import CondenSeqAllImageDataset
from cell_fm.models.vit_cls_condenseq_img.config import ViTConfig
from cell_fm.models.vit_cls_condenseq_img.model import ViTModel
from cell_fm.utils.cli_utils import cli

import numpy as np

import matplotlib
matplotlib.use("Agg")  # important for headless servers
import matplotlib.pyplot as plt
import tifffile as tiff


def save_tif(image, output_path):
    # image is expected in [0,1] float, shape [N, C, H, W] or similar
    # We will just save raw array as uint16 after scaling.
    image = image * 65535.0
    image = image.astype(np.uint16)
    tiff.imwrite(output_path, image, imagej=True)


@cli(ViTConfig, CondenSeqDatasetConfig)
def main(args) -> None:
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config & data
    config = ViTConfig(**vars(args))
    dataconfig = CondenSeqDatasetConfig(**vars(args))

    testset = CondenSeqAllImageDataset(dataconfig, split_key=dataconfig.split_key)

    output_dir = Path(config.output_dir) / config.split_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # model
    model = ViTModel(config=config)
    model.to(device)
    model.eval()

    batch_size = 32

    # loop over each protein (each entry in testset = one protein, multiple cells/images)
    for i, data in enumerate(testset):
        print(f"Evaluating protein {i+1}/{len(testset)}")

        # stack all images for this protein
        protein_imgs = torch.stack(data['protein_imgs']).to(device)    # [N, 1, H, W], range [-1,1]
        nucleus_imgs = torch.stack(data['nucleus_imgs']).to(device)    # [N, 1, H, W]

        # intensity levels per image
        protein_intensity_levels = torch.tensor(
            data['protein_intensity_levels']
        ).float()  # [N]
        protein_intensity_levels = protein_intensity_levels.unsqueeze(-1).to(device)  # [N,1]

        # make dir for this protein / index
        save_file = output_dir / ('{:04d}_'.format(i+1) + str(data['index']))
        save_file.mkdir(parents=True, exist_ok=True)

        # concat channels to match model input layout if your predict() expects [B,2,H,W]
        # nucleus first, then protein
        input_imgs = torch.cat([nucleus_imgs, protein_imgs], dim=1)  # [N, 2, H, W]

        all_preds = []

        # run batched inference
        with torch.no_grad():
            for batch_start in range(0, input_imgs.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, input_imgs.shape[0])

                batch_input_imgs = input_imgs[batch_start:batch_end]  # [B, 2, H, W]
                logits = model.predict(batch_input_imgs)  # [B, num_classes]

                preds = torch.argmax(logits, dim=1).cpu()  # [B]
                all_preds.append(preds)

        all_preds = torch.cat(all_preds)  # [N] on CPU
        # move intensity to CPU and flatten
        intensity_cpu = protein_intensity_levels.squeeze(-1).cpu()  # [N]

        # -----------------------------
        # Save real protein images as tif
        # -----------------------------
        # undo [-1,1] -> [0,1] for saving
        # protein_imgs_vis = (protein_imgs + 1) / 2.0
        # protein_imgs_vis = protein_imgs_vis.clamp(0, 1)

        # save_tif(
        #     protein_imgs_vis.detach().cpu().numpy(), 
        #     save_file / 'real_protein_imgs.tif'
        # )

        # -----------------------------
        # Plot all_preds vs protein_intensity_levels
        # -----------------------------
        # x-axis: intensity
        # y-axis: predicted class (int)
        x_vals = intensity_cpu.numpy()          # shape [N]
        y_vals = all_preds.numpy().astype(int)  # shape [N]

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_vals,
            y_vals,
            alpha=0.6,
            s=20,
        )

        plt.xlabel("Protein Intensity Level")
        plt.ylabel("Predicted Class")
        plt.title("Predicted Class vs Protein Intensity Level")

        # for nicer readability when classes are categorical (0/1 or 0/1/2/3)
        unique_classes = np.unique(y_vals)
        plt.yticks(unique_classes, [str(c) for c in unique_classes])

        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_file / 'pred_vs_intensity.png', dpi=200)
        plt.close()

        # (optional) also dump a CSV of (intensity, pred) pairs for downstream analysis
        out_table = np.stack([x_vals, y_vals], axis=1)
        np.savetxt(
            save_file / "pred_vs_intensity.csv",
            out_table,
            fmt=["%.6f", "%d"],
            delimiter=",",
            header="protein_intensity_level,predicted_class",
            comments=""
        )


if __name__ == "__main__":
    main()
