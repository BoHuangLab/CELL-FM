# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_fm.models.vit_cls_condenseq_img.config import ViTConfig
from cell_fm.models.vit_cls_condenseq_img.model import ViTModel
from cell_fm.utils.cli_utils import cli

import numpy as np

import matplotlib
matplotlib.use("Agg")  # important for headless servers
import matplotlib.pyplot as plt
import tifffile as tiff

from glob import glob
from tqdm import tqdm


def save_tif(image, output_path):
    # image is expected in [0,1] float, shape [N, C, H, W] or similar
    # We will just save raw array as uint16 after scaling.
    image = image * 65535.0
    image = image.astype(np.uint16)
    tiff.imwrite(output_path, image, imagej=True)


@cli(ViTConfig)
def main(args) -> None:
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config & data
    config = ViTConfig(**vars(args))

    # model
    model = ViTModel(config=config)
    model.to(device)
    model.eval()

    # data_roots = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE_panel_v2_selected_log_scale/*/*/'))
    data_roots = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_charged_domain_mutation_random_KRDE_panel_v3_log_scale/*/*/'))

    for data_root in tqdm(data_roots):
        data_root = Path(data_root)

        img = tiff.imread(data_root / 'combined_images.tif')  # [N, 2, H, W]
        protein_intensity_levels = np.load(data_root / 'protein_intensity_levels.npy')  # [N]

        output_dir = data_root

        batch_size = 512

        all_preds = []

        for batch_start in range(0, img.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, img.shape[0])

            batch_input_imgs = torch.from_numpy(img[batch_start:batch_end]).to(device)  # [B, 2, H, W]

            # normalize to [0,1]
            batch_input_imgs = batch_input_imgs / 65535.0
            # normalize to [-1,1]
            batch_input_imgs = batch_input_imgs * 2.0 - 1.0

            with torch.no_grad():
                logits = model.predict(batch_input_imgs)  # [B, num_classes]
                preds = torch.argmax(logits, dim=1).cpu()  # [B]

                all_preds.append(preds)

        # -----------------------------
        # Plot all_preds vs protein_intensity_levels (with jitter)
        # -----------------------------
        x_vals = np.array(protein_intensity_levels)  # shape [N]
        y_vals = torch.cat(all_preds).cpu().numpy().astype(int)  # shape [N]

        rng = np.random.default_rng(seed=42)
        jitter_strength = 0.05
        y_jittered = y_vals + rng.normal(0, jitter_strength, size=y_vals.shape)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            x_vals,
            y_jittered,
            alpha=0.5,
            s=20,
            c=y_vals,
            cmap="coolwarm",
            edgecolors="none"
        )

        plt.xlabel("Protein Intensity Level")
        plt.ylabel("Predicted Class (jittered)")
        plt.title("Predicted Class vs Protein Intensity Level (with jitter)")

        unique_classes = np.unique(y_vals)
        plt.yticks(unique_classes, [str(c) for c in unique_classes])

        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / 'pred_vs_intensity_jitter.png', dpi=200)
        plt.close()

        # Dump a CSV of (intensity, pred) pairs for downstream analysis
        out_table = np.stack([x_vals, y_vals], axis=1)
        np.savetxt(
            output_dir / "pred_vs_intensity.csv",
            out_table,
            fmt=["%.6f", "%d"],
            delimiter=",",
            header="protein_intensity_level,predicted_class",
            comments=""
        )


if __name__ == "__main__":
    main()