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

    # chosen data
    
    # hnRNPA1
    # protein_name = "hnRNPA1"
    # protein_index = 8277

    # NUP98
    # protein_name = "NUP98"
    # protein_index = 6179

    # FUS
    # protein_index = 13629

    # DDX4
    # protein_name = "DDX4"
    # protein_index = 1730

    # LAF-1
    # protein_name = "LAF-1"
    # protein_index = 4260

    # LAF-1_2
    # protein_name = "LAF-1_2"
    # protein_index = 1575 

    # UBQLN2
    # protein_name = "UBQLN2"
    # protein_index = 9898

    # protein_name = '7954'
    # protein_index = 7954

    # protein_name = '10086'
    # protein_index = 10086

    # protein_name = '14547'
    # protein_index = 14547

    # protein_name = '10033'
    # protein_index = 10033

    # protein_name = '10492'
    # protein_index = 10492

    # protein_name = '8991'
    # protein_index = 8991

    # protein_name = '452'
    # protein_index = 452

    # protein_name = '10903'
    # protein_index = 10903

    # protein_name = '14584'
    # protein_index = 14584

    protein_name = '13119'
    protein_index = 13119

    valset = CondenSeqAllImageDataset(dataconfig, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    chosen_nucleus_img = torch.stack(chosen_data['nucleus_imgs'])
    chosen_protein_img = torch.stack(chosen_data['protein_imgs'])
    protein_intensity_levels = chosen_data['protein_intensity_levels']

    output_dir = Path(config.output_dir)
    output_dir = output_dir / f'protein_{protein_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # model
    model = ViTModel(config=config)
    model.to(device)
    model.eval()

    batch_size = 256

    all_preds = []

    for batch_start in range(0, chosen_nucleus_img.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, chosen_nucleus_img.shape[0])

        batch_nucleus_imgs = chosen_nucleus_img[batch_start:batch_end].to(device)
        batch_protein_imgs = chosen_protein_img[batch_start:batch_end].to(device)

        # concat channels to match model input layout if your predict() expects [B,2,H,W]
        # nucleus first, then protein
        input_imgs = torch.cat([batch_nucleus_imgs, batch_protein_imgs], dim=1)  # [B, 2, H, W]

        with torch.no_grad():
            logits = model.predict(input_imgs)  # [B, num_classes]

            preds = torch.argmax(logits, dim=1).cpu()  # [B]
            all_preds.append(preds)
    
    all_preds = torch.cat(all_preds)  # [N] on CPU

    # Sort everything by protein intensity level
    x_vals = np.array(protein_intensity_levels)     # shape [N]
    sort_idx = np.argsort(x_vals)
    x_vals = x_vals[sort_idx]
    all_preds = all_preds[sort_idx]
    chosen_nucleus_img = chosen_nucleus_img[sort_idx]
    chosen_protein_img = chosen_protein_img[sort_idx]

    # -----------------------------
    # Jitter plot: all_preds vs protein_intensity_levels
    # -----------------------------
    # x-axis: intensity
    # y-axis: predicted class (with jitter)
    y_base = all_preds.numpy().astype(int)          # shape [N]

    # jitter strength (tune these)
    y_jitter_scale = 0.01   # jitter in "class units" (e.g., 0.1~0.25)
    x_jitter_scale = 0.0    # set >0 (e.g., 0.01 * x_vals.std()) if x also overlaps a lot

    rng = np.random.default_rng(0)
    y_vals = y_base + rng.normal(0, y_jitter_scale, size=y_base.shape)
    x_plot = x_vals + rng.normal(0, x_jitter_scale, size=x_vals.shape)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        x_plot,
        y_vals,
        alpha=0.6,
        s=20,
    )

    plt.xlabel("Protein Intensity Level")
    plt.ylabel("Predicted Class")
    plt.title("Predicted Class vs Protein Intensity Level (Jitter)")

    unique_classes = np.unique(y_base)
    plt.yticks(unique_classes, [str(c) for c in unique_classes])

    # optionally keep y-limits tight around class range
    plt.ylim(unique_classes.min() - 0.6, unique_classes.max() + 0.6)

    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'pred_vs_intensity.png', dpi=200)
    plt.close()

    # (optional) also dump a CSV of (intensity, pred) pairs for downstream analysis
    out_table = np.stack([x_vals, y_base], axis=1)
    np.savetxt(
        output_dir / "pred_vs_intensity.csv",
        out_table,
        fmt=["%.6f", "%d"],
        delimiter=",",
        header="protein_intensity_level,predicted_class",
        comments=""
    )

    chosen_img = torch.cat([chosen_nucleus_img, chosen_protein_img], dim=1)  # [N, 2, H, W]
    chosen_img = (chosen_img + 1) / 2.0  # shift to [0,1]
    chosen_img = chosen_img.clamp(0.0, 1.0)

    save_tif(
        chosen_img.cpu().numpy(),
        output_dir / "combined_images.tif", 
    )

if __name__ == "__main__":
    main()