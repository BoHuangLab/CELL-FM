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

from glob import glob
from tqdm import tqdm

# from dataclasses import dataclass
# @dataclass
# class ExtendedConfig:
#     mutation_type: str = 'WT'

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

@cli(ViTConfig, CondenSeqDatasetConfig)
def main(args) -> None:
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config & data
    config = ViTConfig(**vars(args))
    dataconfig = CondenSeqDatasetConfig(**vars(args))

    # chosen images
    protein_index = 12626
    index = 0

    valset = CondenSeqAllImageDataset(dataconfig, split_key='all')
    valset.meta_data = valset.meta_data[valset.meta_data['index'] == protein_index]
    chosen_data = valset.__getitem__(0)

    chosen_nucleus_img = chosen_data['nucleus_imgs'][index].unsqueeze(0).to(device)

    # model
    model = ViTModel(config=config)
    model.to(device)
    model.eval()

    # mutation_type = args.mutation_type
    all_mutation_types = [f"{aa1}2{aa2}" for aa1 in AA for aa2 in AA if aa1 != aa2]

    # WT
    all_wt_data_paths = sorted(glob('/hpc/reference/opencell/condenseq/seq2img_all_mutation/PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k/cellfm_test_selected_balanced/WT/*'))

    for mutation_type in tqdm(all_mutation_types):
        batch_size = 512

        all_data_paths = sorted(glob(f'/hpc/reference/opencell/condenseq/seq2img_all_mutation/PT_CondenSeq_CELL-Diff2_Rep_Dev_GFP_e_4_ignl_4_ignah_8_S1_R1_50k/cellfm_test_selected_balanced/{mutation_type}/*'))

        for data_path, all_wt_data_path in zip(all_data_paths, all_wt_data_paths):
            mt_protein_seq = np.loadtxt(os.path.join(data_path, 'protein_seq.txt'), dtype=str)
            wt_protein_seq = np.loadtxt(os.path.join(all_wt_data_path, 'protein_seq.txt'), dtype=str)

            # if mt_protein_seq == wt_protein_seq:
            #     continue

            protein_intensity_levels = np.load(os.path.join(data_path, 'protein_intensity_levels.npy'))
            image_path = os.path.join(data_path, 'imgs.tif')

            save_dir = data_path
            save_dir = Path(save_dir)

            img = tiff.imread(image_path)  # [N, H, W]
            img = torch.from_numpy(img).unsqueeze(1).to(device)  # [N, 1, H, W]

            # normalize to [0,1]
            img = img / 65535.0
            # normalize to [-1,1]
            img = img * 2.0 - 1.0

            all_preds = []

            for batch_start in range(0, img.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, img.shape[0])

                batch_input_imgs = img[batch_start:batch_end]  # [B, 1, H, W]
                input_imgs = torch.cat([chosen_nucleus_img.repeat(batch_input_imgs.shape[0],1,1,1), batch_input_imgs], dim=1)  # [B, 2, H, W]

                with torch.no_grad():
                    logits = model.predict(input_imgs)  # [B, num_classes]
                    preds = torch.argmax(logits, dim=1).cpu()  # [B]
                    all_preds.append(preds)

            protein_intensity_levels = protein_intensity_levels.tolist()

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
            plt.savefig(save_dir / 'pred_vs_intensity_jitter.png', dpi=200)
            plt.close()

            # Dump a CSV of (intensity, pred) pairs for downstream analysis
            out_table = np.stack([x_vals, y_vals], axis=1)
            np.savetxt(
                save_dir / "pred_vs_intensity.csv",
                out_table,
                fmt=["%.6f", "%d"],
                delimiter=",",
                header="protein_intensity_level,predicted_class",
                comments=""
            )


if __name__ == "__main__":
    main()
