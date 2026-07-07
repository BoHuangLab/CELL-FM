# -*- coding: utf-8 -*-
import os
import sys
import math

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from cell_fm.data.opencell_3d_crop_data.dataset import OpenCell3DCropImageOnlyDataset
from cell_fm.models.vae_3d.vae_3d_config import VAE3DConfig
from cell_fm.models.vae_3d.vae_3d_model import VAE3DModel
from cell_fm.utils.cli_utils import cli


@cli(VAE3DConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    valset = OpenCell3DCropImageOnlyDataset(args, split_key="all")

    # Optional cap for a fast, statistically-equivalent estimate (std converges quickly).
    # Set LATENT_STD_MAX_CROPS to limit; unset/0 processes the full dataset.
    max_crops = int(os.environ.get("LATENT_STD_MAX_CROPS", "0"))
    if max_crops > 0 and max_crops < len(valset.img_paths):
        # Deterministic even subsample across the flat crop list.
        stride = len(valset.img_paths) / max_crops
        idx = sorted({int(k * stride) for k in range(max_crops)})
        valset.img_paths = [valset.img_paths[j] for j in idx]
        valset.gene_name_labels = [valset.gene_name_labels[j] for j in idx]
        print(f"Subsampled to {len(valset.img_paths)} crops (LATENT_STD_MAX_CROPS={max_crops})")

    loader = DataLoader(
        valset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=OpenCell3DCropImageOnlyDataset.collate,
    )

    model = VAE3DModel(config=VAE3DConfig(**vars(args)))
    model.to(device)
    model.eval()

    # Global running stats (float64 for numerical stability over the full dataset)
    sum_s = sqsum_s = 0.0      # for .sample()
    sum_m = sqsum_m = 0.0      # for .mean
    count = 0
    chan_sum_s = chan_sqsum_s = None   # per-channel for .sample(), shape (C,)

    n_batches = len(loader)
    with torch.no_grad():
        for i, data in enumerate(loader):
            protein_img = data['batched_data']['protein_img'].to(device)   # (B, 1, D, H, W)

            dist = model.encode(protein_img)
            z = dist.sample().double()                                      # (B, C, D_lat, H_lat, W_lat)
            m = dist.mean.double()

            if chan_sum_s is None:
                C = z.shape[1]
                chan_sum_s = torch.zeros(C, dtype=torch.float64, device=device)
                chan_sqsum_s = torch.zeros(C, dtype=torch.float64, device=device)
                print(f"Latent shape: {tuple(z.shape)}")

            sum_s   += z.sum().item()
            sqsum_s += z.pow(2).sum().item()
            sum_m   += m.sum().item()
            sqsum_m += m.pow(2).sum().item()
            count   += z.numel()

            chan_sum_s   += z.sum(dim=[0, 2, 3, 4])
            chan_sqsum_s += z.pow(2).sum(dim=[0, 2, 3, 4])

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                run_mean = sum_s / count
                run_std = math.sqrt(max(sqsum_s / count - run_mean ** 2, 0.0))
                print(f"[{i+1}/{n_batches}] running sample mean={run_mean:.5f} std={run_std:.5f}")

    # Final global statistics
    mean_s = sum_s / count
    std_s = math.sqrt(max(sqsum_s / count - mean_s ** 2, 0.0))
    mean_m = sum_m / count
    std_m = math.sqrt(max(sqsum_m / count - mean_m ** 2, 0.0))

    chan_n = count // chan_sum_s.shape[0]
    chan_mean_s = chan_sum_s / chan_n
    chan_std_s = (chan_sqsum_s / chan_n - chan_mean_s ** 2).clamp_min(0.0).sqrt()

    print("\n========== 3D VAE latent statistics ==========")
    print(f"Checkpoint : {args.vae_loadcheck_path}")
    print(f"Num crops  : {len(valset)}   Total latent elements: {count}")
    print(f"[sample()] global mean = {mean_s:.6f}   std = {std_s:.6f}")
    print(f"[mean]     global mean = {mean_m:.6f}   std = {std_m:.6f}")
    print(f"[sample()] per-channel std = {[round(v, 6) for v in chan_std_s.tolist()]}")
    print(f"\nSuggested scaling_factor = 1/std = {1.0 / std_s:.6f}")
    print("Apply as: z_scaled = z * scaling_factor (invert before vae.decode)")
    print("==============================================")


if __name__ == "__main__":
    main()
