"""Build the standalone assets the PLS generation notebook needs.

The offline task reaches its conditioning cell through HPAAllImageDataset, which wants a
62 MB metadata CSV and the multi-terabyte cell_crops tree, and decodes all 127 (NLS) or
103 (NES) of the gene's 4-5 MB PNGs to reach one element. The analysis then wants a 60 MB
proteome CSV purely to count twenty integers. All of that resolves to a handful of small
files, so the notebook ships those instead:

    pls_anchor_nls.npz       cell (3, 512, 512) + protein (1, 512, 512), float32 in [-1, 1]
    pls_anchor_nes.npz       same, for the cytosolic anchor
    proteome_aa_counts.json  20 residue counts over the 12,894 HPA proteins
    pls_reference_nls.csv    the published 315 generated signals
    pls_reference_nes.csv    the published 320 generated signals

Two things differ from build_hpa_assets.py, and both matter:

  512 px, not 256   img2seq is a 512 px / sample_size=128 model. Conditioning it on a
                    256 px image is a latent-size mismatch, not a resize.
  protein ships     image_to_sequence takes the protein channel as a *conditioning input*
                    (cell_fm/models/cell_fm/cell_fm_model.py:245). For seq2img it was
                    deliberately withheld, because generating it is the point; here the
                    model reads it.

    python notebooks/tools/build_pls_assets.py

Writes into notebooks/tools/assets/, which upload_weights.py reads from. These are build
products of the cluster and are not tracked; the published copies live in the hpa/ prefix
of the weights repo.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
HPA_ROOT = "/hpc/reference/opencell/human_protein_atlas"

# The two conditioning cells, hard-coded in cell_fm/tasks/cell_fm/pls_generation_hpa.py:69-73
# as (split, dataset index, image index). Resolved here to the single PNG each one reaches.
ANCHORS = {
    # gene PPM1G, Nucleoplasm -- split 'all' item 8259, image_paths[10]
    "nls": ("PPM1G", "Nucleoplasm", f"{HPA_ROOT}/cell_crops/392/392_B9_1_11_cell_image.png"),
    # gene DIAPH1, Cytosol + Plasma membrane -- split 'test' item 24, image_paths[0]
    "nes": ("DIAPH1", "Cytosol, Plasma membrane",
            f"{HPA_ROOT}/cell_crops/1608/1608_B3_1_1_cell_image.png"),
}

BASELINE_CSV = f"{HPA_ROOT}/splits/all_merged_meta_data.csv"
PUBLISHED = {
    "nls": os.path.join(REPO_ROOT, "output/hpa/pls_generation/nls/gen_signals.csv"),
    "nes": os.path.join(REPO_ROOT, "output/hpa/pls_generation/nes/gen_signals.csv"),
}

IMG_CROP_SIZE = 1024
IMG_RESIZE = 512  # the img2seq model's resolution; see the module docstring
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def build_anchor(png_path):
    """Reproduce HPAAllImageDataset's preprocessing for one conditioning image.

    Mirrors cell_fm/data/hpa_data/dataset.py:123-141 (channel unpacking) and :48-65 (the
    transform). data_aug is off for this task, so the pipeline is deterministic. The PNG
    stores channels as MT, ER, nucleus, protein; the model wants nucleus, ER, MT, and
    separately the protein channel.
    """
    import cv2
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor

    data = cv2.imread(png_path, -1)
    if data is None:
        raise SystemExit(f"cannot read {png_path}")
    if data.dtype == np.uint16:
        data = data.astype(np.float32) / 65535
    elif data.dtype == np.uint8:
        data = data.astype(np.float32) / 255
    else:
        raise SystemExit(f"unexpected dtype {data.dtype} in {png_path}")

    microtubules, er, nucleus, protein = (to_tensor(data[:, :, i]) for i in range(4))
    stacked = torch.stack([protein, nucleus, microtubules, er], dim=0)

    transform = transforms.Compose([
        transforms.CenterCrop(IMG_CROP_SIZE),
        transforms.Resize(IMG_RESIZE, antialias=None),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    protein, nucleus, microtubules, er = transform(stacked)

    # concat order matches cell_image='nucl,er,mt' at cell_fm_model.py:190
    cell = torch.cat([nucleus, er, microtubules], dim=0)
    return cell.numpy().astype(np.float32), protein.numpy().astype(np.float32)


def build_baseline_counts():
    """Twenty integers standing in for a 60 MB CSV.

    ana_pls_gen_dis_with_baseline.py compares generated signals against the HPA proteome
    with a per-residue chi-squared, which needs counts rather than frequencies -- so the
    counts are exactly what has to travel, and nothing else does.
    """
    import pandas as pd

    df = pd.read_csv(BASELINE_CSV, low_memory=False)
    counter = Counter()
    for seq in df["sequence"].dropna():
        counter.update(c for c in str(seq).upper() if c in AMINO_ACIDS)
    return {
        "source": "human_protein_atlas/splits/all_merged_meta_data.csv",
        "n_proteins": int(len(df)),
        "counts": {aa: int(counter[aa]) for aa in AMINO_ACIDS},
    }


def read_published(path):
    """The published signals. csv.writer left CRLF endings; strip and rewrite as plain lines."""
    with open(path, newline="") as f:
        return [line.strip().upper() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "assets"),
                    help="where to write the assets (default: alongside this script, "
                         "which is where upload_weights.py looks for them)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for pls_type, (gene, locations, png) in ANCHORS.items():
        cell, protein = build_anchor(png)
        path = os.path.join(args.out, f"pls_anchor_{pls_type}.npz")
        np.savez_compressed(path, cell=cell, protein=protein)
        print(f"{path}")
        print(f"    {gene} ({locations})")
        print(f"    cell    {cell.shape} [{cell.min():.3f}, {cell.max():.3f}]  "
              "channels nucleus, ER, microtubules")
        print(f"    protein {protein.shape} [{protein.min():.3f}, {protein.max():.3f}]")
        print(f"    {os.path.getsize(path):,} B")

    baseline = build_baseline_counts()
    path = os.path.join(args.out, "proteome_aa_counts.json")
    with open(path, "w") as f:
        json.dump(baseline, f, indent=1)
    total = sum(baseline["counts"].values())
    print(f"{path}  {baseline['n_proteins']:,} proteins, {total:,} residues  "
          f"{os.path.getsize(path):,} B")

    for pls_type, src in PUBLISHED.items():
        seqs = read_published(src)
        path = os.path.join(args.out, f"pls_reference_{pls_type}.csv")
        with open(path, "w") as f:
            f.write("signal\n")
            f.write("".join(f"{s}\n" for s in seqs))
        lengths = sorted({len(s) for s in seqs})
        print(f"{path}  {len(seqs)} signals, {lengths[0]}-{lengths[-1]} aa  "
              f"{os.path.getsize(path):,} B")


if __name__ == "__main__":
    main()
