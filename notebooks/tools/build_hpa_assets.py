"""Build the two standalone assets the NLS screening notebook needs.

The offline task reaches its anchor cell through HPAAllImageDataset, which wants a 62 MB
metadata CSV and the multi-terabyte cell_crops tree, and decodes all 166 of H3C13's 5 MB
PNGs to reach element 33. All of that resolves to one file and a deterministic transform,
so the notebook ships the result instead:

    anchor_cell.npy    (3, 256, 256) float32 in [-1, 1], channels nucleus, ER, microtubules
    anchor_masks.npz   two 256x256 boolean masks, `nucleus` and `cell`

The mask PNGs are hand-painted ROIs belonging to this one anchor cell -- nothing generates
them -- so they travel with it.

    python notebooks/tools/build_hpa_assets.py

Writes into notebooks/tools/assets/, which upload_weights.py reads from. Those two files
are build products of the cluster and are not tracked; the published copies live in the
hpa/ prefix of the weights repo.
"""

import argparse
import os

import numpy as np

# The anchor: gene H3C13, image_paths[33], from all_merged_meta_data.csv.
ANCHOR_PNG = ("/hpc/reference/opencell/human_protein_atlas/cell_crops/1194/"
              "1194_B2_2_4_cell_image.png")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
MASK_DIR = os.path.join(REPO_ROOT, "output", "hpa", "virtual_staining_dict")
NUCLEUS_MASK = os.path.join(MASK_DIR, "chosen_nucleus_masks.png")
CELL_MASK = os.path.join(MASK_DIR, "chosen_ER_masks.png")

IMG_CROP_SIZE = 1024
IMG_RESIZE = 256


def build_anchor():
    """Reproduce HPAAllImageDataset's preprocessing for the one image the demo conditions on.

    Mirrors cell_fm/data/hpa_data/dataset.py:123-141 (channel unpacking) and :48-65 (the
    transform). data_aug is off for this task, so the pipeline is deterministic. The PNG
    stores channels as MT, ER, nucleus, protein; the model wants nucleus, ER, MT.
    """
    import cv2
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor

    data = cv2.imread(ANCHOR_PNG, -1)
    if data is None:
        raise SystemExit(f"cannot read {ANCHOR_PNG}")
    if data.dtype == np.uint16:
        data = data.astype(np.float32) / 65535

    microtubules, er, nucleus, protein = (to_tensor(data[:, :, i]) for i in range(4))
    stacked = torch.stack([protein, nucleus, microtubules, er], dim=0)

    transform = transforms.Compose([
        transforms.CenterCrop(IMG_CROP_SIZE),
        transforms.Resize(IMG_RESIZE, antialias=None),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    protein, nucleus, microtubules, er = transform(stacked)

    anchor = torch.cat([nucleus, er, microtubules], dim=0)
    return anchor.numpy().astype(np.float32), protein.numpy().astype(np.float32)


def build_masks():
    from PIL import Image

    def binary(path):
        return np.array(Image.open(path).convert("RGB"))[:, :, 0] > 0

    nucleus, cell = binary(NUCLEUS_MASK), binary(CELL_MASK)
    if (nucleus & ~cell).any():
        raise SystemExit("nucleus mask is not contained in the cell mask")
    return nucleus, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "assets"),
                    help="where to write the assets (default: alongside this script, "
                         "which is where upload_weights.py looks for them)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    anchor, real_protein = build_anchor()
    nucleus, cell = build_masks()

    anchor_path = os.path.join(args.out, "anchor_cell.npy")
    masks_path = os.path.join(args.out, "anchor_masks.npz")
    np.save(anchor_path, anchor)
    np.savez_compressed(masks_path, nucleus=nucleus, cell=cell)

    print(f"{anchor_path}  {anchor.shape} {anchor.dtype} "
          f"[{anchor.min():.3f}, {anchor.max():.3f}]  {os.path.getsize(anchor_path):,} B")
    print(f"{masks_path}  nucleus {nucleus.sum():,} px, cell {cell.sum():,} px, "
          f"cytoplasm {(cell & ~nucleus).sum():,} px  {os.path.getsize(masks_path):,} B")

    # The real protein channel is not shipped -- the point is to generate it -- but it is
    # the ground truth this anchor's H3C13 staining actually looked like, so report it.
    print(f"(real protein channel, not shipped: [{real_protein.min():.3f}, "
          f"{real_protein.max():.3f}])")


if __name__ == "__main__":
    main()
