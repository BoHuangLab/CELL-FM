"""Build the standalone assets the OpenCell virtual-staining notebook needs.

The offline task reaches its conditioning cell through OpenCellCropAllImageDataset, which
wants a 12 MB metadata CSV and the 25 GB cell_crops tree, and decodes all 86 of ATG7's
crops to reach one of them. The notebook ships these instead:

    vs_anchor_nucleus.npy   (1, 256, 256) float32 in [-1, 1] -- the one nucleus every
                            offline OpenCell run conditions on
    vs_genes.csv            1311 genes: name, protein name, uniprot, ensembl, locations,
                            sequence. The metadata CSV minus image_paths, which is 92% of it
    vs_reference_cells.npz  4 genes x (nucleus, protein), float16 -- the real image shown
                            beside each generation in section 2

The vs_ prefix is load-bearing: upload_weights.py --only is a plain substring match, so
without it "--only opencell" would sweep up cellfm_vs.bin and re-push 3.7 GB of weights to
refresh a 0.9 MB csv. "--only opencell/vs_" selects exactly these three.

Two things differ from build_hpa_assets.py, and both matter:

  one channel      cell_image='nucl' with in_channels=1 makes cond_conv expect a
                   (B, 1, 256, 256) conditioning tensor (cell_fm_model.py:299-302). The HPA
                   builder concatenates three channels; concatenating here would fail
                   inside a Conv2d with a shape error.
  the protein
  channel ships    but only for the reference genes, and paired with *that gene's
                   own* nucleus. Generating on the shared ATG7 anchor and comparing against
                   a real image from some other cell would not be like for like.

    python notebooks/tools/build_opencell_assets.py

Writes into notebooks/tools/assets/, which upload_weights.py reads from. These are build
products of the cluster and are not tracked; the published copies live in the opencell/
prefix of the weights repo.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPENCELL_ROOT = "/hpc/reference/opencell/opencell"
META_CSV = f"{OPENCELL_ROOT}/all_2d_proj_256_crop_merged_meta_data.csv"

# The conditioning cell, hard-coded in cell_fm/tasks/cell_fm/virtual_staining_opencell.py
# :41-45 as (gene ATG7, nucleus_imgs[1]). Every published OpenCell generation uses it, which
# is what makes those images comparable to each other -- and what lets the check at the
# bottom of this file confirm the reproduction bit for bit.
ANCHOR_GENE = "ATG7"
ANCHOR_INDEX = 1

# One gene per visually distinct compartment, drawn from the 327 that carry a *single*
# location label, have at least 40 crops and fit inside max_protein_sequence_len. Textbook
# markers where there is one: LMNB1 is lamin B1, TOMM20 the standard mitochondrial import
# receptor, PXN is paxillin, MECP2 the Rett syndrome chromatin protein.
REFERENCE_GENES = ["POLR1A", "SNRPF", "LSM14A", "DDX6"]
# Two matched pairs, chosen so the model has something to prove rather than a set of easy
# wins: POLR1A and SNRPF are both nuclear and should not look alike (RNA polymerase I in the
# fibrillar centre of the nucleolus, against an Sm core protein spread over chromatin), and
# LSM14A and DDX6 carry the *same* OpenCell annotation, big_aggregates, being both P-body
# proteins -- a model that had only learned the coarse label would render them identically.

# Which crop to bake. Index 0 alone is a poor default: measured across the first eight, it
# has the weakest protein-channel contrast for several genes, which would put washed-out
# cells in the panel. Take the highest-contrast of the first few instead -- deterministic,
# and a display choice rather than a filter on anything measured.
#
# One crop per gene, not sixteen: the notebook shows a single real image beside each
# generation and embeds only generated cells, so more would be download nobody reads.
REFERENCE_CANDIDATES = 8

IMG_CROP_SIZE = 256
IMG_RESIZE = 256


def get_img(image_path):
    """Reproduce OpenCellCropDataset.get_img for one crop.

    Mirrors cell_fm/data/opencell_crop_data/dataset.py:123-143 (the percentile clip and
    min-max) and :50-67 (the transform). data_aug is off for evaluation, so the chain is
    deterministic: at 256 px the crop and resize are both no-ops and only the normalisation
    does anything, but they are kept so this stays a transcription rather than a shortcut.

    The TIFF stores nucleus in channel 0 and protein in channel 1; the tensor handed to the
    transform is stacked the other way round, [protein, nucleus], which is why the return
    is in that order too.
    """
    import tifffile as tiff
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor

    img = tiff.imread(os.path.join(OPENCELL_ROOT, image_path)).astype(np.float32)
    if img.shape[0] != 2:
        raise SystemExit(f"expected a 2-channel tiff, got {img.shape} in {image_path}")
    nucleus_img, protein_img = img[0], img[1]

    # clip the images to 1% - 99% range through np.percentile
    nucleus_img = np.clip(nucleus_img, np.percentile(nucleus_img, 1), np.percentile(nucleus_img, 99))
    protein_img = np.clip(protein_img, np.percentile(protein_img, 1), np.percentile(protein_img, 99))

    # normalize the images to [0, 1]
    nucleus_img = (nucleus_img - nucleus_img.min()) / (nucleus_img.max() - nucleus_img.min())
    protein_img = (protein_img - protein_img.min()) / (protein_img.max() - protein_img.min())

    stacked = torch.stack([to_tensor(protein_img), to_tensor(nucleus_img)], dim=0)
    transform = transforms.Compose([
        transforms.CenterCrop(IMG_CROP_SIZE),
        transforms.Resize(IMG_RESIZE, antialias=None),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    protein_img, nucleus_img = transform(stacked)
    return protein_img.numpy(), nucleus_img.numpy()


def image_paths(row):
    """The comma-joined path list is the only index into the crop tree; there is no
    filename convention to reconstruct, and crops that failed QC are simply absent."""
    return [p.strip() for p in row["image_paths"].split(",")]


def verify_anchor(anchor):
    """Check the anchor against the offline task's own output.

    Every tif under 2d_proj_256_crop_dataset_virtual_staining_same_nucl carries the
    conditioning nucleus in channel 0, written as ((x + 1) / 2).clip(0, 1) * 65535 rounded
    to uint16 (virtual_staining_opencell.py:77-91). So the published generations pin down
    the anchor exactly, and a mismatch here means the preprocessing above has drifted from
    the dataset class -- which would otherwise show up only as subtly wrong images.
    """
    import glob

    import tifffile as tiff

    published = sorted(glob.glob(
        f"{OPENCELL_ROOT}/2d_proj_256_crop_dataset_virtual_staining_same_nucl/*/*.tif"))
    if not published:
        print("    (no published generations on disk; skipping the exactness check)")
        return

    expect = np.round(((anchor + 1) / 2).clip(0, 1) * 65535).astype(np.uint16)[0]
    sample = published[:: max(1, len(published) // 40)][:40]
    bad = [f for f in sample if not np.array_equal(tiff.imread(f)[0], expect)]
    if bad:
        raise SystemExit(
            f"anchor does not match the published generations ({len(bad)}/{len(sample)} "
            f"differ, e.g. {bad[0]}) -- the preprocessing above has drifted")
    print(f"    verified bit for bit against {len(sample)} of {len(published):,} "
          "published generations")


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "assets"),
                    help="where to write the assets (default: alongside this script, "
                         "which is where upload_weights.py looks for them)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pd.read_csv(META_CSV, low_memory=False)
    meta_by_gene = meta.set_index("gene_name")

    # --- the shared conditioning nucleus -------------------------------------------------
    anchor_row = meta_by_gene.loc[ANCHOR_GENE]
    anchor_path = image_paths(anchor_row)[ANCHOR_INDEX]
    _, anchor = get_img(anchor_path)
    path = os.path.join(args.out, "vs_anchor_nucleus.npy")
    np.save(path, anchor)
    print(f"{path}")
    print(f"    {ANCHOR_GENE} crop {ANCHOR_INDEX}: {os.path.basename(anchor_path)}")
    print(f"    {anchor.shape} [{anchor.min():.3f}, {anchor.max():.3f}]  "
          f"{os.path.getsize(path):,} B")
    verify_anchor(anchor)

    # --- the gene table ------------------------------------------------------------------
    # Everything except image_paths, which is 92% of the file and useless without the crops.
    table = meta[["gene_name", "protein_name", "uniprot", "ensembl", "locations", "sequence"]]
    path = os.path.join(args.out, "vs_genes.csv")
    table.to_csv(path, index=False)
    lengths = table.sequence.str.len()
    print(f"{path}  {len(table):,} genes, {os.path.getsize(path):,} B")
    print(f"    sequence length {lengths.min()}-{lengths.max()} aa, median {int(lengths.median())}; "
          f"{int((lengths > 2048).sum())} over max_protein_sequence_len")

    # --- the reference cells -------------------------------------------------------------
    arrays, index = {}, []
    for gene in REFERENCE_GENES:
        if gene not in meta_by_gene.index:
            raise SystemExit(f"reference gene {gene} is not in {META_CSV}")
        row = meta_by_gene.loc[gene]
        candidates = image_paths(row)[:REFERENCE_CANDIDATES]
        pairs = [get_img(p) for p in candidates]
        best = int(np.argmax([p.std() for p, _ in pairs]))
        protein, nucleus = pairs[best]
        arrays[f"{gene}/protein"] = protein.astype(np.float16)
        arrays[f"{gene}/nucleus"] = nucleus.astype(np.float16)
        index.append({"gene_name": gene, "locations": row["locations"],
                      "protein_name": row["protein_name"], "uniprot": row["uniprot"],
                      "crop": os.path.basename(candidates[best]),
                      "contrast": round(float(protein.std()), 3)})

    idx = pd.DataFrame(index)
    arrays["index"] = idx.to_csv(index=False)
    path = os.path.join(args.out, "vs_reference_cells.npz")
    np.savez_compressed(path, **arrays)
    print(f"{path}  {len(REFERENCE_GENES)} genes, {os.path.getsize(path):,} B")
    print(idx[["gene_name", "locations", "contrast", "crop"]].to_string(index=False))


if __name__ == "__main__":
    main()
