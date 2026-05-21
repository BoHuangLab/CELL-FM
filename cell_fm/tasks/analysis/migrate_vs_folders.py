# -*- coding: utf-8 -*-
"""
Migrate virtual-staining output folders from {gene_name}_{uniprot_id}
to {uniprot_id}, using hardlinks (zero extra disk space, instant).
"""
from pathlib import Path
import pandas as pd

SRC_DIR = Path("/hpc/reference/opencell/opencell/2d_proj_256_crop_virtual_staining_all_remaining_proteins")
DST_DIR = Path("/hpc/reference/opencell/opencell/2d_proj_256_crop_virtual_staining_all_remaining_proteins_corrected")
CSV     = Path("/hpc/projects/group.huang/dihan.zheng/CELL-FM/cell_fm/tasks/analysis/output/remaining_human_proteins.csv")

df = pd.read_csv(CSV)
df = df[df["sequence"].str.len().between(101, 2048)].reset_index(drop=True)

DST_DIR.mkdir(parents=True, exist_ok=True)

copied = skipped = missing = 0

for _, row in df.iterrows():
    gene_name  = str(row["gene_name"])
    uniprot_id = str(row["uniprot"])

    src = SRC_DIR / f"{gene_name}_{uniprot_id}"
    dst = DST_DIR / uniprot_id

    if not src.exists():
        missing += 1
        continue

    if dst.exists():
        skipped += 1
        continue

    dst.mkdir(parents=True, exist_ok=True)
    for tif in src.glob("*.tif"):
        dst_tif = dst / tif.name
        if not dst_tif.exists():
            dst_tif.hardlink_to(tif)

    copied += 1

print(f"Copied:  {copied}")
print(f"Skipped: {skipped}  (already exist in dst)")
print(f"Missing: {missing}  (no src folder)")
print(f"DST total folders: {len(list(DST_DIR.iterdir()))}")
