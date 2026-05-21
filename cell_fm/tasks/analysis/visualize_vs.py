# -*- coding: utf-8 -*-
"""
Virtual Staining Viewer
Run:  streamlit run cell_fm/tasks/analysis/visualize_vs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import streamlit as st

DATA_DIR = Path(
    "/hpc/reference/opencell/opencell"
    "/2d_proj_256_crop_virtual_staining_all_remaining_proteins_corrected"
)
CSV_PATH = "/hpc/projects/group.huang/dihan.zheng/CELL-FM/cell_fm/tasks/analysis/output/remaining_human_proteins.csv"

NCOLS = 8
NROWS = 4


@st.cache_data
def load_proteins() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)[["gene_name", "protein_name", "uniprot"]]
    # one single directory listing instead of one stat() per protein
    existing = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}
    df["has_data"] = df["uniprot"].isin(existing)
    return df.reset_index(drop=True)


@st.cache_data
def load_tif_files(uniprot_id: str) -> list[str]:
    return sorted(str(p) for p in (DATA_DIR / uniprot_id).glob("*.tif"))


@st.cache_data
def render_grid(uniprot_id: str, protein_only: bool) -> list[np.ndarray]:
    paths = load_tif_files(uniprot_id)
    return [tif_to_rgb(Path(p), protein_only) for p in paths]


def tif_to_rgb(path: Path, protein_only: bool) -> np.ndarray:
    arr = tifffile.imread(path)        # (2, H, W) uint16
    nucleus = arr[0] / 65535.0
    protein = arr[1] / 65535.0
    if protein_only:
        rgb = np.stack([protein, protein, protein], axis=-1)
    else:
        blue = np.clip(protein + nucleus, 0, 1)
        rgb  = np.stack([protein, protein, blue], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# ── layout ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Virtual Staining Viewer", layout="wide")
st.title("Virtual Staining Viewer")

df = load_proteins()

query = st.text_input(
    "Search by gene name, protein name, or UniProt ID",
    placeholder="e.g. ATG7, Autophagy, Q96EK6",
)

if query:
    mask = (
        df["gene_name"].str.contains(query, case=False, na=False)
        | df["protein_name"].str.contains(query, case=False, na=False)
        | df["uniprot"].str.contains(query, case=False, na=False)
    )
    filtered = df[mask]
else:
    filtered = df

st.caption(f"Showing {len(filtered):,} of {len(df):,} proteins — click a row to view images")

event = st.dataframe(
    filtered[["gene_name", "protein_name", "uniprot", "has_data"]],
    selection_mode="single-row",
    on_select="rerun",
    use_container_width=True,
    height=300,
)

selected_rows = event.selection.rows
if not selected_rows:
    st.info("Select a protein in the table above to view generated images.")
    st.stop()

row        = filtered.iloc[selected_rows[0]]
uniprot_id = str(row["uniprot"])
gene_name  = row["gene_name"]
prot_name  = row["protein_name"]

st.divider()
st.subheader(f"{gene_name} · {prot_name} · `{uniprot_id}`")

folder = DATA_DIR / uniprot_id
if not folder.exists():
    st.error(f"No output folder found for {uniprot_id}.")
    st.stop()

tif_paths = load_tif_files(uniprot_id)
if not tif_paths:
    st.warning(f"Folder exists but contains no TIF files: {folder}")
    st.stop()

protein_only = st.toggle("Protein channel only", value=False)
st.caption(f"{len(tif_paths)} images · {folder}")

images = render_grid(uniprot_id, protein_only)
for row_idx in range(NROWS):
    cols = st.columns(NCOLS)
    for col_idx in range(NCOLS):
        img_idx = row_idx * NCOLS + col_idx
        if img_idx >= len(images):
            break
        cols[col_idx].image(images[img_idx], caption=str(img_idx + 1), use_container_width=True)
