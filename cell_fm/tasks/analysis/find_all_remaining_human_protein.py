# -*- coding: utf-8 -*-
"""
Find all human proteins (UniProt Swiss-Prot reviewed) that are NOT covered
by either OpenCell or HPA, i.e.:
    remaining = all_human − (OpenCell ∪ HPA)
"""
from pathlib import Path

import pandas as pd
import requests

OPENCELL_CSV = "/hpc/reference/opencell/opencell/all_merged_meta_data.csv"
HPA_CSV      = "/hpc/reference/opencell/human_protein_atlas/splits/all_merged_meta_data.csv"
OUT_DIR      = Path(__file__).parent / "output"
OUT_CSV      = OUT_DIR / "remaining_human_proteins.csv"

# cached copy of the UniProt download so we don't re-fetch every run
UNIPROT_CACHE = OUT_DIR / "uniprot_human_swissprot.tsv"

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=tsv"
    "&query=reviewed:true+AND+organism_id:9606"
    "&fields=accession,gene_names,protein_name,sequence"
)


def fetch_uniprot_human(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading UniProt cache from {cache_path}")
        return pd.read_csv(cache_path, sep="\t")

    print("Downloading UniProt reviewed human proteome …")
    resp = requests.get(UNIPROT_URL, stream=True, timeout=120)
    resp.raise_for_status()
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"Saved UniProt cache to {cache_path}")
    return pd.read_csv(cache_path, sep="\t")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    uniprot = fetch_uniprot_human(UNIPROT_CACHE)
    # UniProt TSV columns: Entry, Gene Names, Protein names, Sequence
    uniprot = uniprot.rename(columns={
        "Entry":          "uniprot",
        "Gene Names":     "gene_name",
        "Protein names":  "protein_name",
        "Sequence":       "sequence",
    })

    opencell = pd.read_csv(OPENCELL_CSV)
    hpa      = pd.read_csv(HPA_CSV)

    covered_ids = set(opencell["uniprot"].dropna()) | set(hpa["uniprot"].dropna())

    remaining = uniprot[~uniprot["uniprot"].isin(covered_ids)].copy()
    out = remaining[["gene_name", "protein_name", "uniprot", "sequence"]].reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"UniProt reviewed human proteins: {len(uniprot)}")
    print(f"Covered (OpenCell ∪ HPA):        {len(covered_ids)}")
    print(f"Remaining proteins:              {len(out)}")
    print(f"Saved to:                        {OUT_CSV}")


if __name__ == "__main__":
    main()
