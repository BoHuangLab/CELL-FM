"""Push the CELL-FM checkpoints and notebook assets to a HF model repo.

NOTE: pass a token via the HF_TOKEN environment variable; never write one into this
file or beside it.

The Space pulls its weights from that repo at startup rather than carrying ~2.1 GB
of LFS itself. Run this once from the cluster:

    python notebooks/tools/upload_weights.py --repo BoHuangLab/CELL-FM

The built assets come from build_hpa_assets.py, build_pls_assets.py and
build_opencell_assets.py, which must run first. --only takes a substring of the target
name, so --only opencell/vs_ refreshes the OpenCell notebook's assets without re-pushing
the 3.7 GB of weights that --only opencell would also match.

Add --private to keep the weights unlisted, and --dry-run to see what would be
uploaded without touching the Hub.
"""

import argparse
import os

DEFAULT_SOURCES = {
    # target name in the repo -> checkpoint on the cluster
    "condenseq/cellfm_seq2img.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_condenseq/cellfm_seq2img/checkpoint-50000/pytorch_model.bin"
    ),
    "condenseq/vae.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_condenseq/vae/checkpoint-50000/pytorch_model.bin"
    ),
    "condenseq/vit_cls.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-Diff2-Dev/"
        "pretrain_condenseq_celldiff2_split/PT_CondenSeq_img_ViT_cls_R1/"
        "checkpoint-10000/pytorch_model.bin"
    ),
    # NLS screening: HPA virtual staining. The two assets are built by build_hpa_assets.py
    # and stand in for the multi-terabyte HPA crops tree -- the notebook needs exactly one
    # anchor cell and the two masks drawn on it, nothing else.
    "hpa/cellfm_seq2img.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_hpa/cellfm_seq2img/checkpoint-50000/pytorch_model.bin"
    ),
    "hpa/vae.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_hpa/vae/checkpoint-50000/pytorch_model.bin"
    ),
    # img2seq runs at 512 px and needs its own VAE; vae.bin above is the 256 px one that
    # pairs with the seq2img generator, and the NLS notebook fetches it by that name.
    "hpa/cellfm_img2seq.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_hpa/cellfm_img2seq/checkpoint-60000/pytorch_model.bin"
    ),
    "hpa/vae_512.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "pretrain_hpa/vae_512/checkpoint-50000/pytorch_model.bin"
    ),
    # PLS generation: img2seq conditions on a cell that already shows the localisation
    # being asked for, so each anchor ships its protein channel as well as the cell stack.
    # Built by build_pls_assets.py, which must run first.
    "hpa/pls_anchor_nls.npz": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "pls_anchor_nls.npz"),
    "hpa/pls_anchor_nes.npz": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "pls_anchor_nes.npz"),
    "hpa/proteome_aa_counts.json": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "proteome_aa_counts.json"),
    "hpa/pls_reference_nls.csv": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "pls_reference_nls.csv"),
    "hpa/pls_reference_nes.csv": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "pls_reference_nes.csv"),
    "hpa/anchor_cell.npy": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "anchor_cell.npy"),
    "hpa/anchor_masks.npz": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "anchor_masks.npz"),
    # OpenCell virtual staining, fine-tuned from the pretrained model rather than trained
    # from scratch. 256 px with a single nucleus conditioning channel, where the HPA pair
    # takes three -- so this generator wants opencell/vae.bin and nothing else: its own
    # copy of the VAE tensors is bitwise identical to that file.
    "opencell/cellfm_vs.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "finetune_opencell/cellfm_vs/checkpoint-100000/pytorch_model.bin"
    ),
    "opencell/vae.bin": (
        "/hpc/projects/group.huang/dihan.zheng/CELL-FM/"
        "finetune_opencell/vae/checkpoint-50000/pytorch_model.bin"
    ),
    # OpenCell virtual staining assets, from build_opencell_assets.py. The vs_ prefix keeps
    # them addressable by --only without dragging cellfm_vs.bin along; see the note above.
    "opencell/vs_anchor_nucleus.npy": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "vs_anchor_nucleus.npy"),
    "opencell/vs_genes.csv": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "vs_genes.csv"),
    "opencell/vs_reference_cells.npz": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "vs_reference_cells.npz"),
}

CARD = """---
license: mit
library_name: cell-fm
tags:
  - biology
  - microscopy
  - protein
  - condensate
  - flow-matching
---

# CELL-FM weights

Checkpoints behind the [CELL-FM CondenSeq demo]({space_url}).

| File | Model | Source checkpoint |
|---|---|---|
| `condenseq/cellfm_seq2img.bin` | CELL-FM CS sequence-to-image generator (includes the ESM-C 600M encoder) | `pretrain_condenseq/cellfm_seq2img/checkpoint-50000` |
| `condenseq/vae.bin` | Image VAE, 160x160, 3 down blocks, 4 latent channels | `pretrain_condenseq/vae/checkpoint-50000` |
| `condenseq/vit_cls.bin` | ViT condensed/diffuse classifier, 2-channel 160x160 input | `PT_CondenSeq_img_ViT_cls_R1/checkpoint-10000` |
| `hpa/cellfm_seq2img.bin` | CELL-FM virtual-staining generator for HPA, 256x256, 3-channel conditioning (includes the ESM-C 600M encoder) | `pretrain_hpa/cellfm_seq2img/checkpoint-50000` |
| `hpa/vae.bin` | Image VAE at 256x256, the one `hpa/cellfm_seq2img.bin` was trained against | `pretrain_hpa/vae/checkpoint-50000` |
| `hpa/cellfm_img2seq.bin` | CELL-FM image-to-sequence model for HPA, 512x512, 3-channel conditioning (includes the ESM-C 600M encoder) | `pretrain_hpa/cellfm_img2seq/checkpoint-60000` |
| `hpa/vae_512.bin` | Image VAE at 512x512, the one `hpa/cellfm_img2seq.bin` was trained against — a different model from `hpa/vae.bin`, not a rename | `pretrain_hpa/vae_512/checkpoint-50000` |
| `opencell/cellfm_vs.bin` | CELL-FM virtual-staining generator for OpenCell, 256x256, single nucleus conditioning channel, fine-tuned (includes the ESM-C 600M encoder) | `finetune_opencell/cellfm_vs/checkpoint-100000` |
| `opencell/vae.bin` | Image VAE at 256x256, the one `opencell/cellfm_vs.bin` was trained against — OpenCell-finetuned, so not interchangeable with `hpa/vae.bin` despite the matching shape | `finetune_opencell/vae/checkpoint-50000` |
| `opencell/vs_anchor_nucleus.npy` | The nucleus every OpenCell generation is conditioned on: `(1, 256, 256)` float32 in [-1, 1]. Gene ATG7, crop `CID001813_FID00035838_proj_11` — bit-for-bit the conditioning channel of the published offline run | built |
| `opencell/vs_genes.csv` | OpenCell's 1,311 genes: name, protein name, UniProt accession, Ensembl id, localization annotation and sequence. The metadata table minus its image paths | built |
| `opencell/vs_reference_cells.npz` | 17 genes spanning distinct compartments, each with its own nucleus and its real protein image, float16 — the ground truth the notebook's generations are shown against | built |
| `hpa/anchor_cell.npy` | The fixed cell every NLS-screening image is conditioned on: `(3, 256, 256)` float32 in [-1, 1], channels nucleus, ER, microtubules. HPA gene H3C13, cell crop `1194_B2_2_4` | built |
| `hpa/anchor_masks.npz` | Two 256x256 boolean masks over that cell, `nucleus` and `cell`; cytoplasm is `cell & ~nucleus` | built |
| `hpa/pls_anchor_nls.npz` | The cell PLS generation conditions on for nuclear signals: `cell` `(3, 512, 512)` nucleus/ER/microtubules and `protein` `(1, 512, 512)`, float32 in [-1, 1]. HPA gene PPM1G (Nucleoplasm), crop `392_B9_1_11` | built |
| `hpa/pls_anchor_nes.npz` | The same for export signals. HPA gene DIAPH1 (Cytosol, Plasma membrane), crop `1608_B3_1_1` | built |
| `hpa/proteome_aa_counts.json` | Residue counts over the 12,894 HPA proteins (7,940,784 residues), the proteome baseline the frequency analysis compares against | built |
| `hpa/pls_reference_nls.csv` | The 320 published NLS signals: 20 independent draws at each of 16 tail lengths, 10-25 aa | `output/hpa/pls_generation/nls` |
| `hpa/pls_reference_nes.csv` | The same 320 for export signals | `output/hpa/pls_generation/nes` |

Hyperparameters for the CondenSeq models are set in `pipeline.py` in the Space and mirror
`scripts/cell_fm_cs/evaluate_seq2img.sh` and
`scripts/vit_cls_condenseq_img/pretrain.sh` in the CELL-FM repository. The HPA
hyperparameters are spelled out in `notebooks/nls_screening.ipynb` and mirror
`scripts/cell_fm/evaluate_virtual_staining_hpa_dict.sh`. The img2seq pair mirrors
`scripts_local/cell_fm/evaluate_img2seq_hpa_v2.sh`: 512 px, `sample_size` 128,
`encoder_patch_size` 8, `img_generator_patch_size` 4, 8 attention heads. The OpenCell pair
mirrors `scripts/cell_fm/evaluate_virtual_staining_opencell.sh`: 256 px, `sample_size` 64,
`encoder_patch_size` 4, `img_generator_patch_size` 2, 18 attention heads, `cell_image` `nucl`
— spelled out in `notebooks/opencell_vs.ipynb`. That single conditioning channel is the one
architectural difference a caller can see: the HPA generator takes three.

Each generator must be loaded with the VAE it was trained against — pairing
`cellfm_img2seq.bin` with the 256 px `vae.bin` gives a latent-size mismatch.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BoHuangLab/CELL-FM")
    ap.add_argument("--space-url", default="https://huggingface.co/spaces/BoHuangLab/CELL-FM")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="",
                    help="upload only the targets whose name contains this substring, e.g. "
                         "--only pls_reference. The card is rewritten either way, so a "
                         "partial upload still leaves the repo describing all of it.")
    args = ap.parse_args()

    sources = {k: v for k, v in DEFAULT_SOURCES.items() if args.only in k}
    if not sources:
        raise SystemExit(f"--only {args.only!r} matched none of: "
                         + ", ".join(DEFAULT_SOURCES))

    missing = [p for p in sources.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing checkpoint(s):\n  " + "\n  ".join(missing))

    total = sum(os.path.getsize(p) for p in sources.values())
    for name, path in sources.items():
        print(f"  {name:22s} {os.path.getsize(path)/1e9:5.2f} GB  <- {path}")
    print(f"  {'total':22s} {total/1e9:5.2f} GB  -> {args.repo}")

    if args.dry_run:
        print("\ndry run, nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    for name, path in sources.items():
        print(f"uploading {name} ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
        )

    api.upload_file(
        path_or_fileobj=CARD.format(space_url=args.space_url).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"\ndone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
