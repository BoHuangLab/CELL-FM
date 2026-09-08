"""Push the three CELL-FM CondenSeq checkpoints to a HF model repo.

NOTE: pass a token via the HF_TOKEN environment variable; never write one into this
file or beside it.

The Space pulls its weights from that repo at startup rather than carrying ~2.1 GB
of LFS itself. Run this once from the cluster:

    python notebooks/tools/upload_weights.py --repo BoHuangLab/CELL-FM

The hpa/ assets come from build_hpa_assets.py, which must run first.

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
    "hpa/anchor_cell.npy": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "anchor_cell.npy"),
    "hpa/anchor_masks.npz": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "anchor_masks.npz"),
}

CARD = """---
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
| `hpa/vae.bin` | Image VAE, 256x256, 3 down blocks, 4 latent channels | `pretrain_hpa/vae/checkpoint-50000` |
| `hpa/anchor_cell.npy` | The fixed cell every NLS-screening image is conditioned on: `(3, 256, 256)` float32 in [-1, 1], channels nucleus, ER, microtubules. HPA gene H3C13, cell crop `1194_B2_2_4` | built |
| `hpa/anchor_masks.npz` | Two 256x256 boolean masks over that cell, `nucleus` and `cell`; cytoplasm is `cell & ~nucleus` | built |

Hyperparameters for the CondenSeq models are set in `pipeline.py` in the Space and mirror
`scripts/cell_fm_cs/evaluate_seq2img.sh` and
`scripts/vit_cls_condenseq_img/pretrain.sh` in the CELL-FM repository. The HPA
hyperparameters are spelled out in `notebooks/nls_screening.ipynb` and mirror
`scripts/cell_fm/evaluate_virtual_staining_hpa_dict.sh`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BoHuangLab/CELL-FM")
    ap.add_argument("--space-url", default="https://huggingface.co/spaces/BoHuangLab/CELL-FM")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = [p for p in DEFAULT_SOURCES.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing checkpoint(s):\n  " + "\n  ".join(missing))

    total = sum(os.path.getsize(p) for p in DEFAULT_SOURCES.values())
    for name, path in DEFAULT_SOURCES.items():
        print(f"  {name:22s} {os.path.getsize(path)/1e9:5.2f} GB  <- {path}")
    print(f"  {'total':22s} {total/1e9:5.2f} GB  -> {args.repo}")

    if args.dry_run:
        print("\ndry run, nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    for name, path in DEFAULT_SOURCES.items():
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
