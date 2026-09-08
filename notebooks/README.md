# Notebooks

Colab notebooks, one per **application** of CELL-FM — the same split the
[hosted app](https://huggingface.co/spaces/BoHuangLab/CELL-FM) uses for its tabs. A
notebook is the version you can open, edit and run against your own sequences; the Space
is the version with no setup.

| Notebook | | |
|---|---|---|
| [`condensate_titration.ipynb`](condensate_titration.ipynb) | Sequence to condensate titration curve, AUC and AAC | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/condensate_titration.ipynb) |

## Condensate titration

Give it a 66-residue IDP sequence; it generates one microscopy image per protein-intensity
level over a log-spaced concentration ladder, classifies each image condensed or diffuse,
smooths the calls into a titration curve, and integrates the curve into **AUC**
(condensation propensity) and **AAC** (condensation lost to reentrant dissolution). It
walks through all three stages separately rather than calling `pipeline.run`, so each can
be inspected on its own: the generated images, the per-image condensed/diffuse calls, and
the curve they smooth into. The sequence goes in through a text box that validates as you
type, defaulting to the NUP98 IDP.

Runtime, measured per sequence:

| Preset | Ladder | ODE steps | A40 | rough T4 |
|---|---|---|---|---|
| `quick` (default) | 192 | 50 | ~30 s | ~2 min |
| `standard` | 256 | 100 | ~40 s | ~3 min |
| `paper` | 512 | 100 | ~80 s | ~6 min |

Setup downloads ~4.4 GB once: 2.1 GB of CELL-FM checkpoints from
[`BoHuangLab/CELL-FM`](https://huggingface.co/BoHuangLab/CELL-FM), and 2.3 GB for the
ESM-C 600M encoder the generator is built around, which the `esm` package fetches when the
model is constructed. Everything is public — no HF token needed.

## Where the code comes from

The notebook does not vendor the pipeline. It `snapshot_download`s the public Space
(`repo_type="space"`, ~400 kB) and imports `pipeline.py` and `metrics.py` from it, so the
notebook and the app cannot drift apart: same model configs, same fixed reference nucleus,
same metric definitions, same assets. Model code comes with that snapshot as an
import-closed subset of `cell_fm/`.

If the Space ever moves, the two constants to change are `CODE` in the fetch cell and
`pipeline.MODEL_REPO` (overridable with `CELLFM_MODEL_REPO`).

## Reading the numbers

Absolute AUC moves with the seed, the GPU and the preset — at `paper` the NUP98 wild type
lands around 0.49, with a spread of roughly ±0.02 between seeds, and c<sub>sat</sub>, being
a single threshold crossing, moves further — so **comparisons between sequences are the
signal, not the third decimal place**.

If you loop the notebook over several sequences, hold `SEED` fixed. Paired that way the
shared sampling noise largely cancels, and a difference well under the run-to-run spread
stays resolvable; taken across different seeds, GPUs or presets it means nothing.

## Dependency pins, and why

Colab ships torch, pandas and matplotlib; the notebook installs the rest. Three choices
are deliberate:

- **`esm==3.1.4` with `--no-deps`.** Its metadata requires `torchtext`, which has no wheel
  past Python 3.11 and would pull torch backwards. Nothing on the ESM-C code path imports
  it; the notebook installs the packages the import closure actually needs.
- **`numpy<2`.** `esm` pins `biotite==0.41.2`, which requires NumPy 1.x, and biotite sits
  on the ESM-C import path. Colab ships NumPy 2, so it has to come down; wheels built
  against NumPy 2 keep working under 1.x, so nothing else breaks. The install cell
  restarts the kernel by itself if NumPy changed underneath it.
- **No `flash-attn`.** Without it ESM-C falls back to its pure-torch rotary embedding,
  verified to give identical results, and skips a fragile CUDA build.

Neither `biotite==0.41.2` nor NumPy 1.x ships wheels past **Python 3.12**, and Colab is
now on 3.13, so pip builds both from source. That works — it costs a few minutes on the
install cell. The notebook says so up front rather than letting a long silent build look
like a hang.

The NumPy downgrade also makes pip list every preinstalled Colab package that wants NumPy
2 (opencv, jax, shap and so on), and repeat that `esm` asked for `torchtext` and
`transformers<4.47`. None of that is a failure: nothing on this path imports those
packages, and every symbol the pipeline takes from `transformers` still exists in 5.x. The
install cell explains the messages and then checks the two pins that actually matter.
