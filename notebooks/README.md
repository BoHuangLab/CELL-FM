# Notebooks

Colab notebooks, one per **application** of CELL-FM. A notebook is the version you can
open, edit and run against your own sequences; the
[hosted app](https://huggingface.co/spaces/BoHuangLab/CELL-FM) is the version with no
setup, and currently covers the first of them.

| Notebook | | |
|---|---|---|
| [`condensate_titration.ipynb`](condensate_titration.ipynb) | Sequence to condensate titration curve, AUC and AAC | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/condensate_titration.ipynb) |
| [`nls_screening.ipynb`](nls_screening.ipynb) | Slide a window along a sequence and score each fragment for nuclear localisation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/nls_screening.ipynb) |
| [`pls_generation.ipynb`](pls_generation.ipynb) | Run the model backwards: design localization signals from a cell that already shows the localization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/pls_generation.ipynb) |
| [`opencell_vs.ipynb`](opencell_vs.ipynb) | Name a protein and see it virtually stained into a cell, with thirteen known proteins as the check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/opencell_vs.ipynb) |

All four notebooks pull their model code from the public Space with `snapshot_download`, so
they cannot drift from each other or from the hosted app on the thing they share — the model
implementation. None vendors any model code of its own.

What is *not* shared is the configuration. The Space currently serves one application,
condensate titration, so there is no hosted counterpart to pin the other three against;
each transcribes its hyperparameters from the shell script that produced its checkpoint, and
says which one in the cell that builds the config.

`opencell_vs.ipynb` is the only one built on the OpenCell fine-tune, and the only one whose
conditioning image has a **single** channel: `cell_image='nucl'` against the HPA notebooks'
`nucl,er,mt`. Handing that generator three channels fails inside a `Conv2d` rather than
degrading quietly. It is also the only notebook whose input is a protein *identity* rather
than a sequence, so it ships OpenCell's 1,311-gene table as an asset and falls back to
UniProt for anything outside it.

## Dependency pins, and why

Colab ships torch, numpy, pandas and matplotlib; the notebook installs the rest. Two
choices are deliberate:

- **`esm==3.2.1.post1` with `--no-deps`.** Its metadata requires `torchtext`, which has no
  wheel past Python 3.11 and would pull torch backwards. Nothing on the ESM-C code path
  imports it; the notebook installs the packages the import closure actually needs.
- **`transformers<4.48.2`.** `esm`'s own bound, and it is behavioural rather than cosmetic.
  4.47 replaced the special-token properties on `PreTrainedTokenizer` with a
  `_special_tokens_map` served through `__getattr__`; a tokenizer that has not adapted
  returns `mask_token = None`, and generation then dies in `esm/utils/encoding.py` with
  `replace() argument 2 must be str, not None` — at generation time, not at import.
  `esm 3.2` adapted to it, and the bound is where that adaptation stops being tested.
- **No `flash-attn`.** Without it ESM-C falls back to its pure-torch rotary embedding,
  verified to give identical results, and skips a fragile CUDA build.
- **`USE_TF=0` and `USE_FLAX=0`**, set before `transformers` is first imported.
  `transformers` probes for a TensorFlow backend at import time and loads it if present;
  Colab ships TensorFlow, nothing here uses it, and the import costs seconds for nothing.

### What used to be here

The notebook pinned `esm==3.1.4`, and that one pin forced four more. `esm 3.1.4` requires
`biotite==0.41.2`, biotite 0.41 requires NumPy 1.x, and biotite sits on the ESM-C import
path — so the whole stack came down to `numpy<2`, dragging `pandas<3` and
`tifffile<2026.4` with it. On Colab's Python 3.13 neither NumPy 1.26 nor biotite 0.41 has
a wheel, so pip built both from source, and the NumPy downgrade collided with every
preinstalled Colab package that expects NumPy 2 — around fifteen lines of red on an
install that had in fact succeeded.

`esm 3.2` moved to `biotite>=1.0`, which is NumPy 2 clean, so all of that goes away: no
source builds, NumPy stays where Colab put it, and the only remaining complaint is that
`transformers < 4.48.2` wants `huggingface-hub < 1.0` while Colab's preinstalled gradio
wants a newer one. Nothing here imports gradio.

The upgrade is safe for the checkpoints: `esmc.py`, `transformer_stack.py`, `blocks.py`,
`attention.py` and `regression_head.py` are byte-identical between 3.1.4 and 3.2.1, so the
tensor names in the published weights are unchanged. That matters because CELL-FM's loader
filters a checkpoint down to keys that match by name and shape and silently drops the
rest — a renamed ESM parameter would have loaded a randomly initialised encoder rather
than failing.

The NumPy downgrade also makes pip list every preinstalled Colab package that wants NumPy
2 — opencv, jax, shap and so on. None of those is imported here, and a wheel compiled
against NumPy 2 still runs under 1.x, so they are noise. The one `esm` bound left
unenforced is `torchtext`, which nothing on the ESM-C path imports and which has no wheel
for this Python. The install cell explains the messages, then checks the four pins that
actually matter rather than leaving you to read pip's output.
