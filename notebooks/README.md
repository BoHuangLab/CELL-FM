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

All four notebooks fetch their model code by shallow-cloning this repository, and print the
commit they resolved, so a result can always be traced back to the code that produced it.
None vendors any model code of its own. Weights come separately, from
[the weights repo](https://huggingface.co/BoHuangLab/CELL-FM) — code from the repository
that defines it, weights from the repo built to hold them.

They used to clone the published Space instead. That worked while the Space happened to
carry everything they needed, and stopped the moment one of them needed
`cell_fm/models/vit_cls`, which the Space does not ship. Cloning the repository removes the
whole class of problem: anything in CELL-FM is reachable from a notebook. It is also why
the CondenSeq app's `pipeline.py` and `metrics.py` now live in `cell_fm/apps/condenseq/`
rather than only inside the Space directory, which is not tracked.

The notebooks track `master`, so a breaking change reaches them as soon as it lands — the
printed commit is what makes that debuggable rather than mysterious. The Space is a separate
deployment on its own cadence and can lag behind; it is no longer what the notebooks read.

## Dependency pins, and why

Colab ships torch, numpy, pandas and matplotlib; the notebook installs the rest. Two
choices are deliberate:

- **`esm==3.2.1.post1` with `--no-deps`.** Its metadata requires `torchtext`, which has no
  wheel past Python 3.11 and would pull torch backwards. Nothing on the ESM-C code path
  imports it; the notebook installs the packages the import closure actually needs.
- **`transformers>=4.47,<4.48.2`.** A range, and both ends matter for different reasons.
  4.47 is where `transformers` stopped assigning special tokens with `setattr` and began
  serving them through `__getattr__`. `esm 3.2`'s `EsmSequenceTokenizer` builds
  `cls_token`, `mask_token` and the rest as read-only properties on top of that, so on
  4.46 or older it cannot construct at all: `property 'cls_token' of
  'EsmSequenceTokenizer' object has no setter`, raised while the model is being built,
  before any check in the notebook runs. Above 4.48.2 — `esm`'s own declared bound — the
  plumbing moves again and `mask_token` comes back `None`, which kills generation later,
  inside `esm/utils/encoding.py`, with `replace() argument 2 must be str, not None`.

  The lower bound is not redundant. Colab preinstalls a `transformers` that already
  satisfies `<4.48.2`, and pip leaves a satisfied requirement alone rather than upgrading
  it — so with only the upper bound the pin quietly did nothing on a fresh runtime, and
  whether the notebook worked depended on which version Colab happened to ship.
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
