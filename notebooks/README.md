# Notebooks

Colab notebooks, one per **application** of CELL-FM — the same split the
[hosted app](https://huggingface.co/spaces/BoHuangLab/CELL-FM) uses for its tabs. A
notebook is the version you can open, edit and run against your own sequences; the Space
is the version with no setup.

| Notebook | | |
|---|---|---|
| [`condensate_titration.ipynb`](condensate_titration.ipynb) | Sequence to condensate titration curve, AUC and AAC | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/condensate_titration.ipynb) |
| [`nls_screening.ipynb`](nls_screening.ipynb) | Slide a window along a sequence and score each fragment for nuclear localisation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoHuangLab/CELL-FM/blob/master/notebooks/nls_screening.ipynb) |

Both notebooks pull their model code from the public Space with `snapshot_download`, so the
notebooks and the hosted app cannot drift: same configs, same fixed conditioning image, same
metric definitions. Neither vendors any model code of its own.

`nls_screening.ipynb` additionally needs the `hpa/` files in
[`BoHuangLab/CELL-FM`](https://huggingface.co/BoHuangLab/CELL-FM) — the virtual-staining
checkpoint and VAE, plus the anchor cell and the two masks drawn on it. Build the assets
with [`tools/build_hpa_assets.py`](tools/build_hpa_assets.py), then push everything with
[`tools/upload_weights.py`](tools/upload_weights.py); until that runs the notebook cannot
load its model.
It is also the slower of the two by a wide margin: 1.66 s per generated image on an A40,
measured, against 0.16 s for condensate.

## Dependency pins, and why

Colab ships torch, pandas and matplotlib; the notebook installs the rest. Every choice
below is deliberate, and several exist only because another one forces them:

- **`esm==3.1.4` with `--no-deps`.** Its metadata requires `torchtext`, which has no wheel
  past Python 3.11 and would pull torch backwards. Nothing on the ESM-C code path imports
  it; the notebook installs the packages the import closure actually needs.
- **`numpy<2`.** `esm` pins `biotite==0.41.2`, which requires NumPy 1.x, and biotite sits
  on the ESM-C import path — `esm.utils.structure.protein_chain` imports
  `biotite.structure.io.npz`, a module biotite deleted in 1.0, so a newer biotite is not a
  way out. Colab ships NumPy 2, so it has to come down. The install cell restarts the
  kernel by itself if NumPy changed underneath it.
- **`pandas<3`.** Forced by the line above. A wheel *compiled* against NumPy 2 still runs
  under 1.x, but pandas 3 calls `np.dtypes.StringDType`, a NumPy 2 Python API that does
  not exist in 1.x, so Colab's pandas dies on import with `module 'numpy.dtypes' has no
  attribute 'StringDType'`. pandas 2.2.3 is the last release that works under the pin and
  ships a wheel for Python 3.13.
- **`tifffile<2026.4`.** Section 6 writes the generated stack as a TIFF, and the current
  `tifffile` declares `numpy>=2.1`. It carries no NumPy 2 API that would actually break
  under the pin, but that declaration is the same signal `pandas` gave before it broke, so
  the notebook takes the last release that accepts NumPy 1.x rather than betting on it.
- **`transformers<4.47`.** `esm`'s own bound, and it is behavioural, not cosmetic. 4.47
  replaced the special-token properties on `PreTrainedTokenizer` with a
  `_special_tokens_map` served through `__getattr__`. `esm 3.1.4`'s `EsmSequenceTokenizer`
  still expects the property, so on anything newer `tokenizer.mask_token` is `None` and
  generation dies in `esm/utils/encoding.py` with `replace() argument 2 must be str, not
  None`. 4.46.3 is the last release with the property; it pulls `tokenizers 0.20` and
  `huggingface_hub 0.36`, both of which have wheels for Python 3.13.
- **`USE_TF=0` and `USE_FLAX=0`**, set before `transformers` is first imported. Not a
  pin, but the same root cause. `transformers` probes for a TensorFlow backend at import
  time and `image_transforms.py` acts on it with `if is_tf_available(): import
  tensorflow`; Colab has TensorFlow, its tflite utils import jax, and `jax/_src/dtypes.py`
  runs `np.dtypes.StringDType()` at module level. Under `numpy<2` that kills the import of
  `transformers.modeling_utils`, and with it `diffusers`, `cell_fm` and the pipeline.
  Nothing here uses either backend, so both are switched off.
- **No `flash-attn`.** Without it ESM-C falls back to its pure-torch rotary embedding,
  verified to give identical results, and skips a fragile CUDA build.

Neither `biotite==0.41.2` nor NumPy 1.x ships wheels past **Python 3.12**, and Colab is
now on 3.13, so pip builds both from source. That works — it costs a few minutes on the
install cell. The notebook says so up front rather than letting a long silent build look
like a hang.

The NumPy downgrade also makes pip list every preinstalled Colab package that wants NumPy
2 — opencv, jax, shap and so on. None of those is imported here, and a wheel compiled
against NumPy 2 still runs under 1.x, so they are noise. The one `esm` bound left
unenforced is `torchtext`, which nothing on the ESM-C path imports and which has no wheel
for this Python. The install cell explains the messages, then checks the four pins that
actually matter rather than leaving you to read pip's output.
