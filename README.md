# DINOv2 Multitask Learning for Neuron Images

This repository contains PyTorch training code for jointly classifying
culture-day groups and predicting elastic modulus from single-cell TIFF images.
A DINOv2 ViT-S/14 backbone feeds a day-classification head and a day-aware
regression head. Performance is evaluated with stratified out-of-fold (OOF)
predictions. The dataset is distributed separately and is not included here.

## Repository layout

```text
.
├── data/
│   └── README.md           # expected schema and local layout
├── scripts/
│   └── prepare_data.py     # safe, repeatable archive extraction
├── src/
│   └── train.py            # training and OOF evaluation entry point
├── outputs/                # generated artifacts (ignored by Git)
├── environment.yml
└── requirements.txt
```

## Installation

Python 3.10 is recommended. Create either a Conda environment:

```bash
conda env create -f environment.yml
conda activate neuron-multitask
```

or a virtual environment with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first training run downloads the DINOv2 ViT-S/14 backbone through
`torch.hub`; network access is therefore required once. Subsequent runs use the
local PyTorch Hub cache.

## Reproduce the experiment

Obtain the dataset from its separate archive and use either of these layouts:

```text
data/images/*.tif
data/labels.xlsx
```

or pass external locations directly with `--image-dir` and `--label-file`.
If the downloaded image archive is a ZIP file, prepare it with:

```bash
python scripts/prepare_data.py --archive /path/to/images.zip
python scripts/validate_data.py
```

Then run training from the repository root:

```bash
python src/train.py
```

For a dataset stored entirely outside the repository:

```bash
python scripts/validate_data.py \
  --image-dir /path/to/dataset/images \
  --label-file /path/to/dataset/labels.xlsx
python src/train.py \
  --image-dir /path/to/dataset/images \
  --label-file /path/to/dataset/labels.xlsx
```

The defaults reproduce the supplied experiment configuration: seed 42, five
folds, 50 epochs, batch size 16, 224 × 224 input canvas, and `R > 0.75`.
To inspect or override any setting:

```bash
python src/train.py --help
python src/train.py --epochs 50 --batch-size 16 --device cuda
```

For the final archived experiment, pin DINOv2 to the exact Git commit used:

```bash
python src/train.py --backbone-repo facebookresearch/dinov2:<commit-hash>
```

For a quick pipeline check (not a paper-quality experiment), use fewer epochs
and folds:

```bash
python src/train.py --epochs 1 --folds 2 --device cpu
```

## Outputs

Each run writes to `outputs/` by default:

- `run_config.json`: exact CLI configuration and core software versions;
- `final_model_dinov2_hybrid_fold_*.pth`: per-fold EMA weights;
- `hybrid_multitask_training_history.csv`: epoch-level losses and accuracy;
- `hybrid_multitask_oof_results.csv`: sample-level OOF predictions and errors;
- `oof_dinov2_hybrid_multitask_ensemble.pth`: model states, calibration models,
  fold indices, predictions, and summary metrics.

Large model artifacts are excluded by `.gitignore`. Publish final checkpoints
through a versioned release or archival repository (for example, Zenodo) and
record their checksums and permanent identifier in the associated paper.

## Evaluation protocol

Samples are stratified jointly by grouped culture day and binned modulus when
every stratum has enough observations; otherwise, stratification falls back to
day group. Each validation sample is predicted only by the fold model that did
not train on it. Reported regression metrics include R², MAE, RMSE, MAPE,
median relative error, and Pearson correlation. Classification metrics include
accuracy, balanced accuracy, macro/weighted F1, and a confusion matrix.

The script also reports linear and day-aware post-hoc calibration estimates.
These are generated with an inner cross-validation operation over the raw OOF
predictions. When writing the paper, distinguish these calibrated estimates
from the uncalibrated primary OOF metrics.

## Reproducibility notes

Python, NumPy, and PyTorch are seeded, and deterministic cuDNN behavior is
requested. Exact floating-point agreement can still vary across operating
systems, accelerators, PyTorch releases, and DINOv2 revisions. Archive the
resolved environment (`python -m pip freeze`) and hardware/CUDA details with
the final paper results.

## Citation

If you use this repository, cite the associated article and the archived
software release. Update `CITATION.cff` with the final author list, article DOI,
repository URL, and release version before making the GitHub repository public.

## License and research-use notice

The source code is released under the MIT License. The separately distributed
biological dataset must carry its own data-use license; see `data/README.md`.
This research code is provided for scientific reproducibility and is not
validated for clinical decision-making.
