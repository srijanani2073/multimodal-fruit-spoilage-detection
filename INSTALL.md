# Installation guide

## Requirements

| | |
|---|---|
| Python | 3.10 or newer (the experiments used 3.14; the tests also pass on 3.13) |
| OS | macOS, Linux or Windows |
| Hardware | Apple Silicon (MPS), an NVIDIA GPU (CUDA) or CPU. Training on CPU works but is slow. |
| Disk | Space for TR-6, plus about 130 MB per trained seed |

## 1. Get the code

```bash
git clone https://github.com/srijanani2073/multimodal-fruit-spoilage-detection.git
cd multimodal-fruit-spoilage-detection
```

## 2. Create an environment

With `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

or with conda:

```bash
conda create -n trimodal python=3.12 -y
conda activate trimodal
```

## 3. Install the dependencies

For the scripts and tests:

```bash
pip install -r requirements.txt
pip install -e .                   # optional: makes `import trimodal` work from anywhere
```

To run the notebooks as well:

```bash
pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name trimodal
```

On a machine with an NVIDIA GPU, install the CUDA build of PyTorch first, following
<https://pytorch.org/get-started/locally/>, then run the lines above. On Apple Silicon the
default wheel already includes MPS support.

## 4. Check the installation (no dataset needed)

```bash
pytest -q tests/
python scripts/check_setup.py --synthetic --no-pretrained
```

Both run on a small generated stand-in for TR-6 and should finish in a few minutes on CPU.
`check_setup.py` ends with `All checks passed.`

## 5. Get the dataset

TR-6 is published separately and is not stored in this repository:

> Unnikrishnan, D., Deepak, K., P T S, Y. A., and T, B. (2026). *Multimodal perishable fruits and
> vegetables dataset.* Data in Brief. <https://doi.org/10.1016/j.dib.2026.112545> —
> files: <https://doi.org/10.6084/m9.figshare.30783827>

Download and unpack it so the folder contains `Classified/` and `Normal/`:

```
TR-6/
├── Classified/
│   ├── Banana/
│   │   ├── Not_spoiled/
│   │   │   ├── sRGB_images/          20250911_093012.jpg ...
│   │   │   └── IR_fusion_images/     20250911_093015_28.4C_33.1C.jpg ...
│   │   └── Spoiled/
│   └── Carrot/ Guava/ Indian_Gooseberry/ Mango/ Tomato/
└── Normal/
```

Then tell the code where it is, in one of three ways:

1. Put it at `data/TR-6` inside the repository (the default; `data/` is git-ignored), or
2. set an environment variable once per shell:
   ```bash
   export TR6_ROOT=~/Desktop/Thesis/TR-6
   ```
3. or pass `--data-root ~/Desktop/Thesis/TR-6` to any script.

The train / validation / test split is the committed manifest
`Code/Manifest/manifests/master_manifest.csv`; override it with `TR6_MANIFEST` or `--manifest`.
`TRIMODAL_OUT` or `--out` changes where outputs go (default `outputs/`).

The notebooks predate the scripts and set their paths in their first code cell (`ROOT` for TR-6;
`MANIFEST`, `GAS_NORM` and `RUNS` under `~/Desktop/convscript/`, which was the repository's location
on the original machine). Edit those lines to match your folders; Jupyter runs each notebook from
`Code/`, so inside this repository `RUNS = Path('../runs')` and `MANIFEST = Path('Manifest/manifests/master_manifest.csv')`.

## 6. Check the dataset

```bash
python scripts/check_setup.py
```

Expected split sizes: train 12 trajectories / 73 days, val 12 / 21, test 11 / 18. ImageNet weights
for ConvNeXt-Tiny and EfficientNet-B0 download on first use (about 130 MB, cached by torchvision).

Next: [scripts/README.md](scripts/README.md) for training and evaluation.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TR-6 images not found` | `--data-root` must point at the folder that *contains* `Classified/`. |
| Split sizes differ from the numbers above | The manifest and image folders come from different copies of TR-6; use the committed manifest. |
| `MPS backend out of memory` | Lower `--batch-size` to 2, or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. |
| An operation is not implemented for MPS | `export PYTORCH_ENABLE_MPS_FALLBACK=1`, or run with `--device cpu`. |
| Weight download fails (offline machine) | Copy the two `.pth` files into `~/.cache/torch/hub/checkpoints/`, or use `--no-pretrained` for testing only. |
| `ModuleNotFoundError: trimodal` in your own code | Run `pip install -e .` from the repository root. The scripts do not need it. |
