# 1. Installation

## Option A — Windows, one click (recommended for a single workstation)

Nothing has to be installed by hand. `benchmark/RUN.bat` installs a private Python 3.11 and the pinned packages into
`benchmark/runtime/` the first time it runs (about 10–20 minutes, ~3 GB). Nothing is installed system-wide.

Requirements:

* Windows 10/11, an NVIDIA GPU with a recent driver (`nvidia-smi` must work)
* about 150 GB free disk space, ideally on an SSD, and ~15 GB free on `C:` for the model weights cache
* internet access the first time (packages, data, pretrained weights)

## Option B — Linux / any Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Tested versions

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch / torchvision | 2.3.1 / 0.18.1 (CUDA 12.1) |
| timm | 1.0.9 |
| NumPy / pandas | 1.26.4 / 2.2.2 |
| scikit-learn / SciPy | 1.5.1 / 1.13.1 |
| matplotlib / Pillow | 3.9.2 / 10.4.0 |
