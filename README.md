# GridSFM: A Foundation Model for Solving AC Optimal Power Flow

## About this repository

This repository contains all the code for the paper **"GridSFM: A Foundation Model for
Solving AC Optimal Power Flow"**.

## Pretrained Resources

We release both the dataset and models trained in this work. Please see:

### Datasets: https://huggingface.co/datasets/lukebhan/gridsfm

Solved AC-OPF scenarios for four grids spanning two orders of magnitude:
`case500_goc` (500 buses), `tx2k` (2,751), `case6470_rte` (6,470) and `activsg10k`
(10,000).

### Models: https://huggingface.co/lukebhan/gridsfm

All finetuned models developed in this work. 

### Upstream GridSFM

This work utilizes a backbone of a pretrained model from the whitepaper release of GridSFM. 
Here are the 
relevant resources.

- **Source:** https://github.com/microsoft/gridSFM
- **Released backbones:** https://huggingface.co/microsoft/GridSFM_Open
- **US power-grid dataset:** https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid
- 
## Getting Started

### 1. Clone and create an environment

```bash
git clone https://github.com/lukebhan/gridsfm.git
cd gridsfm

python -m venv gridsfm_env
source gridsfm_env/bin/activate      # on Windows: gridsfm_env\Scripts\activate
python -m pip install --upgrade pip
```

Conda works equally well:

```bash
conda create -n gridsfm python=3.12 && conda activate gridsfm
```

### 2. Install PyTorch first, matched to your driver

PyTorch and PyTorch Geometric must agree with each other and with your CUDA version, so
install them **before** the rest. Check your driver with `nvidia-smi`, then take the
matching command from [pytorch.org](https://pytorch.org/get-started/locally/). The
released runs used:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
```

CPU-only works for `validate_config.py` and for generating datasets, but not for training.

Verify before continuing. If this prints `False`, fix the PyTorch install rather than
pushing on:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.8.0+cu128 True
```

### 3. Install the remaining Python dependencies

```bash
pip install -r requirements.txt
```

That covers `numpy`, `scipy`, `matplotlib`, `PyYAML` and `huggingface_hub`. It also lists
`torch` and `torch_geometric`, which pip will leave alone if step 2 already satisfied
them.

Confirm the whole stack imports:

```bash
python -c "import torch, torch_geometric, numpy, scipy, matplotlib, yaml, huggingface_hub; print('ok')"
PYTHONPATH=model python -c "import gridsfm; print('gridsfm', gridsfm.__version__)"
```

### 4. Fetch the datasets and checkpoints

Neither is in git. They are published on HuggingFace and are ~29 GB together. Log in
once, then pull whichever you need:

```bash
hf auth login

# datasets (~25 GB) -> generate_finetune_dataset/data/<grid_id>/
hf download lukebhan/gridsfm --repo-type dataset \
    --local-dir generate_finetune_dataset/data

# checkpoints (~4.3 GB) -> finetune_model/checkpoints/
hf download lukebhan/gridsfm --local-dir finetune_model/checkpoints
```

To take one grid instead of all four, filter with `--include`:

```bash
hf download lukebhan/gridsfm --repo-type dataset \
    --include "case500_goc/*" --local-dir generate_finetune_dataset/data
hf download lukebhan/gridsfm \
    --include "base/*" --include "case500_goc/*" --local-dir finetune_model/checkpoints
```

`case500_goc` is the one to start with: 0.72 GB of data and 6 GB of VRAM, against 11 GB
and 30 GB for `activsg10k`. Installing `hf_xet` (`pip install hf_xet`) makes these
downloads chunked and resumable, which matters on the 8.4 GB files.

### 5. Julia, only if you will regenerate datasets

Skip this unless you intend to run the generator. Install Julia from
[julialang.org](https://julialang.org/downloads/), where `juliaup` is the easiest route,
then instantiate the pinned solver environment:

```bash
julia --project=generate_finetune_dataset/src -e 'using Pkg; Pkg.instantiate()'
```

This resolves `Manifest.toml`, which pins the exact versions the released datasets were
produced with: Ipopt, JuMP, PowerModels, JSON3, Polynomials, OrderedCollections, SHA.
First run compiles them and takes several minutes.

```bash
julia --project=generate_finetune_dataset/src -e 'using PowerModels, Ipopt; println("julia ok")'
```

The generator looks for the binary at `~/.local/bin/julia`. If yours is elsewhere, pass
`--julia /path/to/julia` to `generate_finetune_dataset.py`.

## Quick Start

Simple example on how to finetune your own model on the 500 bus grid.

```bash
cd finetune_model

# check the config resolves and every path exists (no GPU, seconds)
python scripts/validate_config.py --config config/case500_goc_finetune.yaml

# build the prepared-case cache once per dataset
python scripts/prep_cache.py --config config/case500_goc_finetune.yaml

# fine-tune on 100 training cases
python scripts/finetune.py --config config/case500_goc_finetune.yaml \
    --train_subset 100 --run_label n0100 --device cuda
```

## Training from Scratch

If you want to regenerate the datasets rather than download them, run the generator
first. Each grid targets 2,600 feasible scenarios, and the cost is dominated by the
*infeasible* solves, which run to the iteration cap before failing.

```bash
cd generate_finetune_dataset

# validate every config against its grid (no solving)
python scripts/validate_config.py

# ~1 min, proves the whole path works
python scripts/generate_finetune_dataset.py --config config/tx2k.yaml \
    --total_num_feasible 5 --num_workers 5 --dry_run

# the full run
python scripts/generate_finetune_dataset.py --config config/tx2k.yaml \
    --num_workers 196 --total_num_feasible 2600
```

Then train:

```bash
cd ../finetune_model
python scripts/prep_cache.py --config config/tx2k_finetune.yaml
bash scripts/train_grid.sh tx2k --go
```

The released checkpoints and how to drive the harness are documented in
[`finetune_model/README.md`](finetune_model/README.md). The perturbation modes are in
[`generate_finetune_dataset/README.md`](generate_finetune_dataset/README.md).

## Repository layout

```
gridsfm/
├── generate_finetune_dataset/   the dataset pipeline
│   ├── config/                  one YAML per grid: seeds, perturbation ranges, splits
│   ├── topologies/              Matpower .m source grids
│   ├── src/                     source code
│   ├── scripts/                 runnable dataset scripts
│   └── data/                    generated datasets land here
│
├── finetune_model/              the fine-tuning harness
│   ├── config/                  defaults.yaml + one file per (grid, recipe)
│   ├── src/                     source code
│   ├── scripts/                 runnable finetuning scripts
│   ├── checkpoints/             base/ and <grid_id>/<run_label>/
│   ├── cache/                   prepared-case cache, machine-local
│   ├── results/                 figures, history, metrics per run
│   └── logs/                    training logs
│
├── model/                       the `gridsfm` package, vendored from microsoft/gridSFM
├── requirements.txt             Python dependencies for everything above
└── LICENSE
```

Each of the three top-level directories has its own README with the detail:
[`generate_finetune_dataset/`](generate_finetune_dataset/README.md),
[`finetune_model/`](finetune_model/README.md), [`model/`](model/README.md).

## Questions or issues

Please open a GitHub issue for the repository, or contact
[lbhan@ucsd.edu](mailto:lbhan@ucsd.edu).

## Licensing

This work is released under the MIT License.

[`model/`](model/) is vendored from [microsoft/gridSFM](https://github.com/microsoft/gridSFM),
which is also MIT-licensed, copyright Microsoft Corporation. Its notice is carried in
[`model/README.md`](model/README.md). The released backbone weights
(`microsoft/GridSFM_Open`) and the `microsoft/GridSFM_US_power_grid` dataset carry their
own terms. See their HuggingFace repositories.

## Citation

If you use this code, the datasets or the checkpoints, please cite:

```bibtex
@article{gridsfm2026,
  title   = {GridSFM: A Foundation Model for Solving AC Optimal Power Flow},
  author  = {},
  journal = {},
  year    = {},
  url     = {https://github.com/lukebhan/gridsfm}
}
```
