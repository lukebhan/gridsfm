<div align="center">
  <a href="https://www.microsoft.com/en-us/research/">
    <img align="left" src="media/microsoft.png" width="215" height="46" alt="Microsoft Research">
  </a>
  <a href="https://ucsd.edu/">
    <img src="media/ucsd.png" width="235" height="46" alt="UC San Diego">
  </a>
  <a href="https://www.washington.edu/">
    <img align="right" src="media/uw.png" width="67" height="46" alt="University of Washington">
  </a>
</div>

<br> <br>

# GridSFM: A Foundation Model for Solving AC Optimal Power Flow

## About this repository

This repository contains all the code for the paper **"GridSFM: A Foundation Model for
Solving AC Optimal Power Flow"**.

## Pretrained Resources

Neither is stored in git. Step 4 of Getting Started below downloads them.

### Datasets: https://huggingface.co/datasets/lukebhan/gridsfm

Solved AC-OPF scenarios for four grids spanning two orders of magnitude:
`case500_goc` (500 buses), `tx2k` (2,751), `case6470_rte` (6,470) and `activsg10k`
(10,000). Each holds **2,600 scenarios proven feasible by Ipopt**, published as a fixed
2,000 / 100 / 500 train / val / test split that the training harness reads verbatim and
never re-shuffles.

A scenario is the base grid with several perturbations composed on top of it. Demand is
scaled, generator merit order reshuffled, units tripped, branches derated and voltage
bands tightened, and the result is then solved to optimality. Every case carries the full grid, the optimal
solution, the duals, and metadata recording which perturbations fired and how hard. About
25 GB in total, from 0.72 GB for `case500_goc` to 11 GB for `activsg10k`.

### Models: https://huggingface.co/lukebhan/gridsfm

Seven checkpoints per grid, 28 in all. Six are the **data-scaling series**, the same
fine-tuning recipe at 10, 25, 50, 100, 200 and 500 training cases. The seventh,
`n1000_scratch`, is the **no-pretraining control**: identical harness, random
initialisation, 1,000 cases, which is what shows the pretrained backbone is doing the
work. Also included is `base/gridsfm_open_v2.pt`, the upstream Microsoft backbone every
fine-tune starts from.

Each run directory ships `control_model.pt` (the weights the paper reports), plus
`best_val.pt` and `last.pt`. See
[`finetune_model/README.md`](finetune_model/README.md) for the full breakdown and loading
code. About 4.3 GB in total.

If you regenerate either, place checkpoints under
`finetune_model/checkpoints/<grid_id>/<run_label>/` and datasets under
`generate_finetune_dataset/data/<grid_id>/`.

### Upstream GridSFM

This work builds on the original GridSFM release from Microsoft Research. **Everything in
[`model/`](model/) is their code, vendored unmodified apart from deletions** so this
repository runs standalone. See [`model/README.md`](model/README.md) for exactly what was
kept, what was removed and why. It is MIT-licensed, copyright Microsoft Corporation.

- **Source:** https://github.com/microsoft/gridSFM
- **Released backbones:** https://huggingface.co/microsoft/GridSFM_Open
- **US power-grid dataset:** https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid

Every fine-tune here starts from `GridSFM-open-v2` (15.1 M parameters), which lives at
`finetune_model/checkpoints/base/gridsfm_open_v2.pt`. To pull a release checkpoint
directly:

```python
from gridsfm import load_from_hf
model = load_from_hf("microsoft/GridSFM_Open", device="cuda")
```

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

The published datasets and checkpoints are what the released numbers come from, so the
fastest path is to evaluate rather than retrain. Build the prepared-case cache once per
grid, then fine-tune a point of the data-scaling curve:

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

Additional entry points:

- `bash scripts/train_grid.sh case500_goc --go` drives the whole `n = 10 … 500` series
  across whatever GPUs are idle.
- `bash scripts/train_grid.sh case500_goc --scratch --go` runs the no-pretraining control.
- `python generate_finetune_dataset/scripts/validate_dataset.py --dir data/case500_goc`
  audits a published dataset and replays its generation statistics.

Each run writes figures, per-epoch and per-step history, timings and a resolved config to
`finetune_model/results/<grid_id>/<run_label>/`, and the shipped weights to
`finetune_model/checkpoints/<grid_id>/<run_label>/control_model.pt`.

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

# the real thing
python scripts/generate_finetune_dataset.py --config config/tx2k.yaml \
    --num_workers 196 --total_num_feasible 2600
```

Budget ~1–2 GB of RAM per worker on a 10k-bus grid. On a 256-core host at 196 workers the
released runs took 3.7 min (`case500_goc`) to 87 min (`case6470_rte`) of wall clock,
against 3.3 h to 233 h of aggregate solver time.

Regeneration is exact: every scenario RNG is seeded from
`SHA-256("master | grid_id | mode | scenario_index")`, derived from the grid's *name*
rather than its file path, so the same config reproduces the same scenarios on any
machine or Julia version.

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
├── media/                       README images
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
