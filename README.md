<div align="center">
  <a href="https://www.microsoft.com/en-us/research/">
    <img align="left" src="media/microsoft.png" width="258" height="55" alt="Microsoft Research">
  </a>
  <a href="https://ucsd.edu/">
    <img src="media/ucsd.png" width="281" height="55" alt="UC San Diego">
  </a>
  <a href="https://www.washington.edu/">
    <img align="right" src="media/uw.png" width="255" height="55" alt="University of Washington">
  </a>
</div>

<br> <br>

# Fine-Tuning Grid Foundation Models for AC-OPF Warm Starts

<div align="center">
 <a href="#"><img alt="Perturbation and dataset pipeline" src="media/pipeline.png" width="100%"/></a>
</div>

<div align="center">
 <a href="#"><img alt="Data-scaling curves across four grids" src="media/scaling.png" width="100%"/></a>
</div>

## About this repository

This repository contains the implementation used to adapt the released **GridSFM** grid
foundation model into a *control model*: a network that predicts generator dispatch and
voltage setpoints accurately enough to **warm-start an AC-OPF solve**. The codebase
includes:

- A dataset generator that perturbs a base grid topology, solves each scenario to
  optimality with Ipopt, and publishes fixed `train`/`val`/`test` splits.
- A fine-tuning harness built around a self-supervised *elastic* AC-OPF objective
  evaluated at the power-flow-closed operating point, with a supervised anchor.
- The four generated datasets (2,600 solved scenarios per grid) and every fine-tuned
  checkpoint, including a from-scratch control arm.
- A trimmed copy of the `gridsfm` inference package, so the repository runs standalone.

```
generate_finetune_dataset/   perturb -> solve (Ipopt/PowerModels) -> publish JSONL splits
finetune_model/              fine-tune the released backbone into the control model
model/                       vendored from microsoft/gridSFM (MIT) -- see model/README.md
```

Four grids are covered, spanning two orders of magnitude in size:

| grid | buses | active gens | scenarios | yield | dataset |
|---|---:|---:|---:|---:|---:|
| `case500_goc` | 500 | 171 | 2,600 | 96.0% | 0.72 GB |
| `tx2k` | 2,751 | 736 | 2,600 | 64.6% | 4.5 GB |
| `case6470_rte` | 6,470 | 761 | 2,600 | 58.9% | 8.6 GB |
| `activsg10k` | 10,000 | 1,937 | 2,600 | 88.0% | 11 GB |

*yield* is the fraction of perturbed scenarios that Ipopt proved feasible; only feasible
cases are published.

## Installation

```bash
git clone https://github.com/lukebhan/gridsfm.git
cd gridsfm
python -m venv gridsfm_env
source gridsfm_env/bin/activate   # on Windows: gridsfm_env\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `torch`, `torch_geometric`, `numpy`, `scipy`, `matplotlib`, `PyYAML`,
`huggingface_hub`. Install the PyTorch build matching your CUDA driver from
[pytorch.org](https://pytorch.org) first.

Generating new datasets additionally needs **Julia ≥ 1.12** with Ipopt, PowerModels,
JuMP, JSON3 and SHA. Training and evaluation do not — the published datasets are already
JSONL.

```bash
julia --project=generate_finetune_dataset/src -e 'using Pkg; Pkg.instantiate()'
```

## Pretrained Resources

Checkpoints and datasets for this repository are organized around the local
`finetune_model/checkpoints/` and `generate_finetune_dataset/data/` folders.

- **Models:** https://huggingface.co/lukebhan/gridsfm
- **Datasets:** https://huggingface.co/datasets/lukebhan/gridsfm

If you download or regenerate these separately, place checkpoints under
`finetune_model/checkpoints/<grid_id>/<run_label>/` and datasets under
`generate_finetune_dataset/data/<grid_id>/`.

### Upstream GridSFM

This work builds on the original GridSFM release from Microsoft Research. **Everything in
[`model/`](model/) is their code, vendored unmodified apart from deletions** so this
repository runs standalone — see [`model/README.md`](model/README.md) for exactly what was
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
first. Each grid targets 2,600 feasible scenarios; the cost is dominated by the
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
[`finetune_model/README.md`](finetune_model/README.md); the perturbation modes in
[`generate_finetune_dataset/README.md`](generate_finetune_dataset/README.md).

## Questions or issues

If you have questions or run into issues, please open a GitHub issue for the repository.

## Licensing

This work is released under the MIT License.

[`model/`](model/) is vendored from [microsoft/gridSFM](https://github.com/microsoft/gridSFM),
which is also MIT-licensed, copyright Microsoft Corporation; its notice is carried in
[`model/README.md`](model/README.md). The released backbone weights
(`microsoft/GridSFM_Open`) and the `microsoft/GridSFM_US_power_grid` dataset carry their
own terms — see their HuggingFace repositories.
