# finetune_model

Fine-tunes the released GridSFM backbone into the **control model** that warm-starts an
AC-OPF solve, using a dataset from [`generate_finetune_dataset`](../generate_finetune_dataset/).

Install and checkpoint downloads are covered in the [root README](../README.md). This file
documents the training harness.

```
config/       defaults.yaml + one file per (grid, recipe), every key repeated explicitly
src/          importable modules -- no argv parsing, no side effects at import
scripts/      the entry points you actually run
checkpoints/  base/ (released backbone) and <grid_id>/<run_label>/ (the shipped weights)
results/      <grid_id>/<run_label>/ figures, history, metrics, timings  (created on run)
logs/         <grid_id>/<run_label>/ training logs                       (created on run)
cache/        <grid_id>/ prepared-case cache      (generated, machine-local, not portable)
```

`results/`, `logs/` and `cache/` ship empty. They are outputs, and the cache is not
portable between machines. Every output is scoped by `<run_label>`, so a data-scaling
sweep writes N independent result sets instead of overwriting one. The cache is **not**
scoped that way, since it is per dataset and shared by every run.

## What is released

```
checkpoints/
├── base/gridsfm_open_v2.pt          the released backbone every fine-tune starts from
└── <grid_id>/<run_label>/           one directory per training run
```

`base/gridsfm_open_v2.pt` is `GridSFM-open-v2`, 15.1 M parameters, 61 MB, the upstream
Microsoft release, not trained here. It is what every config's `base_checkpoint` names,
and what `model.from_scratch: true` ignores.

### The run labels

Seven runs per grid, 28 in total, all four grids identical in structure.

| run label | objective | init | train cases | what it is for |
|---|---|---|---:|---|
| `n0010` | elastic | `gridsfm_open_v2` | 10 | data-scaling curve |
| `n0025` | elastic | `gridsfm_open_v2` | 25 | data-scaling curve |
| `n0050` | elastic | `gridsfm_open_v2` | 50 | data-scaling curve |
| `n0100` | elastic | `gridsfm_open_v2` | 100 | data-scaling curve |
| `n0200` | elastic | `gridsfm_open_v2` | 200 | data-scaling curve |
| `n0500` | elastic | `gridsfm_open_v2` | 500 | data-scaling curve |
| `n1000_scratch` | control | **random** (64×4) | 1000 | the no-pretraining control |

The six `n<nnnn>` runs are the same recipe (`config/<grid>_finetune.yaml`) at increasing
training-set size, and the subsets are **stratified and nested**, with `subset(10)` ⊂
`subset(100)` ⊂ …, so a difference between two points is more data, not different data.
Val and test are never subsetted, so every point is selected on the same val cases and
scored on the same held-out test cases.

`n1000_scratch` (`config/<grid>_scratch1k.yaml`) is what makes the rest mean something: the
same architecture family and the same harness, trained from random init on 1000 cases,
which is *more* data than any fine-tuned point gets. It answers "is the pretrained backbone doing
anything, or would this much data have sufficed on its own?"

### Files inside a run directory

| file | what it is |
|---|---|
| `control_model.pt` | **the deliverable.** A copy of whichever checkpoint `train.select_on` chose. Every shipped run used `select_on: val`, so it is weight-identical to `best_val.pt` (the bytes differ, because it is re-serialised on save). |
| `best_val.pt` | the epoch with the lowest validation score, `val_score_pg_weight × Pg% + val_score_v_weight × V(p.u.)` |
| `last.pt` | the final epoch, whenever training stopped |
| `best_train_loss.pt` | lowest epoch *training* loss. Written by the `control`-mode loop only, so `n1000_scratch` has it and the elastic runs do not |
| `control_model.json` | `control`-mode sidecar: the same summary as `test_metrics.json` (test/val errors, ship epoch, wall time, resolved hyperparameters, config and git sha) |
| `arch.json` | `from_scratch` runs only: `{"hidden_dim": 64, "num_blocks": 4}`, so a loader never has to infer the architecture from tensor shapes. Inferring it wrongly once loaded a 64×4 checkpoint into the 128×8 backbone |

Fine-tuned runs are ~175 MB per directory, three 61 MB copies of the 128×8 backbone.
`n1000_scratch` is ~31 MB, since `from_scratch` builds a 64×4 backbone of 1.96 M
parameters.

### Loading one

Run outputs are **bare `state_dict`s**, not the wrapped release format. Only
`base/gridsfm_open_v2.pt` carries the `{state_dict, metadata}` envelope that
`gridsfm.checkpoint.load_model` expects. So build the architecture first, then load the
weights into it.

```python
import torch
from gridsfm.checkpoint import load_model

# a fine-tuned run: the base checkpoint supplies the 128x8 architecture
model = load_model("checkpoints/base/gridsfm_open_v2.pt", device="cuda")
model.load_state_dict(torch.load("checkpoints/tx2k/n0100/control_model.pt",
                                 weights_only=True, map_location="cuda"))
model.clip_mode = "sigmoid"      # what the configs train with
model.eval()
```

```python
import json, torch
from gridsfm.model import GridTransformerBackbone

# a from-scratch run: arch.json supplies the 64x4 architecture
arch = json.load(open("checkpoints/tx2k/n1000_scratch/arch.json"))
model = GridTransformerBackbone(hidden_dim=arch["hidden_dim"],
                                num_blocks=arch["num_blocks"])
model.load_state_dict(torch.load("checkpoints/tx2k/n1000_scratch/control_model.pt",
                                 weights_only=True, map_location="cpu"))
model.clip_mode = "sigmoid"
model.eval()
```

`scripts/_bootstrap.py` puts `src/` and the repo's `model/` directory on `sys.path`, so
you never set `PYTHONPATH` yourself. The bundled `gridsfm` takes precedence over any
installed copy.

## Usage

```bash
python scripts/validate_config.py --config config/case500_goc_finetune.yaml  # no GPU
python scripts/prep_cache.py --config config/case500_goc_finetune.yaml       # once per dataset
python scripts/finetune.py --config config/case500_goc_finetune.yaml \
    --train_subset 100 --run_label n0100 --device cuda

bash scripts/train_grid.sh case500_goc              # plan only
bash scripts/train_grid.sh case500_goc --go         # n = 10,25,50,100,200,500
bash scripts/train_grid.sh case500_goc --scratch --go
```

`train_grid.sh` takes a GPU only when `nvidia-smi` reports no compute process on it and at
least `MIN_FREE_GB` (default 40) free, so an idle card belonging to someone else is used
and a busy one is left alone. That is a courtesy check, not a reservation. Restrict the
pool with `CAND="0 6" bash scripts/train_grid.sh …`.

Every `finetune.py` flag overrides the corresponding config key for that run only. The
file is not modified, and the **resolved** config is written to
`results/<grid_id>/<run_label>/resolved_config.yaml`, so a run is reproducible from its own
output. Beyond the above: `--epochs` `--min_epochs` `--batch` `--accum` `--lr` `--seed`
`--select_on` `--plot_every` `--data_root` `--data_dir`, the elastic anchor weights
`--w_control[_start|_end|_epochs]`, and `--closure_solver superlu|cudss`.
