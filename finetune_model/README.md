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

`results/`, `logs/` and `cache/` ship empty — they are outputs, and the cache is not
portable between machines. Every output is scoped by `<run_label>`, so a data-scaling
sweep writes N independent result sets instead of overwriting one; the cache is **not**,
since it is per dataset and shared by every run.

## What is released

```
checkpoints/
├── base/gridsfm_open_v2.pt          the released backbone every fine-tune starts from
└── <grid_id>/<run_label>/           one directory per training run
```

`base/gridsfm_open_v2.pt` is `GridSFM-open-v2`, 15.1 M parameters, 61 MB — the upstream
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
training-set size, and the subsets are **stratified and nested** — `subset(10)` ⊂
`subset(100)` ⊂ … — so a difference between two points is more data, not different data.
Val and test are never subsetted, so every point is selected on the same val cases and
scored on the same held-out test cases.

`n1000_scratch` (`config/<grid>_scratch1k.yaml`) is what makes the rest mean something: the
same architecture family and the same harness, trained from random init on 1000 cases —
*more* data than any fine-tuned point gets. It answers "is the pretrained backbone doing
anything, or would this much data have sufficed on its own?"

### Files inside a run directory

| file | what it is |
|---|---|
| `control_model.pt` | **the deliverable.** A copy of whichever checkpoint `train.select_on` chose. Every shipped run used `select_on: val`, so it is weight-identical to `best_val.pt` (the bytes differ — it is re-serialised on save). |
| `best_val.pt` | the epoch with the lowest validation score, `val_score_pg_weight × Pg% + val_score_v_weight × V(p.u.)` |
| `last.pt` | the final epoch, whenever training stopped |
| `best_train_loss.pt` | lowest epoch *training* loss. Written by the `control`-mode loop only, so `n1000_scratch` has it and the elastic runs do not |
| `control_model.json` | `control`-mode sidecar: the same summary as `test_metrics.json` (test/val errors, ship epoch, wall time, resolved hyperparameters, config and git sha) |
| `arch.json` | `from_scratch` runs only: `{"hidden_dim": 64, "num_blocks": 4}`, so a loader never has to infer the architecture from tensor shapes. Inferring it wrongly once loaded a 64×4 checkpoint into the 128×8 backbone |

Fine-tuned runs are ~175 MB per directory (three 61 MB copies of the 128×8 backbone);
`n1000_scratch` is ~31 MB, since `from_scratch` builds a 64×4 backbone of 1.96 M
parameters.

### Loading one

Run outputs are **bare `state_dict`s**, not the wrapped release format — only
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

### What the from-scratch control measured

`control_model.json` survives inside each `n1000_scratch/`, so these numbers are readable
straight from the checkpoints. `baseline` is the un-fine-tuned backbone on the same val
cases.

| grid | scored gens / V-buses | topk_pg / topk_v | ship epoch | TEST Pg | TEST V (p.u.) | baseline val Pg | train |
|---|---|---:|---:|---:|---:|---:|---:|
| `case500_goc` | 170 / 112 | 7 / 3 | 290 of 330 | 2.08% | 0.0050 | 40.4% | 0.7 h |
| `tx2k` | 735 / 409 | 32 / 10 | 321 of 361 | 0.69% | 0.0049 | 45.0% | 3.4 h |
| `case6470_rte` | 760 / 451 | 33 / 11 | 393 of 400 | 3.84% | 0.0110 | 49.3% | 6.1 h |
| `activsg10k` | 1936 / 1255 | 84 / 32 | 397 of 400 | 0.76% | 0.0030 | 27.9% | 9.0 h |

The top-k widths are derived, not configured, and land on the configured fractions: 84/1936
= 4.34% against `topk_pg_frac: 0.0435`, and 32/1255 = 2.55% against `topk_v_frac: 0.0254`.
Note `activsg10k`'s 1,255 scored V-control buses against 1,728 nominal ones — the rest are
PV buses demoted to PQ because their available generators have no net reactive range.

The elastic runs' equivalents live in `results/<grid_id>/<run_label>/test_metrics.json` and
`elastic_summary.json`, which are regenerated by training rather than shipped.

### One irregularity

`case6470_rte/n0025/` has no `best_val.pt` — only `control_model.pt` and `last.pt`. The
deliverable is intact and is *not* a copy of `last.pt`, so selection happened normally; the
intermediate was pruned afterwards.

`scripts/_bootstrap.py` puts `src/` and the repo's `model/` directory on `sys.path`, so
you never set `PYTHONPATH` yourself; the bundled `gridsfm` takes precedence over any
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

Every `finetune.py` flag overrides the corresponding config key for that run only; the
file is not modified, and the **resolved** config is written to
`results/<grid_id>/<run_label>/resolved_config.yaml`, so a run is reproducible from its own
output. Beyond the above: `--epochs` `--min_epochs` `--batch` `--accum` `--lr` `--seed`
`--select_on` `--plot_every` `--data_root` `--data_dir`, the elastic anchor weights
`--w_control[_start|_end|_epochs]`, and `--closure_solver superlu|cudss`.

## The split is not ours to make

The harness reads **three separate files** — `train_data.jsonl`, `val_data.jsonl`,
`test_data.jsonl` — exactly as published, and never re-splits, re-shuffles or merges them.

| split | job | how often it is read |
|---|---|---|
| train | fits the weights | every epoch |
| val | chooses which epoch ships | every epoch, no gradient |
| test | the reported number | **once**, after selection |

`train.select_on` accepts `val`, `final` or `train_loss` — deliberately **not** `test`.
Three independent guards enforce this: `config.validate` rejects `select_on: test` and
refuses two split keys that resolve to the same file; `cache.build` raises if any two
splits share a cache key; and `train.run` asserts it again from the manifest before the
first epoch.

## The objective

**`mode: elastic`** — what the four `*_finetune.yaml` recipes use. Self-supervised AC-OPF
evaluated at the power-flow-**closed** operating point, with a supervised anchor:

```
L = sum_i c_i(Pg_i)                      # generation cost
  + rho_g * sum log(1 + |F|)             # equality slack, two-sided
  + rho_h * sum log(1 + relu(h~))        # inequality slack, one-sided
  + w_control * control_loss(pred, gt)   # GT anchor
  + w_merit   * 0.5 * mean(F^2)          # feasibility descent
```

`F` is the nodal power-balance residual (active at non-slack buses, reactive at PQ buses);
`h~` collects the gen-P / gen-Q / voltage / thermal inequality residuals. The model's
controls are pushed through a Newton power flow ([`src/closure.py`](src/closure.py)) and
the objective is evaluated at the converged point, so what is optimised is a real
operating point rather than a regression target. `closure.tol_start → tol_end` is a
curriculum: the feasibility gap is loose early and tightens over `tol_epochs`.

**`mode: control`** — what the `*_scratch1k.yaml` recipes use. The max-aligned anchor
alone, which is also the elastic loss's anchor term:

```
L_q  = mean_weight*mean(e) + topk_weight*mean(topk(e)) + max_weight*max(e)
total = w_pg * L_pg + w_v * L_v,          e = |pred - gt| / tol
```

summed over generators (`tol_pg`) and V-control buses (`tol_v`).

* **tol-normalised absolute** error, not range-normalised: range-norm degenerates on fixed
  generators (`Pmax == Pmin`) and silently stops scoring them.
* the **top-k and max** terms make the loss track the *worst* control error, which is what
  drives the downstream corrector's cost. A plain mean lets a handful of bad generators
  hide behind ~1900 good ones.
* therefore top-k must be taken **within a case**. Pooled across a batch it selects the
  worst k in the *batch*, and a case whose worst error is merely average contributes
  nothing to the max-aligned part.

So `train.batch > 1` requires `loss.segmented_topk: true`; the validator refuses the unsafe
combination rather than letting it quietly change what is optimised.

## The recipes

Every key lives in each config explicitly with its own comment — read one file and you
know what a run did. What differs *between* the four elastic recipes is the part worth
knowing; everything else is shared (`epochs: 400`, `select_on: val`,
`early_stop_patience: 20`, `seed: 0`, adam + cosine with `lr_min_frac: 0.05`,
`clip_grad: 1.0`, `tol_pg: 0.01`, `tol_v: 0.001`, `topk_pg_frac: 0.0435`,
`topk_v_frac: 0.0254`, `max_newton: 40`, `closure.workers: 16`, `solver: superlu`).

| | `case500_goc` | `tx2k` | `case6470_rte` | `activsg10k` |
|---|---:|---:|---:|---:|
| `train.batch` | 16 | 4 | 4 | 4 |
| `train.lr` | 1e-4 | 1e-4 | 1e-4 | 3e-4 |
| `train.min_epochs` | 100 | 50 | 50 | 50 |
| `loss.w_control_start → end` | 32 → 32 | 1600 → 1600 | 13.05 → 85 | 300 → 300 |
| `loss.w_merit` | 170 | 144 | 200 | 2000 |
| `loss.rho_g` / `rho_h` | 10 / 10 | 10 / 10 | 10.4 / 14.5 | 10 / 10 |
| `closure.tol_start → end` | 0.1 → 1e-8 | 0.1 → 1e-8 | 1e-5 (fixed) | 1e-5 (fixed) |
| `closure.warm_start` | true | true | **false** | true |
| `cache.workers` / `mem_budget_gb` | 12 / 6 | 16 / 8 | 16 / 8 | 20 / 10 |
| `runtime.min_free_vram_gb` | 6 | 12 | 22 | 30 |
| `train.select_min_frac_converged` | — | — | 0.99 | 1.0 |

`select_min_frac_converged` refuses to ship an epoch whose closure did not converge on
that fraction of val cases. On the two large grids an epoch can score well precisely
because its power flow stalled, so the guard is what stops a non-operating point from being
selected. `runtime.min_free_vram_gb` is checked before the model is built: the run
**refuses to start** rather than OOM a co-tenant's job.

`*_scratch1k.yaml` differs from the fine-tuning recipe in exactly five keys:
`model.from_scratch: true` (random init at `hidden_dim: 64`, `num_blocks: 4`, ignoring
`base_checkpoint`), `loss.mode: control`, `train.lr: 1e-3`,
`train.early_stop_patience: 40`, `train.min_epochs: 50`. It runs at `--train_subset 1000`.

### Derived, not configured

Three grid-dependent numbers are **measured from the cache** rather than written per
dataset: `topk_pg` (`topk_pg_frac` × mean scored generators), `topk_v` (`topk_v_frac` ×
mean scored V-control buses), and the LRU capacity (`cache.mem_budget_gb` ÷ measured
bytes/case). "Scored" is narrower than "present" — a PV bus whose available generators
have no net reactive range cannot hold voltage, so it is demoted to PQ and not scored; on
ACTIVSg10k that takes 1,728 nominal control buses down to ~1,255. `prep_cache.py` prints
the measured counts and the derived widths.

## Outputs

`results/<grid_id>/<run_label>/` holds `training_curves.png` / `elastic_training.png`,
`test_diagnostics.png`, `history_epoch.csv/.json`, `history_step.csv`,
`test_metrics.json`, `elastic_summary.json` (elastic runs), `test_arrays.npz`,
`timing.json` and `resolved_config.yaml`.

Per-step history exists because epoch means hide what you need it for: pre-clip gradient
norms run in the hundreds to thousands on this loss (the 10k run peaked at 8069 against
`clip_grad=1.0`), so **expect `clip_frac ≈ 1.0`**. That is the regime, not a bug, and it is
invisible in an epoch average.

`checkpoints/<grid_id>/<run_label>/` holds `control_model.pt` plus `best_val.pt` and
`last.pt`; control-mode runs also write `best_train_loss.pt` and a `control_model.json`
sidecar carrying the same summary as `test_metrics.json`, and `from_scratch` runs write
`arch.json` (`hidden_dim` / `num_blocks`) so the eval harness never infers the architecture
from tensor shapes.

## Why batch size is not free

At batch 1 a 500-bus graph spends ~75% of the step on kernel-launch overhead. Measured on
case500, real forward+backward:

| `batch` | ms/step | ms/case | 30 epochs × 2000 cases |
|---|---:|---:|---:|
| 1 | 96.6 | 96.6 | ~97 min |
| 4 | 99.1 | 24.8 | ~25 min |
| 16 | 151.1 | 9.4 | ~9 min |

The substitution of 4 × 1 for 1 × 4 was measured, not assumed: the loss differs by 2.5e-5
relative and the gradient cosine against 1 × 4 is 0.9999977 — float-summation noise rather
than a change in semantics — so `lr` and step count carry over unchanged. Larger still
trades memory and optimizer steps for wall clock: on an RTX 5090, case500 at batch 4 uses
1.8 GB and ~26 min for 30 epochs, while batch 32 reaches 13.9 GB and ~8 min but leaves only
62 optimizer steps per epoch.

## Notes on the cache

* Keyed on `md5(grid_id | mode | scenario_id | scenario_seed)` — **not** on the split.
  `(mode, scenario_id, scenario_seed)` already identifies a case uniquely, and keying on
  the split too would mean a re-split invalidated every case that moved: harmless on
  case500, a full rebuild on a 10k-bus dataset where prep is the expensive part. The
  manifest records which keys belong to which split, which is where that belongs.
* Machine-local and path-independent, but **not portable** — rebuild it rather than
  copying it. `prep_cache.py` is incremental, so re-running after an `--append` only
  prepares the new cases; writes go to `.tmp` then `os.replace`.
* Workers seek to a byte offset and read **one line** rather than receiving the record
  through a pipe — records are 0.3–3 MB and pickling them dominates the prep itself.
* A case that fails to prepare is counted (`n_failed`) and skipped rather than killing the
  build; the build **does** refuse to continue if any two splits share a case, which would
  mean a held-out case had been published twice. `load_manifest` refuses a cache whose
  `grid_id` does not match the config.

## Hardcoded parameters

After exposing the settings above, this is what is still fixed in source, and why:

| where | value | why it is not a config key |
|---|---|---|
| `masks.py` | `qrange < 1e-6` | PV→PQ demotion threshold. A bus whose available generators have no net reactive range physically cannot hold voltage; this is a zero test, not a tuning knob |
| `store.py` | `Prange.clamp_min(1e-3)`, `Vband.clamp_min(0.02)` | normaliser floors guarding divide-by-zero on fixed generators and degenerate voltage bands |
| `evaluate.py` | `pg_pct = Σ\|err\| / Σ\|gt\|` | aggregate-relative by definition, not a mean of per-unit ratios — small units would otherwise dominate the number |
| `train.py` | float64 loss and metrics | control errors are compared against p.u. tolerances of 1e-3; float32 would put rounding noise at the same order as `tol_v` |
| `train.py` | `batch` clamped to the training-set size | a micro-batch larger than the training set makes the step loop empty, so the run would complete every epoch without a single optimizer step and report a perfectly flat loss |
| `optim.py` | head match: name ends in `head`, contains `.head`, or contains `pred` | how `freeze: trunk` finds the output heads; matched by name so it survives trunk refactors, and errors out if it matches nothing |
