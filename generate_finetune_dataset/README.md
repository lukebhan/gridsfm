# generate_finetune_dataset

Perturbs a base grid topology, solves each scenario to optimality with Ipopt, exports
grid + solution as `.pyg.json`, and publishes fixed `train`/`val`/`test` splits as JSONL.

Install and dataset downloads are covered in the [root README](../README.md). This file
documents the generator itself.

```
config/       one YAML per grid: topology, seeds, perturbation ranges, split ratios
topologies/   Matpower .m files
src/          Python orchestration + Julia solve/export, with Project.toml as the Julia env
scripts/      entry points
data/         the published datasets
```

Generating datasets does not need PyTorch: the Python side is orchestration only and the
solving is Julia, so `PyYAML` from the repo-root `requirements.txt` plus a Julia 1.12 env
is enough if you are not training.

## The released datasets

Four grids, 2,600 feasible scenarios each, published as a fixed 2000 / 100 / 500 split.
Same seeds (`master 20260818`, `split 0`) and the same mode probabilities
(`loads 1.00 · costs 1.00 · killgen 0.30 · derate 0.20 · vsqueeze 0.15`) throughout, so
the four differ only in grid and load range.

| grid | buses | gens (active) | branches (rated) | control buses | loaded | ceiling | load sf | killgen |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `case500_goc` | 500 | 224 (171) | 733 (733) | 114 | 76.3% | 1.31× | 0.80–1.20 | 1–3 |
| `tx2k` | 2,751 | 1,099 (736) | 5,344 (5,344) | 946 | 79.4% | 1.26× | 0.80–**1.30** | 1–3 |
| `case6470_rte` | 6,470 | 1,330 (761) | 9,005 (9,005) | 676 | 81.9% | 1.22× | 0.80–1.20 | 1–3 |
| `activsg10k` | 10,000 | 2,485 (1,937) | 12,706 (10,244) | 1,728 | 88.8% | 1.13× | 0.80–1.10 | 2–6 |

*ceiling* is the static lossless capacity bound `ΣPmax / ΣPd` over active generators.
`tx2k` is the one grid whose range deliberately breaches it
(`allow_load_sf_above_ceiling: true`), to keep diversity at the top of the load range.
That costs solve time, not dataset quality, since only feasible cases are published.
`activsg10k` trips 2–6 units because tripping 1 of 1,937 is not a perturbation at that
scale.

```
data/<grid_id>/
├── train_data.jsonl   one scenario per line (compact JSON)
├── val_data.jsonl
├── test_data.jsonl
├── splits.json        seeds, ratios, mode probabilities, case names per split
├── _records/          one line per ATTEMPTED case -- drives the statistics and resume
└── generate.log       the run that produced this dataset, full statistics included
```

## Usage

```bash
python scripts/validate_config.py                        # check every config, no solving
python scripts/generate_finetune_dataset.py --config config/tx2k.yaml \
    --num_workers 196 --total_num_feasible 2600
python scripts/validate_dataset.py --dir data/tx2k       # audit + replay the statistics
```

`--total_num_feasible` is a count of **feasible** cases, not scenarios: feasibility is
only knowable by solving, so a fixed scenario count under-delivers unpredictably. Other
flags: `--max_iters` (overrides the config's Ipopt cap), `--out`, `--append` (amend a
dataset without disturbing the existing split), `--keep_staging`, `--dry_run`, `--julia`,
`--project`. `validate_dataset.py` takes `--full` to parse every line instead of
sampling.

## Perturbation modes

Modes **compose**: a scenario applies every mode whose `mode_probs` draw succeeds to the
same grid, so it can be simultaneously more loaded, re-dispatched and short a unit. A
gate is drawn for every mode even at probability 1.0, so the RNG stream does not depend
on the probability *values*. Lowering `killgen` from 0.30 to 0.20 changes which
scenarios trip a unit without also reshuffling every load factor.

`loads` and `costs` are unconditional, and the rest are gated independently, so **2–4 modes
are active at once and never exactly 1** (mean 2.65 across all four datasets). Order is
fixed at `loads → costs → killgen → derate → vsqueeze`. `costs` must precede `killgen`,
since it reshuffles the merit order over units that are still online.

| mode | what it does | severity recorded |
|---|---|---|
| `loads` | one system factor `sf ~ U[lo, hi]`, then every load's `pd`/`qd` scaled by `sf · U[1±jitter]` | `sf` |
| `costs` | picks `cost_frac` of online gens, buckets by cost-function degree, permutes cost vectors **within** each bucket. Degree preserved, and what changes is *which* units are cheap | % of gens whose curve moved |
| `killgen` | trips `k` units, `k` from `killgen_nk` by inverse CDF. Candidates are online units above `killgen_pmax_threshold`, clamped so `killgen_min_online` stay up | units tripped |
| `derate` | scales `rate_a`/`b`/`c` by `U[derate_lo, derate_hi]` on `derate_frac` of in-service **rated** branches, independently per branch | mean derate factor |
| `vsqueeze` | on `vsqueeze_frac` of buses, raises `vmin` and lowers `vmax` by draws in `[0, vsqueeze_delta]`, and a band that would invert reverts | mean band shrink (p.u.) |

`cost_frac` is the fraction *entering* the shuffle, not the fraction swapped. A
same-degree pool smaller than `costs_min_pool` is skipped, so at `cost_frac: 0.40` the
released runs measured 22.9–31.2% of the fleet actually changing curve.

Because modes compose, a case cannot be attributed to a single mode, so yield is reported
**marginally**: each row covers the scenarios in which that mode fired, and the rows
overlap. `derate` is consistently the worst, at 36% yield on `tx2k` and 41% on
`case6470_rte` against ~60% overall, because congesting branches composes badly with a
high load factor.
