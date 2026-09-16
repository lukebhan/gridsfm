# generate_finetune_dataset

Perturbs a base grid topology, solves each scenario to optimality with Ipopt, exports
grid + solution as `.pyg.json`, and publishes fixed `train`/`val`/`test` splits as JSONL.

Install and dataset downloads are covered in the [root README](../README.md). This file
documents the generator itself.

```
config/            one YAML per grid: topology, seeds, perturbation ranges, split ratios
topologies/        Matpower .m files
src/               Python orchestration + Julia solve/export; Project.toml is the Julia env
scripts/           entry points
data/              the published datasets
requirements.txt   PyYAML -- this subproject alone needs nothing else from pip
```

Generating datasets does not need PyTorch: the Python side is orchestration only, and the
solving is Julia. `pip install -r requirements.txt` here is enough if you are not training.

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
(`allow_load_sf_above_ceiling: true`), to keep diversity at the top of the load range;
that costs solve time, not dataset quality, since only feasible cases are published.
`activsg10k` trips 2–6 units because tripping 1 of 1,937 is not a perturbation at that
scale.

What the runs produced — 196 workers on a 256-core host, Julia 1.12.6, Ipopt capped at
500 iterations:

| grid | explored | feasible | yield | wall clock | solver time | s/case | hit iter cap | train / val / test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `case500_goc` | 2,947 | 2,830 | **96.0%** | 3.7 min | 3.3 h | 4.0 | 4 | 0.58 / 0.03 / 0.14 GB |
| `tx2k` | 4,071 | 2,631 | **64.6%** | 50.1 min | 58.7 h | 51.9 | 86 | 3.64 / 0.18 / 0.91 GB |
| `case6470_rte` | 4,475 | 2,636 | **58.9%** | 87.4 min | 233.5 h | 187.9 | 885 | 7.06 / 0.35 / 1.76 GB |
| `activsg10k` | 3,125 | 2,751 | **88.0%** | 77.1 min | 210.9 h | 243.0 | 141 | 8.99 / 0.45 / 2.25 GB |

Yield tracks headroom, not size: `case500_goc` sits 1.31× above its load and loses almost
nothing, while `tx2k` pays for its deliberate breach and `case6470_rte` — the tightest
ratio of load range to ceiling — spends 74% of its solver time on cases that were
discarded. Infeasible cases are the expensive ones: they run to the iteration cap before
failing.

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

## Config

Every key under `perturbations` is **required** — a config that omits one is rejected
rather than silently defaulted. `solver` and `splits` are the only blocks with fallbacks.
Each shipped config carries a per-key comment; the keys are:

```yaml
grid_id: tx2k                 # scenario seeds derive from THIS NAME, not the file path
topology: topologies/....m
seeds:    {master: 20260818, split: 0}
solver:   {max_iters: 500, tol: 1.0e-6, acceptable_tol: 1.0e-4}
perturbations:
  mode_probs: {loads: 1.0, costs: 1.0, killgen: 0.30, derate: 0.20, vsqueeze: 0.15}
  load_sf_lo: 0.80            # system load scale factor, uniform draw
  load_sf_hi: 1.20            # VALIDATED against the grid's capacity ceiling
  load_jitter: 0.10           # per-load +-10% on top of the system factor
  cost_frac: 0.40             # fraction of online gens entering the cost shuffle
  costs_min_pool: 2           # a same-degree bucket smaller than this is skipped
  killgen_nk: [1, 2, 3]       # units tripped
  killgen_probs: [0.7, 0.2, 0.1]
  killgen_min_online: 2
  killgen_pmax_threshold: 0.01
  derate_frac: 0.10
  derate_lo: 0.70
  derate_hi: 0.95
  vsqueeze_frac: 0.10
  vsqueeze_delta: 0.01
splits:   {train: 0.769231, val: 0.038462, test: 0.192307}   # over FEASIBLE cases
```

`mode_probs.loads` must be `1.0`: every record carries one severity scalar,
`perturb_param`, and it is the system load factor — only defined for every scenario if
`loads` fires on every scenario.

`load_sf_hi` is checked against the capacity ceiling and the config is **rejected** if it
exceeds it, unless `allow_load_sf_above_ceiling: true` downgrades that to a warning.
Warnings are also issued when `load_sf_hi` is within 3% of the ceiling, when `load_sf_lo`
puts load below total `Pmin`, when `killgen_nk` is under 0.1% of the online fleet, and
when branches have `rate_a = 0` (never selected by `derate` — 2,462 of ACTIVSg10k's
12,706). `validate_config.py` prints all of this per config.

## Perturbation modes

Modes **compose**: a scenario applies every mode whose `mode_probs` draw succeeds to the
same grid, so it can be simultaneously more loaded, re-dispatched and short a unit. A
gate is drawn for every mode even at probability 1.0, so the RNG stream does not depend
on the probability *values* — lowering `killgen` from 0.30 to 0.20 changes which
scenarios trip a unit without also reshuffling every load factor.

`loads` and `costs` are unconditional; the rest are gated independently, so **2–4 modes
are active at once and never exactly 1** (mean 2.65 across all four datasets). Order is
fixed at `loads → costs → killgen → derate → vsqueeze`; `costs` must precede `killgen`,
since it reshuffles the merit order over units that are still online.

| mode | what it does | severity recorded |
|---|---|---|
| `loads` | one system factor `sf ~ U[lo, hi]`, then every load's `pd`/`qd` scaled by `sf · U[1±jitter]` | `sf` |
| `costs` | picks `cost_frac` of online gens, buckets by cost-function degree, permutes cost vectors **within** each bucket. Degree preserved; what changes is *which* units are cheap | % of gens whose curve moved |
| `killgen` | trips `k` units, `k` from `killgen_nk` by inverse CDF. Candidates are online units above `killgen_pmax_threshold`, clamped so `killgen_min_online` stay up | units tripped |
| `derate` | scales `rate_a`/`b`/`c` by `U[derate_lo, derate_hi]` on `derate_frac` of in-service **rated** branches, independently per branch | mean derate factor |
| `vsqueeze` | on `vsqueeze_frac` of buses, raises `vmin` and lowers `vmax` by draws in `[0, vsqueeze_delta]`; a band that would invert reverts | mean band shrink (p.u.) |

`cost_frac` is the fraction *entering* the shuffle, not the fraction swapped — a
same-degree pool smaller than `costs_min_pool` is skipped, so at `cost_frac: 0.40` the
released runs measured 22.9–31.2% of the fleet actually changing curve.

Because modes compose, a case cannot be attributed to a single mode, so yield is reported
**marginally**: each row covers the scenarios in which that mode fired, and the rows
overlap. `derate` is consistently the worst — 36% yield on `tx2k`, 41% on `case6470_rte`
against ~60% overall — because congesting branches composes badly with a high load factor.

## Output schema (`.pyg.json`)

One object per case — `{grid, solution, metadata}` — written pretty-printed in staging
and compacted one-per-line into the published JSONL. `src/export_gridsfm.jl` is the schema
authority; the training pipeline loads it directly as a PyG `HeteroData`.

**Every node and edge row carries a trailing availability column** (1.0 in service, 0.0
out), appended last so earlier column indices are unchanged. All elements are kept — no
active-only filtering — so topology is stable across outage scenarios and the model sees
an outaged element rather than a vanished one.

| `grid.nodes` | columns |
|---|---|
| `bus` | 5 — `base_kv, bus_type, vmin, vmax, avail` (avail = 0 iff `bus_type == 4`) |
| `generator` | 12 — `mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cp2, cp1, cp0, avail` |
| `load` | 3 — `pd, qd, avail` |
| `shunt` | 3 — `bs, gs, avail` |

| `grid.edges` | columns |
|---|---|
| `ac_line` | 10 — `angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c, avail` |
| `transformer` | 12 — `angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to, avail` |
| `generator_link` / `load_link` / `shunt_link` | `senders` / `receivers` only |

`solution.nodes.bus` is `[va, vm]`, `solution.nodes.generator` is `[pg, qg]`,
`solution.edges.*.features` is `[pt, qt, pf, qf]`, and `solution.duals` carries bus
`[λ_p, λ_q, μ_vmin, μ_vmax]`, generator `[μ_pmin, μ_pmax, μ_qmin, μ_qmax]` and per-branch
`[μ_therm_f, μ_therm_t]`. Duals are not consumed by the training pipeline; what matters is
that they stay row-aligned with the feature arrays, so they fall back to full-length zeros
when the constraint-position match cannot be made.

All values are per-unit on `baseMVA` (`base_kv` in kV and `tap` as a nameplate ratio are
the exceptions), angles are radians, and `senders`/`receivers` are **0-indexed row
positions** into the corresponding node array. Original PowerModels ids survive in
`metadata`'s id maps for round-tripping, alongside `objective`, `termination_status`,
`scenario_id`, `perturb_mode`, `feasible`, `grid_id`, `seed_master`, `scenario_seed`,
`solve_seconds`, `solver_iters`, `modes_applied` and every mode's descriptor.

On an infeasible termination every solution and dual field is zeroed with its shape
preserved, so the loader never has to handle a missing field. Generator cost is exported
only for PowerModels cost model 2 with degree ≤ 2; piecewise-linear costs are a hard
**error**, not an approximation, because the solve used the original cost.

## Design notes

**One pool, stop at the target.** Every worker draws from one global pool and the run
stops the instant `--total_num_feasible` feasible cases exist; nothing estimates or tracks
yield. That bounds wasted work to whatever is in flight at that instant — at most
`num_workers - 1` solves, the floor for any parallel scheme. The cutoff reaches workers
through a marker file checked **before every case**, so no signal handling or mid-solve
kill is needed and no partially written case is ever produced. The orchestrator polls
every 3 s (the waste knob), then gives workers 120 s to drain.

**Workers are independent processes, not Julia `Distributed` workers.** A worker death
costs one shard rather than the run, and a `kill -9` on the orchestrator cannot orphan a
worker pool.

**Cases are staged individually, then consolidated.** Per-case files make a run resumable
and killable; JSONL is the publication format. Many workers appending to one JSONL would
interleave and corrupt it. Each staged file is written to `.tmp` and `mv`d into place.

**Only feasible cases are published**; infeasible ones leave a record line. They are not
wasted information — they are what the yield statistics are computed from. Records are
deduplicated on `(mode, sidx)` and tolerate a torn final line, and scenario indices never
reuse an explored index, so a resumed run cannot regenerate a perturbation.

**Every solve starts from the neutral midpoint prior** — `pg`/`qg` at limit midpoints,
`vm` at band midpoint, `va = 0`. PowerModels otherwise starts from the case file's shipped
dispatch, which every perturbed scenario has invalidated. This is also the prior written
into the generator *features*, where it doubles as a leak guard: the true dispatch is the
label, so it is replaced by a solution-free prior in the input.

**A single seeded shuffle makes the split.** Every scenario draws from the same
distribution, so a plain shuffle is already representative and needs no stratification.
