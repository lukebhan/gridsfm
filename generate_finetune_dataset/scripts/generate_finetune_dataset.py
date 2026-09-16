#!/usr/bin/env python3
"""Generate a dataset of perturbed AC-OPF scenarios for one grid.

    python scripts/generate_finetune_dataset.py \
        --config config/case500_goc.yaml \
        --num_workers 32 --total_num_feasible 2500 \
        --out /path/to/output

Generates until --total_num_feasible FEASIBLE cases exist, not until a fixed scenario
count is attempted: feasibility is only knowable by solving, so a fixed count
under-delivers unpredictably. Resumable: existing cases are skipped, so re-running tops
a dataset up.
"""
from __future__ import annotations
import argparse, os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as cfgmod, dataset, orchestrate, stats as statsmod   # noqa: E402


def main() -> int:
    # stdout is block-buffered when redirected to a file, which leaves a nohup log empty
    # for the whole run. Force line buffering so progress is visible as it happens.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config/<grid>.yaml")
    ap.add_argument("--out", default=None,
                    help="root for output; final layout is "
                         "<out>/<grid_id>/{train,val,test}_data.jsonl (default: <repo>/data)")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--total_num_feasible", type=int, required=True,
                    help="stop once this many feasible cases exist; the published "
                         "dataset is trimmed to exactly this number")
    ap.add_argument("--max_iters", type=int, default=None,
                    help="Ipopt max_iter (overrides the config)")
    ap.add_argument("--julia", default=os.path.expanduser("~/.local/bin/julia"))
    ap.add_argument("--project", default=None, help="Julia project dir (default: <root>/src)")
    ap.add_argument("--append", action="store_true",
                    help="AMEND an existing dataset: keep every existing case and its "
                         "train/val/test assignment, and add new cases until the dataset "
                         "holds --total_num_feasible in total. Existing test cases stay in "
                         "test, so a held-out claim survives the amendment.")
    ap.add_argument("--keep_staging", action="store_true",
                    help="keep the per-case .pyg.json staging files after consolidation")
    ap.add_argument("--dry_run", action="store_true",
                    help="validate the config, generate a handful of cases, then stop")
    a = ap.parse_args()

    cfg = cfgmod.load_config(a.config, root=ROOT)
    if a.max_iters:
        cfg["solver"]["max_iters"] = a.max_iters

    # <out>/<grid_id>/ holds the published .jsonl; _staging/ holds resumable per-case files
    out_root = a.out or os.path.join(ROOT, "data")
    grid_dir = os.path.join(out_root, cfg["grid_id"])
    staging = os.path.join(grid_dir, "_staging")

    p = cfg["perturbations"]
    print("=" * 78)
    print(cfg["_stats"].summary())
    print(f"  seeds: master={cfg['seeds']['master']} split={cfg['seeds']['split']}")
    print(f"  load sf [{p['load_sf_lo']}, {p['load_sf_hi']}] (ceiling "
          f"{cfg['_stats'].lossless_sf_ceiling:.2f}x) | killgen {p['killgen_nk']} | "
          f"max_iters {cfg['solver']['max_iters']}")
    print("  mode probabilities: " + "  ".join(
        f"{m}={float(p['mode_probs'][m]):.2f}" for m in orchestrate.MODE_ORDER
        if m in p["mode_probs"]))
    sp = cfg["splits"]
    print(f"  splits train={sp['train']:.0%} val={sp['val']:.0%} test={sp['test']:.0%}"
          f"  -> {grid_dir}/{{train,val,test}}_data.jsonl")
    for w in cfg["_warnings"]:
        print(f"  WARNING: {w}")
    print("=" * 78)

    target = min(a.total_num_feasible, orchestrate.DRY_RUN_CASES) if a.dry_run \
        else a.total_num_feasible
    if a.dry_run:
        print(f"DRY RUN: {target} cases\n")

    # On --append, tell the orchestrator what is already PUBLISHED. Without this it counts
    # the durable records, which include feasible cases that the publish step trimmed and
    # whose staging files were then deleted, so it would think the target was already met
    # and generate nothing.
    published = 0
    _man = os.path.join(grid_dir, "splits.json")
    if a.append and os.path.isfile(_man):
        _prior = json.load(open(_man))
        published = sum(len(_prior.get("cases", {}).get(s, []))
                        for s in ("train", "val", "test"))

    t0 = time.time()
    res = orchestrate.generate(
        cfg, staging, num_workers=min(a.num_workers, max(1, target)),
        total_num_feasible=target, julia=a.julia,
        project=a.project or os.path.join(ROOT, "src"),
        max_rounds=orchestrate.DRY_RUN_ROUNDS if a.dry_run else orchestrate.MAX_ROUNDS,
        published=published)

    print(f"\n{'='*78}\nfeasible {res['feasible']} in {(time.time()-t0)/60:.1f} min")
    print(f"  {'kind':<10}{'feasible':>9}{'infeasible':>12}{'yield':>8}")
    for k in sorted(set(res["per_mode"]) | set(res["infeasible"])):
        f_, i_ = res["per_mode"].get(k, 0), res["infeasible"].get(k, 0)
        print(f"  {k:<10}{f_:>9}{i_:>12}{100*f_/max(f_+i_,1):>7.0f}%")
    if res["bad"]:
        print(f"  CORRUPT: {len(res['bad'])} files")
        for f_, why in res["bad"][:5]:
            print(f"    {os.path.basename(f_)}: {why}")

    for line in dataset.verify_schema(res["feasible_files"]):
        print(f"  schema {line}")

    if a.dry_run:
        print(f"\n  dry run complete; staging kept at {staging}")
        return 0

    man_path = os.path.join(grid_dir, "splits.json")
    prior = json.load(open(man_path)) if (a.append and os.path.isfile(man_path)) else None

    if prior:
        # Publish every newly generated feasible case, trimmed to the shortfall against
        # the target, and preserve every prior assignment. Re-splitting everything would
        # move cases between train and test and silently invalidate any held-out result
        # already reported from this dataset.
        assigned = {name: set(prior["cases"][name]) for name in ("train", "val", "test")}
        known = set().union(*assigned.values())
        new = [f for f in sorted(res["feasible_files"])
               if os.path.basename(f) not in known][:max(0, target - published)]
        n = len(new)
        n_test = int(round(n * cfg["splits"]["test"]))
        n_val = int(round(n * cfg["splits"]["val"]))
        add = dataset.make_splits(new, n_test, n_val, cfg["seeds"]["split"])
        print(f"\n  APPEND: {len(known)} existing cases keep their split; "
              f"{n} new -> train {len(add['train'])} / val {len(add['val'])} / test {len(add['test'])}")
        sp = {k: sorted(assigned[k]) + [os.path.basename(x) for x in add[k]] for k in add}
        sp_files = {k: add[k] for k in add}          # only new files need writing
        append_mode = True
    else:
        # Trim to exactly --total_num_feasible (generation overshoots by whatever was in
        # flight at the cutoff), then split on a shuffle seeded from seeds.split.
        chosen = dataset.take(res["feasible_files"], min(target, res["feasible"]))
        n = len(chosen)
        n_test = int(round(n * cfg["splits"]["test"]))
        n_val = int(round(n * cfg["splits"]["val"]))
        sp_files = dataset.make_splits(chosen, n_test, n_val, cfg["seeds"]["split"])
        sp = {k: [os.path.basename(x) for x in v] for k, v in sp_files.items()}
        append_mode = False

    print(f"\n  publishing -> {grid_dir}")
    for name in ("train", "val", "test"):
        if not sp_files[name]:
            continue
        jp = os.path.join(grid_dir, f"{name}_data.jsonl")
        cnt, size = dataset.write_jsonl(sp_files[name], jp, append=append_mode)
        total_lines = sum(1 for _ in open(jp))
        print(f"    {name+'_data.jsonl':<20} +{cnt:>5} -> {total_lines:>5} cases  {size/1e9:>6.2f} GB")

    with open(man_path, "w") as f:
        json.dump({"grid_id": cfg["grid_id"], "config": os.path.basename(a.config),
                   "seed_master": cfg["seeds"]["master"], "seed_split": cfg["seeds"]["split"],
                   "splits_ratio": cfg["splits"], "mode_probs": p["mode_probs"],
                   "counts": {k: len(v) for k, v in sp.items()}, "cases": sp}, f, indent=1)
    print(f"    splits.json          provenance: seeds, ratios, per-split case names")

    print()
    print(statsmod.report(statsmod.load_records(res["recdir"]),
                          list(orchestrate.MODE_ORDER), cfg["solver"]["max_iters"]))

    if not a.keep_staging:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        print(f"    (staging removed; pass --keep_staging to retain resumable per-case files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
