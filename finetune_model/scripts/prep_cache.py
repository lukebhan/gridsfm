#!/usr/bin/env python3
"""Build the prepared-case cache for a dataset (once per dataset).

prepare_for_inference (cycle basis + Hodge PE) is the expensive part of a launch and
its result is identical every epoch, so it is computed once and stored per case.

    PYTHONPATH is handled internally; just run:
    python scripts/prep_cache.py --config config/case500_goc_finetune.yaml [--workers N]
"""
import argparse, sys
import _bootstrap  # noqa: F401
import config as configmod
import cache as cachemod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None, help="overrides cache.workers")
    ap.add_argument("--data_root", help="overrides data.root (dataset location)")
    ap.add_argument("--data_dir", help="overrides data.dir (this dataset's directory)")
    a = ap.parse_args()

    cfg = configmod.load_config(a.config, data_root=a.data_root, data_dir=a.data_dir)
    for w in configmod.validate(cfg, require_data=True):
        print(f"WARNING: {w}")
    if a.workers:
        cfg["cache"]["workers"] = a.workers
    man = cachemod.build(cfg)
    st = man["stats"]
    print(f"\n  scored controls per case: {st['mean_scored_gens']:.1f} generators, "
          f"{st['mean_scored_vctrl']:.1f} V-control buses")
    cfg = configmod.derive(cfg, st["mean_scored_gens"], st["mean_scored_vctrl"],
                           st["bytes_per_case"])
    d = cfg["_derived"]
    print(f"  derived top-k: topk_pg={d['topk_pg']} ({d['topk_pg_pct']}%)  "
          f"topk_v={d['topk_v']} ({d['topk_v_pct']}%)")
    print(f"  derived LRU: {d['lru_cases']} cases at {d['bytes_per_case']/1024:.0f} KB each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
