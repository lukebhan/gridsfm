#!/usr/bin/env python3
"""Validate configs against their grids without generating anything.

    python scripts/validate_config.py                  # all configs
    python scripts/validate_config.py --config x.yaml   # one
"""
from __future__ import annotations
import argparse, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as cfgmod                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
a = ap.parse_args()
paths = [a.config] if a.config else sorted(glob.glob(os.path.join(ROOT, "config", "*.yaml")))
rc = 0
for p in paths:
    name = os.path.basename(p)
    try:
        c = cfgmod.load_config(p, root=ROOT)
        print(f"[OK]   {name}")
        print("       " + c["_stats"].summary().replace("\n", "\n       "))
        pp = c["perturbations"]
        print(f"       load sf [{pp['load_sf_lo']}, {pp['load_sf_hi']}] vs ceiling "
              f"{c['_stats'].lossless_sf_ceiling:.2f}x | killgen {pp['killgen_nk']}"
              f" ({100*max(pp['killgen_nk'])/c['_stats'].n_gen_active:.1f}% of active gens)")
        print("       mode probs " + "  ".join(
            f"{m}={float(v):.2f}" for m, v in sorted(pp["mode_probs"].items())))
        print(f"       seeds master={c['seeds']['master']} split={c['seeds']['split']}")
        for w in c["_warnings"]:
            print(f"       WARN: {w}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}"); rc = 1
    print()
sys.exit(rc)
