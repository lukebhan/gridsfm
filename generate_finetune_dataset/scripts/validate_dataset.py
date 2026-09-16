#!/usr/bin/env python3
"""Audit a generated dataset.

Works on a PUBLISHED dataset (train/val/test .jsonl + splits.json) and, if present, on
leftover staging. Verifies: every line parses and is feasible, split integrity (no overlap
between train/val/test, no duplicate scenarios), manifest agreement, and replays the
generation statistics from _records/.

    python scripts/validate_dataset.py --dir data/tx2k
"""
from __future__ import annotations
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import dataset, orchestrate, stats as statsmod                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="data/<grid_id>/")
ap.add_argument("--full", action="store_true",
                help="parse every line (slow on multi-GB files) instead of sampling")
a = ap.parse_args()
rc = 0
print(f"dataset: {a.dir}")

# ---- published JSONL splits ----
seen_ids: dict[str, set] = {}
for name in ("train", "val", "test"):
    p = os.path.join(a.dir, f"{name}_data.jsonl")
    if not os.path.isfile(p):
        print(f"  {name}_data.jsonl  MISSING"); continue
    n = bad = infeas = 0
    fired = collections.Counter(); ids = set()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            bad += 1; continue
        md = o.get("metadata", {})
        if not md.get("feasible"):
            infeas += 1
        for _m in md.get("modes_applied", []) or []:
            fired[_m] += 1
        ids.add((md.get("perturb_mode"), md.get("scenario_id")))
    seen_ids[name] = ids
    dupes = n - len(ids)
    flag = ""
    if bad or infeas or dupes:
        rc = 1
        flag = f"  <-- bad_json={bad} infeasible={infeas} duplicates={dupes}"
    mix = "  ".join(f"{m} {100*c/max(n,1):.0f}%" for m, c in sorted(fired.items()))
    print(f"  {name+'_data.jsonl':<18} {n:>6} cases  {os.path.getsize(p)/1e9:>6.2f} GB"
          f"{flag}")
    if mix:
        print(f"    modes fired: {mix}")

# ---- split integrity ----
for x, y in (("train", "test"), ("train", "val"), ("val", "test")):
    if x in seen_ids and y in seen_ids:
        ov = seen_ids[x] & seen_ids[y]
        if ov:
            rc = 1
            print(f"  LEAK: {len(ov)} scenarios appear in both {x} and {y}: {sorted(ov)[:5]}")
if all(k in seen_ids for k in ("train", "val", "test")):
    print(f"  split integrity: no overlap between train/val/test")

# ---- manifest agreement ----
man = os.path.join(a.dir, "splits.json")
if os.path.isfile(man):
    m = json.load(open(man))
    print(f"  splits.json: grid={m['grid_id']} seeds master={m['seed_master']} "
          f"split={m['seed_split']} ratios={m['splits_ratio']}")
    for name in ("train", "val", "test"):
        p = os.path.join(a.dir, f"{name}_data.jsonl")
        if os.path.isfile(p):
            lines = sum(1 for _ in open(p))
            want = m["counts"].get(name, 0)
            ok = lines == want
            rc = rc or (0 if ok else 1)
            print(f"    {name}: manifest {want} vs jsonl {lines}  {'OK' if ok else 'MISMATCH'}")
    cs = m.get("cases", {})
    if cs:
        allc = [c for v in cs.values() for c in v]
        if len(allc) != len(set(allc)):
            rc = 1; print("    manifest lists a case in more than one split  <-- LEAK")
else:
    print("  splits.json MISSING (cannot verify split provenance)"); rc = 1

# ---- leftover staging ----
stg = os.path.join(a.dir, "_staging")
if os.path.isdir(stg):
    s = dataset.scan(stg)
    print(f"  _staging present: {len(s['files'])} files, {len(s['bad'])} corrupt")
    for line in dataset.verify_schema(s["feasible_files"], n=5):
        print(f"    schema {line}")
        if "OK" not in line:
            rc = 1

# ---- generation statistics replay ----
rec = os.path.join(a.dir, "_records")
if os.path.isdir(rec):
    recs = statsmod.load_records(rec)
    if recs:
        cap = max((r.get("iters") or 0) for r in recs)
        print()
        print(statsmod.report(recs, list(orchestrate.MODE_ORDER),
                              cap if cap > 0 else 3000))
else:
    print("  _records missing (no generation statistics available)")

print(f"\nRESULT: {'PASS' if rc == 0 else 'PROBLEMS FOUND'}")
sys.exit(rc)
