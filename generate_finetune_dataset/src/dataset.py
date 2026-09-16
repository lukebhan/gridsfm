"""Inspect and split a generated dataset: corruption, yield, schema, splits."""
from __future__ import annotations
import os, glob, json, collections, random

# A staged file smaller than this is a truncated write, not a small case.
MIN_CASE_BYTES = 1000
# How much of a file's tail to read for the `feasible` flag.
METADATA_TAIL_BYTES = 1400


def scan(outdir: str) -> dict:
    """Cheap scan: reads only each file's tail for the feasible flag, so a large dataset
    can be audited in seconds. Also catches truncated writes."""
    files = sorted(glob.glob(os.path.join(outdir, "**", "*.pyg.json"), recursive=True))
    feas, infeas, bad = collections.Counter(), collections.Counter(), []
    feasible_files = []
    for f in files:
        sz = os.path.getsize(f)
        if sz < MIN_CASE_BYTES:
            bad.append((f, "too small")); continue
        with open(f, "rb") as fh:
            fh.seek(-3, 2)
            if b"}" not in fh.read():
                bad.append((f, "truncated")); continue
            fh.seek(max(0, sz - METADATA_TAIL_BYTES))
            tail = fh.read().decode("utf8", "ignore")
        kind = os.path.basename(f).split("_")[0]
        if '"feasible": true' in tail:
            feas[kind] += 1; feasible_files.append(f)
        elif '"feasible": false' in tail:
            infeas[kind] += 1
        else:
            bad.append((f, "no feasible flag"))
    return dict(files=files, feasible=feas, infeasible=infeas, bad=bad,
                feasible_files=feasible_files)


def verify_schema(files: list[str], n: int = 3) -> list[str]:
    """Full-parse a few files and check the row counts line up across grid and solution."""
    out = []
    rnd = random.Random(0)
    for f in rnd.sample(files, min(n, len(files))):
        o = json.load(open(f))
        g, s = o["grid"]["nodes"], o["solution"]["nodes"]
        problems = []
        if len(g["bus"]) != len(s["bus"]):
            problems.append(f"bus rows {len(g['bus'])} != solution {len(s['bus'])}")
        if len(g["generator"]) != len(s["generator"]):
            problems.append(f"gen rows {len(g['generator'])} != solution {len(s['generator'])}")
        if len(g["bus"][0]) != 5:
            problems.append(f"bus cols {len(g['bus'][0])} != 5 (raw4 + availability)")
        if len(g["generator"][0]) != 12:
            problems.append(f"gen cols {len(g['generator'][0])} != 12 (raw11 + availability)")
        if "objective" not in o["metadata"]:
            problems.append("metadata.objective missing")
        out.append(f"{os.path.basename(f)}: " + ("OK" if not problems else "; ".join(problems)))
    return out


def make_splits(feasible_files: list[str], n_test: int, n_val: int,
                split_seed: int) -> dict:
    """Deterministic train/val/test split over the staged per-case files.

    Every scenario draws its perturbations from the same distribution, so a single
    shuffle is a representative split and no stratification is needed. Files are sorted
    before shuffling because glob order is filesystem-dependent, and the shuffle is
    seeded from the config so the held-out sets are reproducible and auditable.
    """
    fs = sorted(feasible_files, key=str)
    random.Random(split_seed).shuffle(fs)
    return {"test": fs[:n_test],
            "val": fs[n_test:n_test + n_val],
            "train": fs[n_test + n_val:]}


def write_jsonl(files: list[str], path: str, append: bool = False) -> tuple[int, int]:
    """Consolidate per-case .pyg.json files into one .jsonl (one scenario per line).

    Cases are staged as individual files during generation, which is what makes a run
    resumable, and consolidated here. Compact separators roughly halve the size versus
    the pretty-printed staging files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "a" if append else "w") as out:
        for f in files:
            obj = json.load(open(f))
            out.write(json.dumps(obj, separators=(",", ":")) + "\n")
            n += 1
    return n, os.path.getsize(path)


def take(feasible_files: list[str], total: int) -> list[str]:
    """Take exactly `total` feasible cases, in name order.

    Generation overshoots by whatever was in flight when the target was reached, so
    trimming here makes --total_num_feasible exact in the published dataset rather than
    "whatever the last batch happened to produce".

    Name order puts `base_unperturbed` first, so the unperturbed operating point is
    always one of the published cases.
    """
    return sorted(feasible_files)[:total]
