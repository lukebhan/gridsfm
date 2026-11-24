from collections import defaultdict
import csv
import json
import math
import random
import statistics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see the repository README for downloads")


def select_cases(split, settings):
    n_cases, seed = settings["evaluation"]["n_cases"], settings["evaluation"]["seed"]
    groups = defaultdict(list)
    offsets = []
    with split.open("rb") as stream:
        for index, line in enumerate(stream):
            offsets.append(stream.tell() - len(line))
            if not line.strip():
                continue
            md = json.loads(line)["metadata"]
            mode = md.get("perturb_mode")
            groups[mode or "?"].append({"case_idx": index,
                "scenario_id": str(md.get("scenario_id", "")), "mode": mode})
    total = sum(map(len, groups.values()))
    if total < n_cases:
        raise ValueError(f"{split} contains {total} scenarios; the configured draw requires at least {n_cases}")
    exact = {mode: n_cases * len(rows) / total for mode, rows in groups.items()}
    quota = {mode: int(value) for mode, value in exact.items()}
    order = sorted(groups, key=lambda mode: (-(exact[mode] - quota[mode]), mode))
    for mode in order[:n_cases - sum(quota.values())]:
        quota[mode] += 1
    selected = []
    for mode, rows in groups.items():
        rows.sort(key=lambda row: (str(row["mode"]), int(row["scenario_id"] or 0)))
        random.Random(f"{seed}|{mode}").shuffle(rows)
        selected.extend(rows[:quota[mode]])
    return sorted(selected, key=lambda row: row["case_idx"]), offsets


def arm_definitions(models):
    arms = [("cold", "flat", "cost", None), ("dc", "dc", "cost", None)]
    for objective in ("cost", "dist_ctrl"):
        for run in models:
            arm = ("base" if objective == "cost" else "base_proj") if run == "base" else \
                  ("ft:" if objective == "cost" else "ft_proj:") + run
            arms.append((arm, "model", objective, run))
    return arms


def summarize(grid, rows, arms, settings):
    solved = set(settings["evaluation"]["solved_statuses"])
    cold = {r["scenario_id"]: int(r["iterations"]) for r in rows
            if r["arm"] == "cold" and r["status"] in solved}
    records = []
    for arm, *_ in arms:
        attempted = [r for r in rows if r["arm"] == arm]
        ok = [r for r in attempted if r["status"] in solved]
        iterations = sorted(int(r["iterations"]) for r in ok)
        need = math.ceil(0.9 * len(attempted))
        ratios = [cold[r["scenario_id"]] / int(r["iterations"]) for r in ok
                  if r["scenario_id"] in cold and int(r["iterations"]) > 0]
        gaps = [100 * (float(r["gen_cost"]) - float(r["gt_objective"])) / abs(float(r["gt_objective"]))
                for r in ok if float(r["gt_objective"]) != 0]
        scope = "smoke_subset"
        if len(attempted) == settings["evaluation"]["n_cases"]:
            scope = "table_100" if len(attempted) == 100 and settings["evaluation"]["seed"] == 0 else "configured_sample"
        records.append({
            "grid": grid, "arm": arm, "attempted": len(attempted), "converged": len(ok),
            "p90_iterations": iterations[need - 1] if need and len(iterations) >= need else None,
            "speedup": statistics.median(ratios) if ratios else None,
            "paired_cases": len(ratios),
            "convergence_pct": 100 * len(ok) / len(attempted),
            "cost_gap_pct": max(0.0, statistics.median(gaps)) if gaps else None,
            "scope": scope,
        })
    return records


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


