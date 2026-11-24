import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from common import arm_definitions, require_file, select_cases, summarize, write_csv, write_json
from predict import generate_starts


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=HERE.parent / "config/evaluation_gpu.yaml")
    initial, _ = bootstrap.parse_known_args()
    settings = yaml.safe_load(initial.config.read_text())
    defaults, paths = settings["runtime"], settings["paths"]
    grids, model_names = settings["grids"], tuple(settings["evaluation"]["models"])
    n_cases = settings["evaluation"]["n_cases"]
    parser = argparse.ArgumentParser(description="Reproduce the cold/DC/GridSFM rows of GPU Table VI on its exact 100 test cases.")
    parser.add_argument("--config", type=Path, default=initial.config, help="GPU evaluation settings YAML")
    parser.add_argument("--grid", choices=["all", *grids], default=defaults["grid"])
    parser.add_argument("--models", default=defaults["models"], help="all, none, or comma-separated " + ",".join(model_names))
    parser.add_argument("--data-root", type=Path, default=REPO / paths["data_root"])
    parser.add_argument("--checkpoint-root", type=Path, default=REPO / paths["checkpoint_root"])
    parser.add_argument("--device", default=defaults["device"], help="model inference device, independent of the solver GPUs selected by --gpus")
    parser.add_argument("--limit", type=int, default=0, help=f"smoke subset of the configured draw; 0 uses all {n_cases}")
    parser.add_argument("--case-index", type=int, help="smoke test one original zero-based test row from the draw")
    parser.add_argument("--gpus", default=defaults["gpus"], help="comma-separated physical GPU indices; one sequential Julia worker per device")
    parser.add_argument("--julia", default=os.environ.get("JULIA_BIN", defaults["julia"]))
    parser.add_argument("--out", type=Path, help=f"new output directory; default {paths['results_root']}/<UTC timestamp>")
    args = parser.parse_args()
    models = model_names if args.models == "all" else () if args.models == "none" else tuple(args.models.split(","))
    if len(set(models)) != len(models) or any(m not in model_names for m in models):
        parser.error("models must be all, none, or unique members of " + ",".join(model_names))
    gpus = tuple(gpu.strip() for gpu in str(args.gpus).split(","))
    if not gpus or any(not gpu.isdecimal() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("gpus must contain unique comma-separated physical GPU indices")
    if n_cases < 1 or not 0 <= args.limit <= n_cases:
        parser.error(f"case count must be positive; limit must be 0..{n_cases}")
    if defaults["batch_size"] < 1 or defaults["threads_per_worker"] < 1 or defaults["inference_threads"] < 1:
        parser.error("batch size and thread counts must be positive")
    if args.case_index is not None and args.limit:
        parser.error("choose either --case-index or --limit")
    julia = shutil.which(args.julia)
    if julia is None:
        parser.error("Julia not found; pass --julia /path/to/julia or set JULIA_BIN")
    out = (args.out or REPO / paths["results_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if out.exists():
        parser.error("output directory already exists; choose a new --out")
    args.data_root = args.data_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    paths.update(data_root=str(args.data_root), checkpoint_root=str(args.checkpoint_root))
    defaults.update(grid=args.grid, models=args.models, device=args.device,
                    gpus=",".join(gpus), limit=args.limit, julia=julia)
    out.mkdir(parents=True)
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "JULIA_NUM_THREADS"):
        env[variable] = str(defaults["threads_per_worker"])
    base_command = [julia, f"--project={REPO / paths['julia_project']}", str(SRC / "solve_gpu.jl")]
    selected_grids = list(grids) if args.grid == "all" else [args.grid]
    manifest = {"models": list(models), "eval_seed": settings["evaluation"]["seed"], "n_eval": n_cases,
                "config_file": str(args.config.resolve()), "settings": settings,
                "device": args.device, "batch_size": defaults["batch_size"], "gpus": list(gpus),
                "workers": len(gpus), "timing_mode": "sequential_per_gpu", "grids": {}}
    summaries = []
    for grid in selected_grids:
        print(f"Selecting test scenarios: {grid}", flush=True)
        split = (args.data_root / grid / "test_data.jsonl").resolve()
        require_file(split)
        required = [paths["model_checkpoint"].format(grid=grid, run=m) for m in models if m != "base"]
        if "base" in models or "n0100" in models:
            required.append(paths["base_checkpoint"])
        for name in required:
            require_file(args.checkpoint_root / name)
        selected, offsets = select_cases(split, settings)
        if args.case_index is not None:
            selected = [r for r in selected if r["case_idx"] == args.case_index]
            if not selected:
                raise ValueError(f"{grid}: requested case is outside the published 100-case selection")
        elif args.limit:
            selected = selected[:args.limit]
        topology = REPO / paths["topology_root"] / grids[grid]["topology"]
        require_file(topology)
        folder = out / grid
        folder.mkdir()
        record = {"test_data": str(split), "selected": selected}
        manifest["grids"][grid] = record
        write_json(out / "run.json", manifest)
        grid_settings = copy.deepcopy(settings)
        grid_settings["madnlp"].update(grids[grid].get("madnlp", {}))
        predictions = folder / "predictions"
        if models:
            record["prediction_manifest"] = generate_starts(
                grid, split, selected, args.checkpoint_root, predictions, grid_settings,
                device=args.device, runs=models)
            write_json(out / "run.json", manifest)
        arms = arm_definitions(models)
        tasks = [{"arm": arm, "start": start, "objective": objective, "run": run,
                  "case_idx": case["case_idx"]} for arm, start, objective, run in arms for case in selected]
        cfg = folder / "solver_config.json"
        write_json(cfg, {"grid_id": grid, "topology": str(topology), "split_jsonl": str(split),
                         "offsets": offsets, "pred_dir": str(predictions), "settings": grid_settings})
        processes, logs = [], []
        n_workers = min(len(gpus), len(tasks))
        print(f"Solving {grid}: {len(selected)} scenarios, {len(arms)} arms, {n_workers} worker(s); logs: {folder}", flush=True)
        try:
            for worker, gpu in enumerate(gpus[:n_workers]):
                shard = folder / f"tasks_{worker}.jsonl"
                shard.write_text("".join(json.dumps(t) + "\n" for t in tasks[worker::n_workers]))
                log = (folder / f"worker_{worker}.log").open("w")
                logs.append(log)
                worker_env = env.copy()
                worker_env["CUDA_VISIBLE_DEVICES"] = gpu
                processes.append(subprocess.Popen([*base_command, str(cfg), str(shard), str(folder / f"worker_{worker}.csv")],
                                                  env=worker_env, stdout=log, stderr=subprocess.STDOUT))
            codes = [p.wait() for p in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for log in logs:
                log.close()
        rows, missing_csv = [], []
        for worker in range(n_workers):
            path = folder / f"worker_{worker}.csv"
            if not path.is_file():
                missing_csv.append(path.name)
                continue
            with path.open() as stream:
                rows.extend(csv.DictReader(stream))
        rows.sort(key=lambda r: (r["arm"], int(r["case_idx"])))
        if rows:
            write_csv(folder / "results.csv", rows)
        expected_keys = {(t["arm"], t["case_idx"]) for t in tasks}
        actual_keys = {(r["arm"], int(r["case_idx"])) for r in rows}
        if missing_csv or any(codes) or len(rows) != len(tasks) or actual_keys != expected_keys or any(r["status"] == "ERROR" for r in rows):
            raise RuntimeError(f"Incomplete evaluation of {grid}; inspect {folder}/worker_*.log and results.csv")
        summaries.extend(summarize(grid, rows, arms, settings))
        write_csv(out / "summary.csv", summaries)
        print(f"Finished {grid}; {len(selected)} selected scenario(s)", flush=True)
    write_json(out / "run.json", manifest)
    print(f"Wrote {out}/summary.csv")


if __name__ == "__main__":
    main()
