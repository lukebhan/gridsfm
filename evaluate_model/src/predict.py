from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]


def _public_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "finetune_model/src" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model(run, checkpoint, base_checkpoint, device, torch):
    from gridsfm.checkpoint import load_model
    from gridsfm.model import GridTransformerBackbone

    if run == "base":
        model = load_model(checkpoint, device)
    elif run == "n1000_scratch":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        hidden = int(state["blocks.0.attn_ln.bus.weight"].shape[0])
        depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
        model = GridTransformerBackbone(hidden_dim=hidden, num_blocks=depth).to(device)
        model.load_state_dict(state)
    else:
        model = load_model(base_checkpoint, device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.clip_mode = "sigmoid"
    return model.eval()


def _complete(graph, prediction, closure, np, tol, settings):
    case = closure.extract_case(graph)
    pg = np.asarray(prediction["pg"], dtype=np.float64)
    vm = np.asarray(prediction["vm"], dtype=np.float64)
    initial = closure.dc_init_V(case, pg, vm)
    result = closure.newton_pf(case, closure.build_Ssp(case, pg), initial,
                               tol=tol, max_iter=settings["completion"]["max_iter"])
    prediction["completion_converged"] = bool(result.converged)
    prediction["completion_iters"] = int(result.iters)
    if not result.converged:
        prediction["va"] = np.angle(initial).tolist()
        prediction["dc_fallback"] = True
        return
    voltage = result.V
    reactive_bus = (voltage * np.conj(case.Ybus @ voltage)).imag + case.Qd
    weights = np.maximum(case.Qmax - case.Qmin, 0.0) * case.gen_online + settings["completion"]["reactive_weight_epsilon"]
    bus_weights = np.zeros(case.n_bus)
    np.add.at(bus_weights, case.gen2bus, weights)
    reactive = np.where(case.gen_online,
                        reactive_bus[case.gen2bus] * weights / bus_weights[case.gen2bus], 0.0)
    prediction.update(vm=np.abs(voltage).tolist(), va=np.angle(voltage).tolist(),
                      qg=reactive.tolist())


def generate_starts(grid, split_path, selected, checkpoint_root, output_dir,
                    settings, *, device, runs):
    import numpy as np
    import torch

    if device == "cpu":
        torch.set_num_threads(settings["runtime"]["inference_threads"])
    batch_size = settings["runtime"]["batch_size"]
    model_names = settings["evaluation"]["models"]
    sys.path.insert(0, str(REPO / "model"))
    from gridsfm.data import batch_data_list, drop_offline_rows, prepare_for_inference
    dataset = _public_module("_table1_dataset", "dataset.py")
    closure = _public_module("_table1_closure", "closure.py")
    split_path = Path(split_path).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not runs or any(run not in model_names for run in runs) or len(set(runs)) != len(runs):
        raise ValueError(f"runs must be unique members of {model_names}")
    if batch_size < 1 or not selected:
        raise ValueError("batch_size and selected case count must be positive")
    wanted = {int(r["case_idx"]): r for r in selected}
    if len(wanted) != len(selected):
        raise ValueError("duplicate case indices")
    base = checkpoint_root / settings["paths"]["base_checkpoint"]
    checkpoints = {run: base if run == "base" else checkpoint_root / settings["paths"]["model_checkpoint"].format(grid=grid, run=run)
                   for run in runs}
    required = set(checkpoints.values()) | ({base} if "n0100" in runs else set())
    for checkpoint in required:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        if (output_dir / f"{run}.jsonl").exists():
            raise FileExistsError(output_dir / f"{run}.jsonl")

    cases = []
    for index, obj in dataset.iter_records(str(split_path)):
        if index not in wanted:
            continue
        metadata = obj.get("metadata", {})
        identity = wanted[index]
        if (str(metadata.get("scenario_id")) != str(identity["scenario_id"]) or
                metadata.get("perturb_mode") != identity["mode"]):
            raise ValueError(f"test row {index} has a different scenario identity")
        graph = dataset.load_obj(obj, tag=f"{split_path.name}:{index}")
        n_full = graph["generator"].x.size(0)
        kept = drop_offline_rows(graph).get("generator")
        graph = prepare_for_inference(graph)
        cases.append((index, graph, metadata, kept, n_full))
        if len(cases) == len(wanted):
            break
    if len(cases) != len(wanted):
        raise ValueError("test split does not contain every selected case")
    tol = settings["grids"][grid]["completion_tol"]
    report = {"grid": grid, "runs": {}}
    for run in runs:
        print(f"Predicting {grid}/{run}: {len(cases)} cases, batch={batch_size}, device={device}", flush=True)
        model = _model(run, checkpoints[run], base, device, torch)
        records = []
        with torch.no_grad():
            for start in range(0, len(cases), batch_size):
                chunk = cases[start:start + batch_size]
                batch = batch_data_list([case[1].clone() for case in chunk]).to(device)
                predicted = model(batch)
                pg = predicted["generator"].pred[:, 0].double().cpu()
                vm = predicted["bus"].pred[:, 1].double().cpu()
                gen_batch = batch["generator"].batch.cpu()
                bus_batch = batch["bus"].batch.cpu()
                for position, (index, graph, metadata, kept, n_full) in enumerate(chunk):
                    record = {"i": index, "mode": metadata["perturb_mode"],
                              "scenario_id": metadata["scenario_id"],
                              "pg": pg[gen_batch == position].tolist(),
                              "vm": vm[bus_batch == position].tolist()}
                    _complete(graph, record, closure, np, tol, settings)
                    if kept is not None:
                        for key in ("pg", "qg"):
                            if key in record:
                                full = [0.0] * n_full
                                for row, original in enumerate(kept.tolist()):
                                    full[original] = record[key][row]
                                record[key] = full
                    records.append(record)
        path = output_dir / f"{run}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(path)
        failures = [r["i"] for r in records if not r["completion_converged"]]
        report["runs"][run] = {
            "path": str(path), "n": len(records),
            "completion": {"n_converged": len(records) - len(failures),
                           "failed_case_indices": failures, "tol": tol, "max_iter": settings["completion"]["max_iter"]},
        }
        print(f"  Newton completion: {len(records) - len(failures)}/{len(records)}", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return report
