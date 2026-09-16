#!/usr/bin/env python3
"""Fine-tune the GridSFM control model on a generated dataset.

Reads THREE separate files -- train_data.jsonl, val_data.jsonl, test_data.jsonl --
exactly as generate_finetune_dataset published them. It does not re-split, re-shuffle
or merge them: the split is already fixed, stratified by perturbation mode, and
recorded in splits.json. Val chooses the epoch; test is read once at the very end.

    python scripts/finetune.py --config config/case500_goc_finetune.yaml
      [--epochs N] [--batch N] [--lr X] [--select_on val|final|train_loss] [--seed N]
"""
import argparse, os, sys
import _bootstrap  # noqa: F401
import config as configmod
import cache as cachemod
import train as trainmod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--min_epochs", type=int,
                    help="overrides train.min_epochs: early stopping is INERT below this "
                         "epoch. Patience alone cannot express 'do not ask yet' -- on a "
                         "fast-converging grid val stops improving long before the model "
                         "does (case500_goc stopped at 54-68 epochs, best at 34-48).")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--select_on", choices=("val", "final", "train_loss"))
    ap.add_argument("--device")
    ap.add_argument("--train_subset", type=int,
                    help="train on the first N cases of a stratified NESTED ordering "
                         "(0 = all). Drives the data-scaling sweep.")
    ap.add_argument("--run_label", help="names results/logs/checkpoints subdir "
                                        "(default: derived from --train_subset)")
    ap.add_argument("--plot_every", type=int,
                    help="overrides logging.plot_every_epochs. At high epoch counts a "
                         "per-epoch matplotlib render costs more than the training does.")
    ap.add_argument("--data_root", help="overrides data.root (dataset location)")
    ap.add_argument("--data_dir", help="overrides data.dir (this dataset's directory)")
    ap.add_argument("--w_control", type=float,
                    help="overrides loss.w_control (GT/RMSE anchor weight, elastic mode)")
    ap.add_argument("--w_control_start", type=float,
                    help="overrides loss.w_control_start (elastic): heavy early anchor "
                         "weight that pulls controls into the PF-solvable region")
    ap.add_argument("--w_control_end", type=float,
                    help="overrides loss.w_control_end (elastic): balanced end anchor "
                         "weight (RMSE ~ elastic ~ cost)")
    ap.add_argument("--w_control_epochs", type=int,
                    help="overrides loss.w_control_epochs (elastic): epochs to anneal "
                         "w_control_start -> w_control_end")
    ap.add_argument("--closure_solver", choices=("superlu", "cudss"),
                    help="overrides closure.solver (elastic mode): superlu=CPU SuperLU "
                         "(parallel over closure.workers), cudss=GPU cuDSS (serial)")
    a = ap.parse_args()

    _pre = {}
    if a.train_subset is not None:
        _pre["train_subset"] = a.train_subset
    if a.run_label:
        _pre["run_label"] = a.run_label
    cfg = configmod.load_config(a.config, data_root=a.data_root, data_dir=a.data_dir,
                                overrides=_pre)
    for k, dst in (("epochs", "epochs"), ("min_epochs", "min_epochs"),
                   ("batch", "batch"), ("accum", "accum"),
                   ("lr", "lr"), ("seed", "seed"), ("select_on", "select_on")):
        v = getattr(a, k)
        if v is not None:
            cfg["train"][dst] = v
    if a.device:
        cfg["runtime"]["device"] = a.device
    if a.plot_every:
        cfg["logging"]["plot_every_epochs"] = a.plot_every
    if a.w_control is not None:
        cfg["loss"]["w_control"] = a.w_control
    if a.w_control_start is not None:
        cfg["loss"]["w_control_start"] = a.w_control_start
    if a.w_control_end is not None:
        cfg["loss"]["w_control_end"] = a.w_control_end
    if a.w_control_epochs is not None:
        cfg["loss"]["w_control_epochs"] = a.w_control_epochs
    if a.closure_solver is not None:
        cfg.setdefault("closure", {})["solver"] = a.closure_solver
    for w in configmod.validate(cfg, require_data=True):
        print(f"WARNING: {w}")

    man = cachemod.load_manifest(cfg)
    st = man["stats"]
    cfg = configmod.derive(cfg, st["mean_scored_gens"], st["mean_scored_vctrl"],
                           st["bytes_per_case"])
    os.makedirs(cfg["_meta"]["results_dir"], exist_ok=True)
    configmod.dump_resolved(cfg, cfg["_meta"]["results_dir"])
    trainmod.run(cfg, man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
