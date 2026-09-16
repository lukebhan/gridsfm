"""Config loading: `extends` resolution, validation, and the derived-value pass.

Two rules this module exists to enforce:
  1. train and test are SEPARATE inputs, read as given. Nothing here merges,
     shuffles or re-splits them -- generate_finetune_dataset already fixed the
     split. See README "The split is not ours to make".
  2. Grid-dependent numbers (top-k widths, LRU capacity) are DERIVED from the
     dataset, never hardcoded per grid. Hardcoding is what let the 10k run use
     KV=44 against a control-bus count that only scored ~1255.
"""
from __future__ import annotations
import os, copy, hashlib, json
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # finetune_model/


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(path: str, rel_to: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(rel_to, path))


def _load_raw(path: str) -> dict:
    """Merge a config over whatever it `extends`. No path resolution here -- a parent
    like defaults.yaml has no grid_id, so resolution can only run once, on the merged
    result."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_load_raw(_resolve(parent, os.path.dirname(path))), cfg)
    return cfg


def load_config(path: str, data_root: str | None = None,
                data_dir: str | None = None,
                overrides: dict | None = None) -> dict:
    """Load a config, apply `extends`, then resolve paths and derived locations.

    data_root / data_dir come from the CLI and win over the file, so a dataset that has
    moved does not require editing any config.
    """
    path = os.path.abspath(path)
    cfg = _load_raw(path)
    if data_root:
        cfg.setdefault("data", {})["root"] = data_root
        cfg["data"]["dir"] = None
    if data_dir:
        cfg.setdefault("data", {})["dir"] = data_dir
    # applied BEFORE the run label is derived, so --train_subset names the run
    for k, v in (overrides or {}).items():
        if k == "run_label":
            cfg["run_label"] = v
        else:
            cfg.setdefault("data", {})[k] = v
    if "grid_id" not in cfg:
        raise ValueError(f"{path}: grid_id is required (is this a defaults file?)")

    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = path
    cfg["_meta"]["config_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in cfg.items() if k != "_meta"}, sort_keys=True,
                   default=str).encode()).hexdigest()[:16]

    # Split locations resolve in two steps: `dir` (or <root>/<grid_id>) locates the
    # dataset, then each split name resolves inside it. A split value containing a path
    # separator is treated as a path relative to finetune_model/ instead, so an odd
    # layout is still expressible. Paths are relative to finetune_model/, not to the
    # config dir, so `config/x.yaml` and `finetune_model/config/x.yaml` agree.
    d = cfg["data"]
    if d.get("dir"):
        ddir = _resolve(d["dir"], ROOT)
    elif d.get("root"):
        ddir = os.path.join(_resolve(d["root"], ROOT), cfg["grid_id"])
    else:
        ddir = ROOT
    d["dir"] = ddir
    for key in ("train", "val", "test"):
        v = d.get(key)
        if v:
            d[key] = _resolve(v, ROOT) if os.sep in v else os.path.join(ddir, v)
    cfg["base_checkpoint"] = _resolve(cfg["base_checkpoint"], ROOT)
    if not cfg["cache"].get("dir"):
        cfg["cache"]["dir"] = os.path.join(ROOT, "cache", cfg["grid_id"])
    else:
        cfg["cache"]["dir"] = _resolve(cfg["cache"]["dir"], ROOT)

    # Every output is scoped by run label so a data-scaling sweep writes N independent
    # result sets instead of overwriting one.
    import sys as _sys
    _sys.path.insert(0, HERE)
    from subset import subset_label
    label = cfg.get("run_label") or subset_label(int(cfg["data"].get("train_subset", 0)))
    cfg["run_label"] = label
    cfg["_meta"]["run_label"] = label
    cfg["_meta"]["results_dir"] = os.path.join(ROOT, "results", cfg["grid_id"], label)
    cfg["_meta"]["logs_dir"] = os.path.join(ROOT, "logs", cfg["grid_id"], label)
    cfg["_meta"]["ckpt_dir"] = os.path.join(ROOT, "checkpoints", cfg["grid_id"], label)
    return cfg


def validate(cfg: dict, require_data: bool = True) -> list[str]:
    """Hard-error on anything that would produce a wrong result; warn on the rest."""
    warn: list[str] = []
    if not cfg.get("grid_id"):
        raise ValueError("config: grid_id is required")

    for key in ("train", "val", "test"):
        p = cfg["data"].get(key)
        if not p:
            raise ValueError(f"config: data.{key} is required "
                             "(train, val and test are three separate files)")
        if require_data and not os.path.exists(p):
            raise FileNotFoundError(f"data.{key} not found: {p}")
    if not isinstance(cfg["data"].get("drop_offline", True), bool):
        raise ValueError(f"data.drop_offline must be a bool, got "
                         f"{cfg['data'].get('drop_offline')!r}")
    if require_data:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if os.path.samefile(cfg["data"][a], cfg["data"][b]):
                raise ValueError(f"data.{a} and data.{b} are the same file -- "
                                 "the splits would not be independent")

    if require_data and not os.path.exists(cfg["base_checkpoint"]):
        raise FileNotFoundError(f"base_checkpoint not found: {cfg['base_checkpoint']}")

    mode = str(cfg["loss"].get("mode", "control")).lower()
    if mode not in ("control", "elastic"):
        raise ValueError(f"loss.mode must be control|elastic, got {mode!r}")

    t = cfg["train"]
    # Top-k only exists in the max-aligned control loss. The elastic objective
    # have no top-k term at all, so requiring segmented_topk there would reject a
    # perfectly well-defined run over a key it never reads.
    if mode == "control" and t["batch"] > 1 and not cfg["loss"]["segmented_topk"]:
        raise ValueError(
            f"train.batch={t['batch']} with loss.segmented_topk=false: top-k would span "
            "cases and stop tracking the worst generator (the whole point of the loss). "
            "Set loss.segmented_topk: true, or keep batch: 1.")
    if t["select_on"] not in ("val", "final", "train_loss"):
        raise ValueError(f"train.select_on must be val|final|train_loss, got {t['select_on']!r}")
    if t["epochs"] < 1:
        raise ValueError("train.epochs must be >= 1")

    if mode == "control" and cfg["loss"]["segmented_topk"] and t["batch"] == 1:
        warn.append("loss.segmented_topk has no effect at train.batch=1 (identical result).")

    # ---- optimizer / scheduler ----
    if str(t["optimizer"]).lower() not in ("adam", "adamw", "sgd"):
        raise ValueError(f"train.optimizer must be adam|adamw|sgd, got {t['optimizer']!r}")
    if str(t["scheduler"]).lower() not in ("cosine", "step", "none"):
        raise ValueError(f"train.scheduler must be cosine|step|none, got {t['scheduler']!r}")
    if len(t["betas"]) != 2 or not all(0 <= b < 1 for b in t["betas"]):
        raise ValueError(f"train.betas must be two values in [0,1), got {t['betas']}")
    if t["lr"] <= 0:
        raise ValueError(f"train.lr must be > 0, got {t['lr']}")
    if t["accum"] < 1:
        raise ValueError(f"train.accum must be >= 1, got {t['accum']}")
    if not 0 <= t["lr_min_frac"] <= 1:
        raise ValueError(f"train.lr_min_frac must be in [0,1], got {t['lr_min_frac']}")
    if t["warmup_epochs"] >= t["epochs"]:
        raise ValueError(f"train.warmup_epochs={t['warmup_epochs']} leaves no post-warmup "
                         f"epochs out of {t['epochs']}")
    if t["weight_decay"] and str(t["optimizer"]).lower() == "adam":
        warn.append("train.weight_decay with optimizer=adam applies COUPLED L2, not "
                    "decoupled decay; use adamw if you want decoupled.")
    if t["clip_grad"] == 0:
        warn.append("train.clip_grad=0 disables clipping; pre-clip norms on this loss "
                    "run in the hundreds to thousands, so expect instability.")
    if t["early_stop_patience"] and t["select_on"] != "val":
        warn.append(f"train.early_stop_patience is measured on the val score but "
                    f"select_on={t['select_on']!r}; stopping will not match selection.")

    # ---- loss ----
    L = cfg["loss"]
    if not (L["mean_weight"] or L["topk_weight"] or L["max_weight"]):
        raise ValueError("loss: mean_weight, topk_weight and max_weight are all 0 -- "
                         "there is no loss left to minimise")
    if not (L["w_pg"] or L["w_v"]):
        raise ValueError("loss: w_pg and w_v are both 0 -- nothing is being trained")
    if mode == "control" and not L["topk_weight"] and not L["max_weight"]:
        warn.append("loss.topk_weight and loss.max_weight are both 0: the loss is a "
                    "plain mean and is NO LONGER max-aligned. The worst control error "
                    "is what drives corrector cost, so expect worse warm starts.")
    for k in ("tol_pg", "tol_v"):
        if L[k] <= 0:
            raise ValueError(f"loss.{k} must be > 0, got {L[k]}")
    for k in ("topk_pg_frac", "topk_v_frac"):
        if not 0 < L[k] <= 1:
            raise ValueError(f"loss.{k} must be in (0,1], got {L[k]}")
    if not L["w_v"]:
        warn.append("loss.w_v=0: V setpoints are not trained; the warm start will only "
                    "carry Pg.")

    # ---- model / eval / logging ----
    if str(cfg["model"]["freeze"]).lower() not in ("none", "trunk"):
        raise ValueError(f"model.freeze must be none|trunk, got {cfg['model']['freeze']!r}")
    if cfg["model"]["freeze"] != "none":
        warn.append(f"model.freeze={cfg['model']['freeze']}: the validated runs train the "
                    "WHOLE backbone; expect a materially weaker fit.")
    for k in ("batch", "test_batch"):
        if cfg["eval"][k] < 1:
            raise ValueError(f"eval.{k} must be >= 1, got {cfg['eval'][k]}")
    if not (cfg["eval"]["val_score_pg_weight"] or cfg["eval"]["val_score_v_weight"]):
        raise ValueError("eval: both val_score weights are 0 -- selection would be blind")
    for k in ("step_log_every", "plot_every_epochs"):
        if cfg["logging"][k] < 1:
            raise ValueError(f"logging.{k} must be >= 1, got {cfg['logging'][k]}")
    if cfg["logging"]["save_every_epochs"] < 0:
        raise ValueError("logging.save_every_epochs must be >= 0 (0 = off)")
    if cfg["cache"]["workers"] < 1:
        raise ValueError("cache.workers must be >= 1")
    if cfg["cache"]["mem_budget_gb"] <= 0:
        raise ValueError("cache.mem_budget_gb must be > 0")
    ts = int(cfg["data"].get("train_subset", 0))
    if ts < 0:
        raise ValueError("data.train_subset must be >= 0 (0 = use all)")
    if 0 < ts < 10:
        warn.append(f"data.train_subset={ts} is fewer cases than perturbation modes x 2; "
                    "the stratified subset cannot represent every mode evenly.")
    return warn


def derive(cfg: dict, mean_scored_gens: float, mean_scored_vctrl: float,
           bytes_per_case: float) -> dict:
    """Fill the runtime-derived values from what the cache actually measured."""
    L = cfg["loss"]
    kp = L["topk_pg"] or max(1, round(L["topk_pg_frac"] * mean_scored_gens))
    kv = L["topk_v"] or max(1, round(L["topk_v_frac"] * mean_scored_vctrl))
    cap = max(16, int(cfg["cache"]["mem_budget_gb"] * 2**30 / max(bytes_per_case, 1)))
    cfg["_derived"] = dict(
        topk_pg=int(kp), topk_v=int(kv), lru_cases=cap,
        mean_scored_gens=round(mean_scored_gens, 1),
        mean_scored_vctrl=round(mean_scored_vctrl, 1),
        bytes_per_case=int(bytes_per_case),
        topk_pg_pct=round(100 * kp / max(mean_scored_gens, 1), 2),
        topk_v_pct=round(100 * kv / max(mean_scored_vctrl, 1), 2),
    )
    return cfg


def dump_resolved(cfg: dict, outdir: str) -> str:
    """Write the fully resolved config beside the run's results.

    This is the ONLY record of what a run actually used once CLI overrides are folded in,
    and every audit of the shipped checkpoints has gone through it -- it is what showed
    that the n0010..n0500 runs ramped w_control to 300 while no config file said so.
    """
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "resolved_config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return p
