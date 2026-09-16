"""Config loading and validation for dataset generation."""
from __future__ import annotations
import os, yaml
from gridstats import read_matpower, validate_perturbations

REQUIRED_PERTURB = ("mode_probs", "load_sf_lo", "load_sf_hi", "load_jitter", "cost_frac",
                    "costs_min_pool", "killgen_nk", "killgen_probs", "killgen_min_online",
                    "killgen_pmax_threshold", "derate_frac", "derate_lo", "derate_hi",
                    "vsqueeze_frac", "vsqueeze_delta")
KNOWN_MODES = ("loads", "costs", "killgen", "derate", "vsqueeze")


def load_config(path: str, root: str | None = None) -> dict:
    """Read a config, resolve the topology path, and validate it against the grid.

    Fails loudly on anything that would produce a silently degraded dataset: a missing
    topology, an unknown mode, or a load range the grid cannot serve.
    """
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(path)))
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ("grid_id", "topology", "seeds", "perturbations"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    for key in ("master", "split"):
        if key not in cfg["seeds"]:
            raise ValueError(f"{path}: seeds.{key} is required for reproducibility")
        if not isinstance(cfg["seeds"][key], int):
            raise ValueError(f"{path}: seeds.{key} must be an integer, got {cfg['seeds'][key]!r}")

    p = cfg["perturbations"]
    for key in REQUIRED_PERTURB:
        if key not in p:
            raise ValueError(f"{path}: perturbations.{key} is required")

    probs = p["mode_probs"]
    if not isinstance(probs, dict) or not probs:
        raise ValueError(f"{path}: perturbations.mode_probs must be a non-empty mapping "
                         f"of mode name -> probability")
    unknown = set(probs) - set(KNOWN_MODES)
    if unknown:
        raise ValueError(f"{path}: unknown modes {sorted(unknown)}; known: {list(KNOWN_MODES)}")
    for m, pr in probs.items():
        if not isinstance(pr, (int, float)) or not 0.0 <= float(pr) <= 1.0:
            raise ValueError(f"{path}: mode_probs.{m}={pr!r} must be a number in [0, 1]")
    if not any(float(v) > 0 for v in probs.values()):
        raise ValueError(f"{path}: every mode_probs entry is 0; no perturbation would be applied")
    # `perturb_param` -- the severity scalar every record carries -- is the system load
    # factor, which is only defined when `loads` fires on every scenario.
    if float(probs.get("loads", 0.0)) < 1.0:
        raise ValueError(f"{path}: mode_probs.loads must be 1.0; the load factor is the "
                         f"severity scalar recorded for every scenario")

    if not isinstance(p.get("allow_load_sf_above_ceiling", False), bool):
        raise ValueError(f"{path}: perturbations.allow_load_sf_above_ceiling must be a bool")
    if p["load_sf_lo"] >= p["load_sf_hi"]:
        raise ValueError(f"{path}: load_sf_lo must be < load_sf_hi")
    if len(p["killgen_nk"]) != len(p["killgen_probs"]):
        raise ValueError(f"{path}: killgen_nk ({len(p['killgen_nk'])}) and killgen_probs "
                         f"({len(p['killgen_probs'])}) must be the same length")
    ps = sum(p["killgen_probs"])
    if abs(ps - 1.0) > 1e-6:
        raise ValueError(f"{path}: killgen_probs must sum to 1.0, got {ps:.4f}")
    if any(x < 0 for x in p["killgen_probs"]):
        raise ValueError(f"{path}: killgen_probs must be non-negative")
    if p["killgen_min_online"] < 1:
        raise ValueError(f"{path}: killgen_min_online must be >= 1")
    if p["costs_min_pool"] < 2:
        raise ValueError(f"{path}: costs_min_pool must be >= 2 (a shuffle needs two curves)")

    topo = cfg["topology"]
    if not os.path.isabs(topo):
        topo = os.path.join(root, topo)
    if not os.path.isfile(topo):
        raise FileNotFoundError(f"{path}: topology not found: {topo}")
    cfg["_topology_abs"] = topo

    cfg["solver"] = {**{"max_iters": 3000, "tol": 1e-6, "acceptable_tol": 1e-4},
                     **cfg.get("solver", {})}

    sp = cfg.setdefault("splits", {"train": 0.75, "val": 0.07, "test": 0.18})
    for k in ("train", "val", "test"):
        if k not in sp:
            raise ValueError(f"{path}: splits.{k} is required")
        if not 0 <= sp[k] <= 1:
            raise ValueError(f"{path}: splits.{k}={sp[k]} must be in [0,1]")
    tot = sum(sp[k] for k in ("train", "val", "test"))
    if abs(tot - 1.0) > 1e-6:
        raise ValueError(f"{path}: splits must sum to 1.0, got {tot:.4f}")

    stats = read_matpower(topo, cfg["grid_id"])
    cfg["_stats"] = stats
    cfg["_warnings"] = validate_perturbations(stats, p, strict=True)
    return cfg
