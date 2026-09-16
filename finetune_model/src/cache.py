"""Prepared-case cache.

prepare_for_inference (cycle basis + Hodge PE) costs ~29 ms on a 500-bus case and
far more at 10k scale, and it is re-run every launch otherwise. This builds it once
per dataset and stores {data, pg, vm} per case, plus a manifest holding:

  * the key list PER SPLIT -- train and test keys live in separate lists and are
    never concatenated, so the trainer cannot accidentally pool them;
  * the measured control counts that the top-k widths derive from;
  * bytes/case, which sizes the LRU.

Workers seek to a byte offset and read ONE line rather than receiving the record
through a pipe -- records are 0.3-3 MB and pickling them dominates the prep itself.
"""
from __future__ import annotations
import json, os, io, time
import multiprocessing as mp
import torch

from dataset import load_obj, ground_truth, case_key
from masks import scored_counts
from gridsfm.data import prepare_for_inference, drop_offline_rows

_CTX = {}


def line_offsets(path: str) -> list[int]:
    """Byte offset of every non-empty line."""
    offs = []
    with open(path, "rb") as f:
        pos = f.tell()
        for line in f:
            if line.strip():
                offs.append(pos)
            pos = f.tell()
    return offs


def _init(path, cache_dir, grid_id, split, drop_offline):
    _CTX.update(path=path, cache_dir=cache_dir, grid_id=grid_id, split=split,
                drop_offline=drop_offline)


def _prep_one(off: int):
    """-> (key, mode, n_gen_scored, n_vctrl_scored, nbytes, status)
    status: 1 prepared, 0 already cached, -1 failed."""
    path, cache_dir = _CTX["path"], _CTX["cache_dir"]
    grid_id, split = _CTX["grid_id"], _CTX["split"]   # split labels the record, not the key
    drop_off = _CTX["drop_offline"]
    try:
        with open(path) as f:
            f.seek(off)
            obj = json.loads(f.readline())
        md = obj.get("metadata", {})
        key = case_key(grid_id, md, drop_off)
        cf = os.path.join(cache_dir, key + ".pt")
        if os.path.exists(cf):
            o = torch.load(cf, weights_only=False)
            ng, nv = o.get("n_gen_scored", 0), o.get("n_vctrl_scored", 0)
            return key, md.get("perturb_mode"), ng, nv, os.path.getsize(cf), 0
        data = load_obj(obj, tag=f"{split}@{off}")
        pg, vm = ground_truth(obj)
        if drop_off:
            # Offline rows leave the INPUT. `pg` is row-aligned with the generator
            # tensor, so it is filtered by the same indices or every label after the
            # first dropped unit would be attached to the wrong generator. `vm` is
            # per-bus and buses are never dropped.
            kept = drop_offline_rows(data)
            if "generator" in kept:
                idx = kept["generator"].tolist()
                pg = [pg[i] for i in idx]
        data = prepare_for_inference(data)
        ng, nv = scored_counts(data)
        rec = {"data": data, "pg": pg, "vm": vm,
               "n_gen_scored": ng, "n_vctrl_scored": nv,
               "mode": md.get("perturb_mode"), "scenario_id": md.get("scenario_id")}
        tmp = cf + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, cf)                                    # atomic
        return key, md.get("perturb_mode"), ng, nv, os.path.getsize(cf), 1
    except Exception as e:                                     # keep the run alive
        print(f"FAIL offset {off}: {type(e).__name__}: {e}", flush=True)
        return None, None, 0, 0, 0, -1


def build(cfg: dict, splits=("train", "val", "test")) -> dict:
    """Build/refresh the cache for every split and write the manifest."""
    cache_dir = cfg["cache"]["dir"]
    os.makedirs(cache_dir, exist_ok=True)
    nw = int(cfg["cache"]["workers"])
    drop_off = bool(cfg["data"].get("drop_offline", True))
    # Recorded so a cache can be audited for which representation it holds, and so
    # load_manifest can refuse one built the other way.
    man = {"grid_id": cfg["grid_id"], "config_sha256": cfg["_meta"]["config_sha256"],
           "drop_offline": drop_off, "splits": {}, "stats": {}}
    print(f"input representation: "
          f"{'ACTIVE-ONLY (offline rows dropped)' if drop_off else 'ALL ROWS'}",
          flush=True)
    t0 = time.time()
    tot_g = tot_v = tot_b = n_ok = 0

    for split in splits:
        path = cfg["data"][split]
        offs = line_offsets(path)
        print(f"[{split}] {len(offs)} records from {path}", flush=True)
        with mp.Pool(nw, initializer=_init,
                     initargs=(path, cache_dir, cfg["grid_id"], split,
                               drop_off)) as p:
            res = p.map(_prep_one, offs, chunksize=8)
        keys, key_modes, modes, nfail = [], [], {}, 0
        for key, mode, ng, nv, nb, st in res:
            if st < 0:
                nfail += 1
                continue
            keys.append(key)
            key_modes.append(mode)          # aligned with `keys`; lets a training-set
            # subset be stratified by perturbation mode instead of sampled blindly
            modes[mode] = modes.get(mode, 0) + 1
            tot_g += ng; tot_v += nv; tot_b += nb; n_ok += 1
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"[{split}] duplicate cache keys -- refusing to "
                               "continue; the dataset has colliding scenario ids")
        man["splits"][split] = {"keys": keys, "modes": key_modes, "n": len(keys),
                                "n_failed": nfail, "by_mode": modes, "source": path}
        print(f"[{split}] cached {len(keys)}  failed {nfail}  modes {modes}", flush=True)

    # cross-split leakage check, every pair: an identical key in two splits means the
    # same scenario was published twice, which would put a held-out case into training.
    import itertools
    for a, b in itertools.combinations(splits, 2):
        overlap = set(man["splits"][a]["keys"]) & set(man["splits"][b]["keys"])
        if overlap:
            raise RuntimeError(f"{len(overlap)} cases appear in BOTH {a} and {b} -- "
                               f"{b} is not held out. Refusing to continue.")

    man["stats"] = {
        "mean_scored_gens": tot_g / max(n_ok, 1),
        "mean_scored_vctrl": tot_v / max(n_ok, 1),
        "bytes_per_case": tot_b / max(n_ok, 1),
        "total_bytes": tot_b,
        "build_seconds": time.time() - t0,
        "workers": nw,
    }
    mp_path = os.path.join(cache_dir, "manifest.json")
    tmp = mp_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, mp_path)
    print(f"manifest -> {mp_path}  ({man['stats']['total_bytes']/2**30:.2f} GB, "
          f"{man['stats']['build_seconds']:.0f}s)", flush=True)
    return man


def load_manifest(cfg: dict) -> dict:
    p = os.path.join(cfg["cache"]["dir"], "manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no cache manifest at {p} -- run scripts/prep_cache.py --config <cfg> first")
    with open(p) as f:
        man = json.load(f)
    if man.get("grid_id") != cfg["grid_id"]:
        raise RuntimeError(f"cache is for grid_id={man.get('grid_id')!r}, "
                           f"config says {cfg['grid_id']!r}")
    # A cache built with the other input representation holds different tensors under
    # different keys, so a run would find none of its keys and fail late and obscurely.
    # Fail here instead, naming the fix.
    want = bool(cfg["data"].get("drop_offline", True))
    have = man.get("drop_offline")
    if have is not None and bool(have) != want:
        raise RuntimeError(
            f"cache was built with drop_offline={have}, config says {want}. "
            f"These are different input representations. Either set "
            f"data.drop_offline: {str(have).lower()} in the config, or rebuild the "
            f"cache with scripts/prep_cache.py.")
    return man
