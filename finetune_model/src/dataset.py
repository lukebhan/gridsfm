"""JSONL dataset access + the in-memory .pyg.json adapter.

generate_finetune_dataset emits ONE record per line of train_data.jsonl /
test_data.jsonl, each `{grid, solution, metadata}` -- the same envelope
gridsfm.load_pyg_json() reads, but never as a standalone file. load_pyg_json()
opens a path and json.load()s it, so it cannot be pointed at a line. `load_obj`
below is that function's body from the parse onward, and is the only reason this
module needs gridsfm internals.

The two splits are separate files and stay separate all the way through: there is
no function here that concatenates them.
"""
from __future__ import annotations
import json, hashlib, os
from typing import Iterator

from torch_geometric.data import HeteroData
from gridsfm.data import _to_tensor, _wire_edges_into, prepare_for_inference


def load_obj(obj: dict, tag: str = "<jsonl>") -> HeteroData:
    """gridsfm.load_pyg_json(), minus the file read. Same validation, same result."""
    if "grid" not in obj or "nodes" not in obj.get("grid", {}):
        raise ValueError(f"{tag}: expected top-level 'grid' with 'nodes'")
    g = obj["grid"]
    nodes = g["nodes"]
    if not nodes.get("bus"):
        raise ValueError(f"{tag}: missing or empty 'bus' nodes")
    d = HeteroData()
    for nt in ("bus", "generator", "load", "shunt"):
        if nt in nodes and nodes[nt]:
            d[nt].x = _to_tensor(nodes[nt])
    _wire_edges_into(d, g.get("edges", {}), tag)
    return d


def prepare(obj: dict, tag: str = "<jsonl>") -> HeteroData:
    """Full prep: adapter + the expensive cycle-basis / Hodge-PE pass."""
    return prepare_for_inference(load_obj(obj, tag))


def ground_truth(obj: dict) -> tuple[list, list]:
    """(pg per generator, vm per bus) from the solved AC-OPF.

    NOTE the node-type key is `generator`, not `gen` -- the exporter's name.
    """
    sol = obj["solution"]["nodes"]
    return [r[0] for r in sol["generator"]], [r[1] for r in sol["bus"]]


def case_key(grid_id: str, md: dict, drop_offline: bool = False) -> str:
    """Stable cache key identifying a SCENARIO, deliberately independent of which
    split it landed in.

    (mode, scenario_id, scenario_seed) already identifies a case uniquely, and keying
    on the split as well would mean a re-split invalidated every case that moved --
    2600 re-preps on case500 (3 s, harmless) but a full rebuild on a 10k-bus dataset,
    where prep is the expensive part. The manifest still records which keys belong to
    which split, which is where that information belongs.

    scenario_seed is included so an --append-amended dataset cannot collide with a
    stale entry that happened to reuse a scenario_id.
    """
    raw = (f"{grid_id}|{md.get('perturb_mode')}|{md.get('scenario_id')}"
           f"|{md.get('scenario_seed')}")
    # The INPUT REPRESENTATION is part of the identity of a prepared case: an
    # active-only graph and an all-rows graph are different tensors for the same
    # scenario. Without this a cache built one way would be silently reused by a run
    # configured the other way, and the mismatch would surface as an accuracy change
    # with no apparent cause. Only the active-only key is suffixed, so caches built
    # before this existed stay valid.
    if drop_offline:
        raw += "|active_only"
    return hashlib.md5(raw.encode()).hexdigest()


def iter_records(path: str) -> Iterator[tuple[int, dict]]:
    """Stream a JSONL split. Each record is ~283 KB on case500 and ~10x that on
    10k, so this never holds more than one in memory."""
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def count_records(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())

