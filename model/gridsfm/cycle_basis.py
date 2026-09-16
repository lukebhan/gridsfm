"""Cycle-basis precomputation: attach f2 (cycle nodes) to a HeteroData grid."""
from __future__ import annotations

import hashlib
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData

from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as _scipy_mst


def _default_cycle_disk_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "gridsfm" / "cycle_basis"


_DEFAULT_CYCLE_DISK_DIR = _default_cycle_disk_dir()

_CYCLE_CACHE_SCHEMA = 1
_CYCLE_ATTACHED_FLAG = "_gridsfm_cycle_attached"


from .schema import (
    AC_LINE_R_IDX as _AC_LINE_R_IDX,
    AC_LINE_X_IDX as _AC_LINE_X_IDX,
    AC_LINE_RATE_A_IDX as _AC_LINE_RATE_A_IDX,
    TR_R_IDX as _TRANSFORMER_R_IDX,
    TR_X_IDX as _TRANSFORMER_X_IDX,
    TR_RATE_A_IDX as _TRANSFORMER_RATE_A_IDX,
    AC_LINE_KEY as _AC_LINE_KEY,
    TRANSFORMER_KEY as _TRANSFORMER_KEY,
)


def _fundamental_cycles_fast(data: HeteroData) -> List:
    n_bus = int(data['bus'].x.size(0))

    pair_edges: Dict[Tuple[int, int], List[Tuple]] = {}
    if _AC_LINE_KEY in data.edge_types:
        ei = data[_AC_LINE_KEY].edge_index.cpu().numpy()
        for r in range(ei.shape[1]):
            u, v = int(ei[0, r]), int(ei[1, r])
            a, b = (u, v) if u <= v else (v, u)
            pair_edges.setdefault((a, b), []).append((u, v, ("ac_line", r)))
    if _TRANSFORMER_KEY in data.edge_types:
        ei = data[_TRANSFORMER_KEY].edge_index.cpu().numpy()
        for r in range(ei.shape[1]):
            u, v = int(ei[0, r]), int(ei[1, r])
            a, b = (u, v) if u <= v else (v, u)
            pair_edges.setdefault((a, b), []).append((u, v, ("transformer", r)))

    if not pair_edges:
        return []

    pairs = list(pair_edges.keys())
    rows = np.array([p[0] for p in pairs], dtype=np.int32)
    cols = np.array([p[1] for p in pairs], dtype=np.int32)
    n_pairs = len(pairs)
    weights = np.arange(1, n_pairs + 1, dtype=np.float64)
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_w = np.concatenate([weights, weights])
    A = _csr_matrix((sym_w, (sym_rows, sym_cols)), shape=(n_bus, n_bus))

    mst_sparse = _scipy_mst(A).tocoo()
    tree_pairs = set()
    for i, j in zip(mst_sparse.row.astype(np.int64), mst_sparse.col.astype(np.int64)):
        a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
        tree_pairs.add((a, b))

    tree_adj: List[List[int]] = [[] for _ in range(n_bus)]
    for (a, b) in tree_pairs:
        tree_adj[a].append(b)
        tree_adj[b].append(a)

    parent = np.full(n_bus, -1, dtype=np.int64)
    depth = np.full(n_bus, -1, dtype=np.int64)
    component = np.full(n_bus, -1, dtype=np.int64)
    next_comp = 0
    for start in range(n_bus):
        if depth[start] >= 0:
            continue
        depth[start] = 0
        component[start] = next_comp
        stack = [start]
        while stack:
            u = stack.pop()
            for v in tree_adj[u]:
                if depth[v] >= 0:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                component[v] = next_comp
                stack.append(v)
        next_comp += 1

    cycles: List = []
    for (a, b), edges_here in pair_edges.items():
        is_tree_pair = (a, b) in tree_pairs
        for idx, (u, v, key) in enumerate(edges_here):
            if is_tree_pair and idx == 0:
                continue
            if component[u] != component[v]:
                continue
            pu, pv = u, v
            path_u: List[int] = [pu]
            path_v: List[int] = [pv]
            while depth[pu] > depth[pv]:
                pu = int(parent[pu]); path_u.append(pu)
            while depth[pv] > depth[pu]:
                pv = int(parent[pv]); path_v.append(pv)
            while pu != pv:
                pu = int(parent[pu]); path_u.append(pu)
                pv = int(parent[pv]); path_v.append(pv)
            full_path = path_u + path_v[-2::-1]
            cycle_edges: List = []
            for x, y in zip(full_path[:-1], full_path[1:]):
                ax, by = (x, y) if x <= y else (y, x)
                tree_key = pair_edges[(ax, by)][0][2]
                cycle_edges.append((x, y, tree_key))
            cycle_edges.append((v, u, key))
            cycles.append(cycle_edges)
    return cycles


def compute_cycle_basis(data: HeteroData) -> Dict:
    cycles = _fundamental_cycles_fast(data)
    n_cycles = len(cycles)

    if n_cycles == 0:
        return {
            'n_cycles':   0,
            'ac_ce_index': torch.zeros(2, 0, dtype=torch.long),
            'ac_ce_signs': torch.zeros(0, dtype=torch.float32),
            'tr_ce_index': torch.zeros(2, 0, dtype=torch.long),
            'tr_ce_signs': torch.zeros(0, dtype=torch.float32),
            'cycle_x':    torch.zeros(0, 4, dtype=torch.float32),
        }

    ac_attr = data[_AC_LINE_KEY].edge_attr.cpu().numpy() if _AC_LINE_KEY in data.edge_types else None
    tr_attr = data[_TRANSFORMER_KEY].edge_attr.cpu().numpy() if _TRANSFORMER_KEY in data.edge_types else None

    ac_ci_list, ac_ei_list, ac_sign_list = [], [], []
    tr_ci_list, tr_ei_list, tr_sign_list = [], [], []
    cycle_features = torch.zeros(n_cycles, 4, dtype=torch.float32)

    for ci, cycle in enumerate(cycles):
        sum_x = 0.0
        sum_r = 0.0
        min_rate = float('inf')
        n_edges = 0
        for (a, b, key) in cycle:
            etype, row = key
            sign = +1 if a < b else -1
            if etype == "ac_line":
                ac_ci_list.append(ci)
                ac_ei_list.append(row)
                ac_sign_list.append(sign)
                if ac_attr is not None:
                    sum_x   += abs(float(ac_attr[row, _AC_LINE_X_IDX]))
                    sum_r   += abs(float(ac_attr[row, _AC_LINE_R_IDX]))
                    rate_a   = float(ac_attr[row, _AC_LINE_RATE_A_IDX])
                    if rate_a > 0:
                        min_rate = min(min_rate, rate_a)
            else:
                tr_ci_list.append(ci)
                tr_ei_list.append(row)
                tr_sign_list.append(sign)
                if tr_attr is not None:
                    sum_x += abs(float(tr_attr[row, _TRANSFORMER_X_IDX]))
                    sum_r += abs(float(tr_attr[row, _TRANSFORMER_R_IDX]))
                    rate_a = float(tr_attr[row, _TRANSFORMER_RATE_A_IDX])
                    if rate_a > 0:
                        min_rate = min(min_rate, rate_a)
            n_edges += 1
        if min_rate == float('inf'):
            min_rate = 0.0
        cycle_features[ci, 0] = float(n_edges)
        cycle_features[ci, 1] = float(sum_x)
        cycle_features[ci, 2] = float(sum_r)
        cycle_features[ci, 3] = float(min_rate)

    return {
        'n_cycles':    n_cycles,
        'ac_ce_index': torch.tensor([ac_ci_list, ac_ei_list], dtype=torch.long) if ac_ci_list
                       else torch.zeros(2, 0, dtype=torch.long),
        'ac_ce_signs': torch.tensor(ac_sign_list, dtype=torch.float32) if ac_sign_list
                       else torch.zeros(0, dtype=torch.float32),
        'tr_ce_index': torch.tensor([tr_ci_list, tr_ei_list], dtype=torch.long) if tr_ci_list
                       else torch.zeros(2, 0, dtype=torch.long),
        'tr_ce_signs': torch.tensor(tr_sign_list, dtype=torch.float32) if tr_sign_list
                       else torch.zeros(0, dtype=torch.float32),
        'cycle_x':    cycle_features,
    }


class CycleBasisCache:
    """Per-topology cycle-basis cache (keyed by topology hash, optional disk persistence)."""
    def __init__(
        self,
        max_cache: int = 128,
        cache_dir: Union[str, Path, None] = _DEFAULT_CYCLE_DISK_DIR,
    ):
        import threading
        self._store: "OrderedDict[str, Dict]" = OrderedDict()
        self.max_cache = max(1, int(max_cache))
        self._lock = threading.Lock()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                warnings.warn(
                    f"CycleBasisCache: cannot create {self.cache_dir} ({e}); "
                    f"disk persistence disabled, falling back to in-memory cache only.",
                    RuntimeWarning,
                )
                self.cache_dir = None

    @staticmethod
    def _fingerprint(data: HeteroData) -> str:
        from .pe_features import hash_topology
        return hash_topology(data, prefix=f"v{_CYCLE_CACHE_SCHEMA}|".encode())

    def _disk_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"v{_CYCLE_CACHE_SCHEMA}_{hashlib.sha1(key.encode()).hexdigest()}.pt"

    def get_or_compute(self, data: HeteroData) -> Dict:
        key = self._fingerprint(data)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            disk = self._disk_path(key)
            if disk is not None and disk.exists():
                try:
                    blob = torch.load(disk, weights_only=True, map_location="cpu")
                    if isinstance(blob, dict) and blob.get('_schema') == _CYCLE_CACHE_SCHEMA:
                        value = blob['value']
                        self._store[key] = value
                        while len(self._store) > self.max_cache:
                            self._store.popitem(last=False)
                        return value
                except Exception as e:
                    warnings.warn(
                        f"CycleBasisCache: ignoring corrupt or incompatible "
                        f"disk-cache entry {disk} ({e}); recomputing.",
                        RuntimeWarning,
                    )
        value = compute_cycle_basis(data)
        with self._lock:
            if disk is not None:
                try:
                    torch.save({'_schema': _CYCLE_CACHE_SCHEMA, 'value': value}, disk)
                except OSError as e:
                    warnings.warn(
                        f"CycleBasisCache: failed to persist to {disk} ({e}); "
                        f"in-memory cache only for this entry.",
                        RuntimeWarning,
                    )
            self._store[key] = value
            while len(self._store) > self.max_cache:
                self._store.popitem(last=False)
        return value


def attach_branches_as_nodes_(data: HeteroData) -> HeteroData:
    ac_done = 'branch_ac' in data.node_types and data['branch_ac'].x.size(0) > 0
    tr_done = 'branch_tr' in data.node_types and data['branch_tr'].x.size(0) > 0
    ac_src = _AC_LINE_KEY in data.edge_types
    tr_src = _TRANSFORMER_KEY in data.edge_types
    if (ac_done or not ac_src) and (tr_done or not tr_src):
        return data

    if _AC_LINE_KEY in data.edge_types and not ac_done:
        _promote_to_branch_nodes(data, _AC_LINE_KEY, 'branch_ac')
    if _TRANSFORMER_KEY in data.edge_types and not tr_done:
        _promote_to_branch_nodes(data, _TRANSFORMER_KEY, 'branch_tr')
    return data


def _promote_to_branch_nodes(
    data: HeteroData, edge_key: Tuple[str, str, str], branch_nt: str
) -> None:
    ei = data[edge_key].edge_index
    attr = getattr(data[edge_key], 'edge_attr', None)
    if attr is None:
        raise ValueError(
            f"{edge_key} has edge_index but no edge_attr; "
            f"branch-node promotion requires edge features."
        )
    n_branch = ei.size(1)
    dev = ei.device

    data[branch_nt].x = attr.clone()
    branch_ids = torch.arange(n_branch, device=dev)
    bus_endpoint = torch.cat([ei[0], ei[1]], dim=0)
    branch_endpoint = torch.cat([branch_ids, branch_ids], dim=0)
    side_sign = torch.cat([
        torch.ones(n_branch, dtype=torch.float32, device=dev),
        -torch.ones(n_branch, dtype=torch.float32, device=dev),
    ], dim=0).unsqueeze(-1)

    data[('bus', 'endpoint_of', branch_nt)].edge_index = torch.stack(
        [bus_endpoint, branch_endpoint], dim=0).contiguous()
    data[('bus', 'endpoint_of', branch_nt)].edge_attr = side_sign.clone()
    data[(branch_nt, 'endpoint_of', 'bus')].edge_index = torch.stack(
        [branch_endpoint, bus_endpoint], dim=0).contiguous()
    data[(branch_nt, 'endpoint_of', 'bus')].edge_attr = side_sign.clone()


DEFAULT_CYCLE_CACHE = CycleBasisCache()


def prepare_for_grid_transformer_(
    data: HeteroData,
    cache: Optional[CycleBasisCache] = None,
) -> HeteroData:
    attach_branches_as_nodes_(data)
    attach_cycle_basis_(data, cache=cache)
    return data


def attach_cycle_basis_(
    data: HeteroData,
    cache: Optional[CycleBasisCache] = None,
) -> HeteroData:
    if getattr(data, _CYCLE_ATTACHED_FLAG, False):
        return data
    if 'cycle' in data.node_types and data['cycle'].x.size(0) > 0:
        setattr(data, _CYCLE_ATTACHED_FLAG, True)
        return data

    if cache is None:
        cache = DEFAULT_CYCLE_CACHE
    cb = cache.get_or_compute(data)

    n_cycles = cb['n_cycles']
    if n_cycles == 0:
        data['cycle'].x = torch.zeros(0, 4, dtype=torch.float32)
        setattr(data, _CYCLE_ATTACHED_FLAG, True)
        return data

    data['cycle'].x = cb['cycle_x']

    if cb['ac_ce_index'].size(1) > 0:
        data[('cycle', 'in_cycle', 'branch_ac')].edge_index = cb['ac_ce_index'].clone()
        data[('cycle', 'in_cycle', 'branch_ac')].edge_attr  = cb['ac_ce_signs'].unsqueeze(-1).clone()
        data[('branch_ac', 'in_cycle', 'cycle')].edge_index = cb['ac_ce_index'].flip(0).clone()
        data[('branch_ac', 'in_cycle', 'cycle')].edge_attr  = cb['ac_ce_signs'].unsqueeze(-1).clone()
    if cb['tr_ce_index'].size(1) > 0:
        data[('cycle', 'in_cycle', 'branch_tr')].edge_index = cb['tr_ce_index'].clone()
        data[('cycle', 'in_cycle', 'branch_tr')].edge_attr  = cb['tr_ce_signs'].unsqueeze(-1).clone()
        data[('branch_tr', 'in_cycle', 'cycle')].edge_index = cb['tr_ce_index'].flip(0).clone()
        data[('branch_tr', 'in_cycle', 'cycle')].edge_attr  = cb['tr_ce_signs'].unsqueeze(-1).clone()

    setattr(data, _CYCLE_ATTACHED_FLAG, True)
    return data
