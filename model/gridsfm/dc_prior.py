"""DC power-flow prior for θ and branch-P (cached per topology)."""
from __future__ import annotations
import hashlib
from typing import Dict, Optional
import torch
from torch import Tensor
import numpy as np
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import factorized

from .schema import (
    BUS_TYPE_IDX,
    LOAD_PD_IDX,
    SHUNT_GS_IDX,
    AC_LINE_X_IDX,
    TR_X_IDX,
    TR_SHIFT_IDX,
    REACTANCE_EPS,
    REACTANCE_FLOOR,
    DC_PRIOR_SCALE_MIN,
    DC_PRIOR_SCALE_MAX,
)


class DCPriorCache:
    """Per-topology SuperLU-factor cache (keyed by topology + impedances + slack)."""

    def __init__(self, max_cache: int = 64):
        import threading
        from collections import OrderedDict
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._max_cache = max_cache
        self._lock = threading.Lock()

    def topo_key(self, n_bus: int, ei: Tensor, b_ij: np.ndarray, slack_idx: int) -> str:
        h = hashlib.sha1()
        h.update(f"{n_bus}|{ei.shape[1]}|{int(slack_idx)}|".encode())
        if ei.numel() > 0:
            h.update(ei.cpu().numpy().tobytes())
        if b_ij.size > 0:
            h.update(b_ij.tobytes())
        return h.hexdigest()

    def get_or_build(self, n_bus: int, ei_cpu: Tensor,
                     b_ij: np.ndarray, slack_idx: int,
                     key: Optional[str] = None) -> object:
        # Compute the key here when not supplied. Callers that already
        # need the key (e.g. for a per-topology grouping dict) can pass
        # it to skip the second sha1; otherwise the API is safe by
        # default and cannot be silently poisoned with a stale key.
        if key is None:
            key = self.topo_key(n_bus, ei_cpu, b_ij, slack_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        i_idx = ei_cpu[0].numpy()
        j_idx = ei_cpu[1].numpy()

        b_ij_f64 = b_ij.astype(np.float64)
        rows = np.concatenate([i_idx, j_idx, i_idx, j_idx])
        cols = np.concatenate([j_idx, i_idx, i_idx, j_idx])
        vals = np.concatenate([-b_ij_f64, -b_ij_f64, b_ij_f64, b_ij_f64])

        B = csr_matrix((vals, (rows, cols)), shape=(n_bus, n_bus))

        mask = np.ones(n_bus, dtype=bool)
        mask[slack_idx] = False
        B_red = B[mask][:, mask].tocsc()

        try:
            solve_fn = factorized(B_red)
        except RuntimeError as e:
            if "singular" not in str(e).lower():
                raise
            B_red_reg = (B_red + 1e-6 * sparse_eye(B_red.shape[0],
                                                   dtype=B_red.dtype,
                                                   format='csc')).tocsc()
            solve_fn = factorized(B_red_reg)
        value = (solve_fn, mask, slack_idx)
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)
        return value


def compute_dc_prior(
    data,
    genP_pred: Tensor,
    cache: DCPriorCache,
) -> Tensor:
    device = genP_pred.device
    dtype = genP_pred.dtype
    Nb = data['bus'].x.size(0)
    bus_batch = data['bus'].batch.cpu()
    G = int(bus_batch.max().item()) + 1 if bus_batch.numel() > 0 else 1

    P_inject = torch.zeros(Nb, dtype=torch.float64)
    genP_cpu = torch.nan_to_num(genP_pred.detach().cpu().double(),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    if ('generator', 'generator_link', 'bus') in data.edge_types:
        g2b = data['generator', 'generator_link', 'bus'].edge_index[1].cpu()
        Pload_total_g = torch.zeros(G, dtype=torch.float64)
        if ('load', 'load_link', 'bus') in data.edge_types:
            l2b = data['load', 'load_link', 'bus'].edge_index[1].cpu()
            load_batch = bus_batch[l2b]
            Pd = torch.nan_to_num(data['load'].x[:, LOAD_PD_IDX].cpu().double(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            Pload_total_g.scatter_add_(0, load_batch, Pd)
        gen_batch = bus_batch[g2b]
        Pgen_total_g = torch.zeros(G, dtype=torch.float64)
        Pgen_total_g.scatter_add_(0, gen_batch, genP_cpu)
        scale_g = Pload_total_g / Pgen_total_g.clamp_min(1e-6)
        scale_g = scale_g.clamp(DC_PRIOR_SCALE_MIN, DC_PRIOR_SCALE_MAX)
        genP_scaled = genP_cpu * scale_g[gen_batch]
        P_inject.scatter_add_(0, g2b, genP_scaled)
    if ('load', 'load_link', 'bus') in data.edge_types:
        l2b = data['load', 'load_link', 'bus'].edge_index[1].cpu()
        Pd_neg = torch.nan_to_num(data['load'].x[:, LOAD_PD_IDX].cpu().double(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        P_inject.scatter_add_(0, l2b, -Pd_neg)

    if ('shunt', 'shunt_link', 'bus') in data.edge_types:
        s2b = data['shunt', 'shunt_link', 'bus'].edge_index[1].cpu()
        if s2b.numel() > 0 and data['shunt'].x.size(1) > 1:
            gs = torch.nan_to_num(data['shunt'].x[:, SHUNT_GS_IDX].cpu().double(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            P_inject.scatter_add_(0, s2b, -gs)

    P_inject = torch.nan_to_num(P_inject, nan=0.0, posinf=0.0, neginf=0.0)


    bus_type = data['bus'].x[:, BUS_TYPE_IDX].cpu().long()
    P_inject_np = P_inject.numpy()

    all_i, all_j, all_b, all_phi = [], [], [], []
    for et_name in ('ac_line', 'transformer'):
        key = ('bus', et_name, 'bus')
        if key not in data.edge_types:
            continue
        ei = data[key].edge_index.cpu()
        ea = data[key].edge_attr.cpu()
        if ei.numel() == 0:
            continue
        if et_name == 'ac_line':
            x_col = AC_LINE_X_IDX
            phi_col = None
        else:
            x_col = TR_X_IDX
            phi_col = TR_SHIFT_IDX
        x_vals = ea[:, x_col].numpy()
        x_vals = np.where(np.abs(x_vals) < REACTANCE_EPS, REACTANCE_FLOOR, x_vals)
        b_vals = 1.0 / x_vals
        phi_vals = ea[:, phi_col].numpy() if phi_col is not None else np.zeros(ei.size(1))
        all_i.append(ei[0].numpy())
        all_j.append(ei[1].numpy())
        all_b.append(b_vals)
        all_phi.append(phi_vals)

    if not all_i:
        return torch.zeros(Nb, device=device, dtype=dtype)

    all_i = np.concatenate(all_i)
    all_j = np.concatenate(all_j)
    all_b = np.concatenate(all_b)
    all_phi = np.concatenate(all_phi)

    theta_dc_all = np.zeros(Nb, dtype=np.float64)
    bus_batch_np = bus_batch.numpy()

    pending = []
    groups: Dict[str, tuple] = {}

    for g in range(G):
        bus_mask = (bus_batch_np == g)
        bus_idx = np.where(bus_mask)[0]
        n_bus_g = len(bus_idx)
        if n_bus_g < 2:
            continue

        global_to_local = np.full(Nb, -1, dtype=np.int64)
        global_to_local[bus_idx] = np.arange(n_bus_g)

        edge_mask = bus_mask[all_i] & bus_mask[all_j]
        if not edge_mask.any():
            continue
        e_i = global_to_local[all_i[edge_mask]]
        e_j = global_to_local[all_j[edge_mask]]
        e_b = all_b[edge_mask]
        e_phi = all_phi[edge_mask]

        local_bus_type = bus_type[bus_idx].numpy()
        slack_candidates = np.where(local_bus_type == 3)[0]
        slack_local = int(slack_candidates[0]) if len(slack_candidates) > 0 else 0

        P_g = P_inject_np[bus_idx].copy()

        phi_contrib = e_b * e_phi
        np.add.at(P_g, e_i, phi_contrib)
        np.add.at(P_g, e_j, -phi_contrib)

        P_g[slack_local] -= P_g.sum()

        if not np.isfinite(P_g).all():
            continue

        ei_local = torch.stack([torch.from_numpy(e_i), torch.from_numpy(e_j)])
        gkey = cache.topo_key(n_bus_g, ei_local, e_b, slack_local)
        solve_fn, mask, _ = cache.get_or_build(
            n_bus_g, ei_local, e_b, slack_local, key=gkey)

        P_red = P_g[mask]
        slot = {
            'bus_idx': bus_idx, 'mask': mask,
            'n_bus_g': n_bus_g, 'P_red': P_red,
            'theta_red': None,
        }
        pending.append(slot)
        if gkey not in groups:
            groups[gkey] = (solve_fn, [])
        groups[gkey][1].append(slot)

    for solve_fn, slots in groups.values():
        if len(slots) == 1:
            slots[0]['theta_red'] = solve_fn(slots[0]['P_red'])
            continue
        block = np.stack([s['P_red'] for s in slots], axis=1)
        theta_block = solve_fn(block)
        for k, s in enumerate(slots):
            s['theta_red'] = theta_block[:, k]

    for s in pending:
        theta_red = s['theta_red']
        if not np.isfinite(theta_red).all():
            theta_red = np.where(np.isfinite(theta_red), theta_red, 0.0)
        theta_g = np.zeros(s['n_bus_g'], dtype=np.float64)
        theta_g[s['mask']] = theta_red
        theta_dc_all[s['bus_idx']] = theta_g

    return torch.from_numpy(theta_dc_all).to(device=device, dtype=dtype).detach()
