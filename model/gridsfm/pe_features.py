"""Deterministic input features: bus-type one-hot, V setpoint, graph context.

Precomputed DC / ER positional encodings were REMOVED (owner directive): the
learned HodgePE carries topology, so a hand-computed DC-power-flow prior + ER
landmark moments injected as fixed input columns are no longer wanted. What
remains here is only the deterministic, cheap-to-derive input state the model
still consumes: the bus-type one-hot, the per-bus voltage setpoint, and the
per-graph context vector `g_ctx`. `hash_topology` is retained because the
cycle-basis cache keys on it.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .schema import (
    BUS_TYPE_IDX,
    BUS_VMIN_IDX,
    BUS_VMAX_IDX,
    GEN_VG_IDX,
    GEN_PMAX_IDX,
    GEN_QMIN_IDX,
    GEN_QMAX_IDX,
    LOAD_PD_IDX,
    LOAD_QD_IDX,
    AC_LINE_R_IDX,
    AC_LINE_X_IDX,
    AC_LINE_RATE_A_IDX,
    AC_LINE_KEY,
    TR_R_IDX,
    TR_X_IDX,
    TR_RATE_A_IDX,
    TRANSFORMER_KEY,
)


def hash_topology(
    data: HeteroData,
    *,
    include_bus_type: bool = False,
    include_gen_link: bool = False,
    prefix: bytes = b"",
) -> str:
    h = hashlib.sha1()
    if prefix:
        h.update(prefix)
    h.update(f"{int(data['bus'].x.size(0))}|".encode())
    if include_bus_type:
        bx = data['bus'].x
        if bx.size(1) > BUS_TYPE_IDX:
            h.update(b"bus_type|")
            h.update(bx[:, BUS_TYPE_IDX].detach().cpu().contiguous()
                      .numpy().astype('int64').tobytes())
    if include_gen_link:
        gen_key = ('generator', 'generator_link', 'bus')
        if gen_key in data.edge_types:
            h.update(b"gen_link|")
            h.update(data[gen_key].edge_index.detach().cpu().numpy().tobytes())
        else:
            h.update(b"gen_link:none|")
    for et in (AC_LINE_KEY, TRANSFORMER_KEY):
        if et in data.edge_types:
            ei = data[et].edge_index.cpu().numpy()
            h.update(f"{et[1]}|".encode())
            h.update(ei.tobytes())
            ea = data[et].get('edge_attr', None)
            if ea is not None and ea.numel() > 0:
                h.update(ea.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(f"{et[1]}:none|".encode())
    return h.hexdigest()


_PE_ATTACHED_FLAG = "_gridsfm_pe_attached"


def attach_pe_features_(data: HeteroData) -> HeteroData:
    """Attach the deterministic input features the model consumes.

    Appends the bus-type one-hot to `bus.x` (if not already present), derives
    the per-bus voltage setpoint from incident generator Vg, and computes the
    per-graph context vector `g_ctx`. Idempotent via `_PE_ATTACHED_FLAG`.
    """
    if getattr(data, _PE_ATTACHED_FLAG, False):
        return data

    expected_bus_cols = 4 + 1 + 4   # raw + on/off avail + bus-type one-hot
    if int(data['bus'].x.size(1)) < expected_bus_cols:
        bus_x = data['bus'].x
        bus_type = bus_x[:, BUS_TYPE_IDX].long().clamp(1, 4) - 1
        bus_type_oh = torch.nn.functional.one_hot(bus_type, num_classes=4).to(
            dtype=bus_x.dtype)
        data['bus'].x = torch.cat([bus_x, bus_type_oh], dim=1)

    if not hasattr(data['bus'], 'v_setpoint'):
        bus_x = data['bus'].x
        n_bus = bus_x.size(0)
        v_setpoint = torch.zeros(n_bus, 1, dtype=bus_x.dtype, device=bus_x.device)
        if ('generator', 'generator_link', 'bus') in data.edge_types:
            ei = data[('generator', 'generator_link', 'bus')].edge_index
            gen_x = data['generator'].x
            if gen_x.size(0) > 0 and ei.numel() > 0:
                src_vg = gen_x[ei[0], GEN_VG_IDX]
                ones = torch.ones_like(src_vg)
                vg_sum = torch.zeros(n_bus, device=bus_x.device, dtype=src_vg.dtype)
                vg_cnt = torch.zeros(n_bus, device=bus_x.device, dtype=src_vg.dtype)
                vg_sum.scatter_add_(0, ei[1], src_vg)
                vg_cnt.scatter_add_(0, ei[1], ones)
                v_mean = vg_sum / vg_cnt.clamp_min(1.0)
                v_setpoint[:, 0] = v_mean.to(dtype=bus_x.dtype)
        data['bus'].v_setpoint = v_setpoint

    if not hasattr(data, 'g_ctx'):
        data.g_ctx = _compute_graph_context(data).to(
            dtype=data['bus'].x.dtype, device=data['bus'].x.device,
        )

    setattr(data, _PE_ATTACHED_FLAG, True)
    return data


def _compute_graph_context(data: HeteroData) -> Tensor:
    if hasattr(data, "num_graphs") and int(getattr(data, "num_graphs", 1) or 1) > 1:
        raise ValueError(
            "_compute_graph_context is single-graph only. Call it on each "
            "HeteroData before batching, or use batch_data_list(...) which "
            "stacks per-graph g_ctx vectors."
        )
    eps = 1e-6
    n_bus = float(data['bus'].x.size(0)) if 'bus' in data.node_types else 0.0
    n_gen = float(data['generator'].x.size(0)) if 'generator' in data.node_types else 0.0

    if 'load' in data.node_types and data['load'].x.size(0) > 0:
        Pd_tot = float(data['load'].x[:, LOAD_PD_IDX].sum())
        Qd_tot = float(data['load'].x[:, LOAD_QD_IDX].sum())
    else:
        Pd_tot = Qd_tot = 0.0

    if n_gen > 0:
        gx = data['generator'].x
        Pmax_tot = float(gx[:, GEN_PMAX_IDX].sum())
        Qmax_tot = float(gx[:, GEN_QMAX_IDX].sum())
        Qmin_tot = float(gx[:, GEN_QMIN_IDX].sum())
        Qcap_tot = max(Qmax_tot - Qmin_tot, eps)
    else:
        Pmax_tot = eps
        Qcap_tot = eps

    V_spread = float((data['bus'].x[:, BUS_VMAX_IDX] - data['bus'].x[:, BUS_VMIN_IDX]).mean()) \
        if n_bus > 0 else 0.0

    rates = []
    abs_ys = []
    if ('bus', 'ac_line', 'bus') in data.edge_types:
        ea = data[('bus', 'ac_line', 'bus')].edge_attr
        if ea.size(0) > 0:
            rates.append(ea[:, AC_LINE_RATE_A_IDX])
            r = ea[:, AC_LINE_R_IDX]; x = ea[:, AC_LINE_X_IDX]
            z2 = (r.pow(2) + x.pow(2)).clamp_min(eps)
            abs_ys.append(z2.rsqrt())
    if ('bus', 'transformer', 'bus') in data.edge_types:
        ea = data[('bus', 'transformer', 'bus')].edge_attr
        if ea.size(0) > 0:
            rates.append(ea[:, TR_RATE_A_IDX])
            r = ea[:, TR_R_IDX]; x = ea[:, TR_X_IDX]
            z2 = (r.pow(2) + x.pow(2)).clamp_min(eps)
            abs_ys.append(z2.rsqrt())

    if rates:
        all_rates = torch.cat([t.float() for t in rates])
        all_ys    = torch.cat([t.float() for t in abs_ys])
        mean_rate  = float(all_rates.mean())
        mean_abs_y = float(all_ys.mean())
        max_abs_y  = float(all_ys.max())
    else:
        mean_rate = mean_abs_y = max_abs_y = 0.0

    g_ctx = torch.tensor([
        math.log(max(n_bus, 1.0)),
        math.log(n_gen + 1.0),
        Pd_tot / max(Pmax_tot, eps),
        Qd_tot / max(Qcap_tot, eps),
        V_spread,
        math.log1p(mean_rate),
        math.log1p(mean_abs_y),
        math.log1p(max_abs_y),
    ], dtype=torch.float32).unsqueeze(0)
    return g_ctx
