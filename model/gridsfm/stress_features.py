"""Per-graph physics-violation stress features for the feas head."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .schema import (
    BUS_VMIN_IDX as _BUS_VMIN,
    BUS_VMAX_IDX as _BUS_VMAX,
    GEN_PMIN_IDX as _GEN_PMIN,
    GEN_PMAX_IDX as _GEN_PMAX,
    GEN_QMIN_IDX as _GEN_QMIN,
    GEN_QMAX_IDX as _GEN_QMAX,
    LOAD_PD_IDX as _LOAD_PD,
    LOAD_QD_IDX as _LOAD_QD,
    SHUNT_BS_IDX as _SHUNT_BS,
    SHUNT_GS_IDX as _SHUNT_GS,
    AC_LINE_ANGMAX_IDX as _AC_ANGMAX,
    AC_LINE_RATE_A_IDX as _AC_RATE_A,
    TR_ANGMAX_IDX as _TR_ANGMAX,
    TR_RATE_A_IDX as _TR_RATE_A,
    TR_SHIFT_IDX as _TR_SHIFT,
)

_EPS_S2 = 1e-12

# V(3) + KCL_P(3) + KCL_Q(3) + thermal(3) + angle(3) + KVL(3) = 18
# violation dims (targeted at 0 by a stress regularizer), plus cap(4) +
# loading_margin(2) = 6 utilization dims (non-violation). A consumer slicing
# the violation prefix imports `STRESS_VIOL_DIMS` rather than hardcoding 18,
# so reordering the `torch.cat([...])` in `compute_physics_stress` forces
# this constant to move too.
STRESS_DIM = 24
STRESS_VIOL_DIMS = 18


def _pool_stats_per_graph(violations: Tensor, batch_idx: Tensor,
                          n_graphs: int) -> Tensor:
    if violations.numel() == 0:
        return torch.zeros(n_graphs, 3, dtype=torch.float32, device=violations.device)
    dev = violations.device
    dtype = violations.dtype
    sum_v = torch.zeros(n_graphs, dtype=dtype, device=dev)
    cnt = torch.zeros(n_graphs, dtype=dtype, device=dev)
    sum_v.scatter_add_(0, batch_idx, violations)
    cnt.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=dtype))
    mean_v = sum_v / cnt.clamp_min(1.0)
    NEG = torch.finfo(dtype).min
    max_v = torch.full((n_graphs,), NEG, dtype=dtype, device=dev)
    max_v = max_v.scatter_reduce(0, batch_idx, violations, reduce='amax', include_self=True)
    max_v = torch.where(cnt == 0, torch.zeros_like(max_v), max_v)
    above = (violations > 1e-3).to(dtype)
    sum_above = torch.zeros(n_graphs, dtype=dtype, device=dev)
    sum_above.scatter_add_(0, batch_idx, above)
    frac_v = sum_above / cnt.clamp_min(1.0)
    return torch.stack([mean_v, max_v, frac_v], dim=-1)


def compute_physics_stress(
    data: HeteroData,
    V: Tensor,
    theta: Tensor,
    Pg: Tensor,
    Qg: Tensor,
    *,
    ac_flows: Optional[Tensor] = None,
    tr_flows: Optional[Tensor] = None,
    n_graphs: int,
) -> Tensor:
    bus_x = data['bus'].x
    n_bus = bus_x.size(0)
    dtype = V.dtype
    dev = V.device
    bus_batch = getattr(data['bus'], 'batch', None)
    if bus_batch is None:
        bus_batch = torch.zeros(n_bus, dtype=torch.long, device=dev)

    Vmin = torch.nan_to_num(bus_x[:, _BUS_VMIN], nan=0.95).to(dtype)
    Vmax = torch.nan_to_num(bus_x[:, _BUS_VMAX], nan=1.05).to(dtype)
    Vlo = torch.minimum(Vmin, Vmax)
    Vhi = torch.maximum(Vmin, Vmax)
    V_viol = (Vlo - V).clamp_min(0.0) + (V - Vhi).clamp_min(0.0)
    V_stats = _pool_stats_per_graph(V_viol, bus_batch, n_graphs)

    P_inj = torch.zeros(n_bus, dtype=dtype, device=dev)
    Q_inj = torch.zeros(n_bus, dtype=dtype, device=dev)
    if ('generator', 'generator_link', 'bus') in data.edge_types:
        g2b = data[('generator', 'generator_link', 'bus')].edge_index[1]
        if Pg.dim() == 1 and g2b.numel() > 0:
            P_inj.scatter_add_(0, g2b, Pg)
        if Qg.dim() == 1 and g2b.numel() > 0:
            Q_inj.scatter_add_(0, g2b, Qg)
    if ('load', 'load_link', 'bus') in data.edge_types:
        l2b = data[('load', 'load_link', 'bus')].edge_index[1]
        if l2b.numel() > 0:
            Pd = data['load'].x[:, _LOAD_PD].to(dtype)
            Qd = data['load'].x[:, _LOAD_QD].to(dtype)
            P_inj.scatter_add_(0, l2b, -Pd)
            Q_inj.scatter_add_(0, l2b, -Qd)
    if ('shunt', 'shunt_link', 'bus') in data.edge_types:
        s2b = data[('shunt', 'shunt_link', 'bus')].edge_index[1]
        if s2b.numel() > 0:
            sh_x = data['shunt'].x
            bs = sh_x[:, _SHUNT_BS].to(dtype)
            Q_inj.scatter_add_(0, s2b, bs * V[s2b].pow(2))
            if sh_x.size(1) > _SHUNT_GS:
                gs = sh_x[:, _SHUNT_GS].to(dtype)
                P_inj.scatter_add_(0, s2b, -gs * V[s2b].pow(2))

    P_out = torch.zeros(n_bus, dtype=dtype, device=dev)
    Q_out = torch.zeros(n_bus, dtype=dtype, device=dev)
    if ac_flows is not None and ac_flows.numel() > 0 and \
       ('bus', 'ac_line', 'bus') in data.edge_types:
        ac_ei = data[('bus', 'ac_line', 'bus')].edge_index
        Pij = ac_flows[:, 0]
        Qij = ac_flows[:, 1]
        Pji = ac_flows[:, 2]
        Qji = ac_flows[:, 3]
        P_out.scatter_add_(0, ac_ei[0], Pij)
        P_out.scatter_add_(0, ac_ei[1], Pji)
        Q_out.scatter_add_(0, ac_ei[0], Qij)
        Q_out.scatter_add_(0, ac_ei[1], Qji)
    if tr_flows is not None and tr_flows.numel() > 0 and \
       ('bus', 'transformer', 'bus') in data.edge_types:
        tr_ei = data[('bus', 'transformer', 'bus')].edge_index
        Pij = tr_flows[:, 0]
        Qij = tr_flows[:, 1]
        Pji = tr_flows[:, 2]
        Qji = tr_flows[:, 3]
        P_out.scatter_add_(0, tr_ei[0], Pij)
        P_out.scatter_add_(0, tr_ei[1], Pji)
        Q_out.scatter_add_(0, tr_ei[0], Qij)
        Q_out.scatter_add_(0, tr_ei[1], Qji)

    KCL_P = (P_inj - P_out).abs()
    KCL_Q = (Q_inj - Q_out).abs()
    KCL_P_stats = _pool_stats_per_graph(KCL_P, bus_batch, n_graphs)
    KCL_Q_stats = _pool_stats_per_graph(KCL_Q, bus_batch, n_graphs)

    thermal_viols = []
    thermal_batches = []
    loading_margins = []
    loading_batches = []
    for et_key, et_rate_idx in (
        (('bus', 'ac_line', 'bus'), _AC_RATE_A),
        (('bus', 'transformer', 'bus'), _TR_RATE_A),
    ):
        flows = ac_flows if et_key[1] == 'ac_line' else tr_flows
        if flows is None or flows.numel() == 0 or et_key not in data.edge_types:
            continue
        ea = data[et_key].edge_attr
        rate_a = ea[:, et_rate_idx].to(dtype).clamp_min(1e-9)
        S_ij = torch.sqrt(flows[:, 0].square() + flows[:, 1].square() + _EPS_S2)
        S_ji = torch.sqrt(flows[:, 2].square() + flows[:, 3].square() + _EPS_S2)
        S_max_branch = torch.maximum(S_ij, S_ji)
        loading_ratio = S_max_branch / rate_a
        thermal_viols.append((loading_ratio - 1.0).clamp_min(0.0))
        loading_margins.append(loading_ratio - 1.0)
        per_branch_batch = bus_batch[data[et_key].edge_index[0]]
        thermal_batches.append(per_branch_batch)
        loading_batches.append(per_branch_batch)
    if thermal_viols:
        thermal_viol = torch.cat(thermal_viols, dim=0)
        thermal_batch = torch.cat(thermal_batches, dim=0)
    else:
        thermal_viol = torch.zeros(0, dtype=dtype, device=dev)
        thermal_batch = torch.zeros(0, dtype=torch.long, device=dev)
    thermal_stats = _pool_stats_per_graph(thermal_viol, thermal_batch, n_graphs)

    loading_margin_max  = torch.full((n_graphs,), -1.0, dtype=dtype, device=dev)
    loading_margin_mean = torch.full((n_graphs,), -1.0, dtype=dtype, device=dev)
    if loading_margins:
        all_margins = torch.cat(loading_margins, dim=0)
        all_batch   = torch.cat(loading_batches, dim=0)
        NEG = torch.finfo(dtype).min
        max_buf = torch.full((n_graphs,), NEG, dtype=dtype, device=dev)
        max_buf = max_buf.scatter_reduce(
            0, all_batch, all_margins, reduce='amax', include_self=True,
        )
        loading_margin_max = torch.where(
            max_buf < NEG / 2.0,
            torch.full_like(max_buf, -1.0),
            max_buf,
        )
        sum_buf = torch.zeros(n_graphs, dtype=dtype, device=dev)
        cnt_buf = torch.zeros(n_graphs, dtype=dtype, device=dev)
        sum_buf.scatter_add_(0, all_batch, all_margins)
        cnt_buf.scatter_add_(0, all_batch, torch.ones_like(all_margins))
        loading_margin_mean = torch.where(
            cnt_buf > 0,
            sum_buf / cnt_buf.clamp_min(1.0),
            torch.full_like(loading_margin_mean, -1.0),
        )
    loading_margin_stats = torch.stack(
        [loading_margin_max, loading_margin_mean], dim=-1,
    )

    angle_viols = []
    angle_batches = []
    if ('bus', 'ac_line', 'bus') in data.edge_types:
        ei = data[('bus', 'ac_line', 'bus')].edge_index
        if ei.size(1) > 0:
            ea = data[('bus', 'ac_line', 'bus')].edge_attr
            angmax = ea[:, _AC_ANGMAX].to(dtype).clamp_min(1e-9)
            dtheta = (theta[ei[0]] - theta[ei[1]]).abs()
            v_ac = (dtheta / angmax - 1.0).clamp_min(0.0)
            angle_viols.append(v_ac)
            angle_batches.append(bus_batch[ei[0]])
    if ('bus', 'transformer', 'bus') in data.edge_types:
        ei = data[('bus', 'transformer', 'bus')].edge_index
        if ei.size(1) > 0:
            ea = data[('bus', 'transformer', 'bus')].edge_attr
            angmax = ea[:, _TR_ANGMAX].to(dtype).clamp_min(1e-9)
            shift = ea[:, _TR_SHIFT].to(dtype) if ea.size(1) > _TR_SHIFT else torch.zeros(ei.size(1), dtype=dtype, device=dev)
            dtheta = (theta[ei[0]] - theta[ei[1]] - shift).abs()
            v_tr = (dtheta / angmax - 1.0).clamp_min(0.0)
            angle_viols.append(v_tr)
            angle_batches.append(bus_batch[ei[0]])
    if angle_viols:
        angle_viol = torch.cat(angle_viols, dim=0)
        angle_batch = torch.cat(angle_batches, dim=0)
    else:
        angle_viol = torch.zeros(0, dtype=dtype, device=dev)
        angle_batch = torch.zeros(0, dtype=torch.long, device=dev)
    angle_stats = _pool_stats_per_graph(angle_viol, angle_batch, n_graphs)

    if 'cycle' in data.node_types and data['cycle'].x.size(0) > 0:
        # Signed cycle sum of per-branch dtheta in original-edge direction
        # (a cycle-imbalance feature, not the textbook KVL residual).
        n_cycles_total = data['cycle'].x.size(0)
        cycle_batch = getattr(data['cycle'], 'batch', None)
        if cycle_batch is None:
            cycle_batch = torch.zeros(n_cycles_total, dtype=torch.long, device=dev)
        kvl_sum = torch.zeros(n_cycles_total, dtype=dtype, device=dev)
        for et_branch, et_phi_idx in (('branch_ac', None), ('branch_tr', _TR_SHIFT)):
            in_cycle_key = ('cycle', 'in_cycle', et_branch)
            ep_key = (et_branch, 'endpoint_of', 'bus')
            if in_cycle_key not in data.edge_types:
                continue
            if ep_key not in data.edge_types:
                continue
            ic_ei = data[in_cycle_key].edge_index
            ic_attr = data[in_cycle_key].edge_attr
            if ic_ei.size(1) == 0:
                continue
            cycle_idx = ic_ei[0]
            branch_idx = ic_ei[1]
            sign = ic_attr[:, 0].to(dtype) if ic_attr is not None and ic_attr.size(1) >= 1 else torch.ones(ic_ei.size(1), dtype=dtype, device=dev)
            ep_ei = data[ep_key].edge_index
            ep_attr = data[ep_key].edge_attr
            if ep_ei.size(1) == 0:
                continue
            br = ep_ei[0]
            bus_at_branch = ep_ei[1]
            ep_sign = ep_attr[:, 0].to(dtype) if ep_attr is not None and ep_attr.size(1) >= 1 else torch.ones(ep_ei.size(1), dtype=dtype, device=dev)
            n_branch = data[et_branch].x.size(0)
            dtheta_branch = torch.zeros(n_branch, dtype=dtype, device=dev)
            dtheta_branch.scatter_add_(0, br, ep_sign * theta[bus_at_branch])
            if et_phi_idx is not None and ('bus', 'transformer', 'bus') in data.edge_types:
                tr_ea = data[('bus', 'transformer', 'bus')].edge_attr
                if tr_ea.size(0) == n_branch and tr_ea.size(1) > et_phi_idx:
                    shift_per_branch = tr_ea[:, et_phi_idx].to(dtype)
                    dtheta_branch = dtheta_branch - shift_per_branch
            kvl_sum.scatter_add_(0, cycle_idx, sign * dtheta_branch[branch_idx])
        KVL_resid = kvl_sum.abs()
        KVL_stats = _pool_stats_per_graph(KVL_resid, cycle_batch, n_graphs)
    else:
        KVL_stats = torch.zeros(n_graphs, 3, dtype=dtype, device=dev)

    cap_eps = 1e-3
    OUTPUT_CAP = 100.0
    sum_Pd_per_g = torch.zeros(n_graphs, dtype=dtype, device=dev)
    sum_Qd_abs_per_g = torch.zeros(n_graphs, dtype=dtype, device=dev)
    if 'load' in data.node_types and data['load'].x.size(0) > 0:
        load_batch = getattr(data['load'], 'batch', None)
        if load_batch is None:
            load_batch = torch.zeros(data['load'].x.size(0), dtype=torch.long, device=dev)
        Pd_full = torch.nan_to_num(data['load'].x[:, _LOAD_PD], nan=0.0).to(dtype).abs()
        Qd_full = torch.nan_to_num(data['load'].x[:, _LOAD_QD], nan=0.0).to(dtype).abs()
        sum_Pd_per_g.scatter_add_(0, load_batch, Pd_full)
        sum_Qd_abs_per_g.scatter_add_(0, load_batch, Qd_full)
    sum_Pmax_per_g = torch.zeros(n_graphs, dtype=dtype, device=dev)
    sum_Qrange_per_g = torch.zeros(n_graphs, dtype=dtype, device=dev)
    if 'generator' in data.node_types and data['generator'].x.size(0) > 0:
        gen_batch = getattr(data['generator'], 'batch', None)
        if gen_batch is None:
            gen_batch = torch.zeros(data['generator'].x.size(0), dtype=torch.long, device=dev)
        Pmax_g = torch.nan_to_num(data['generator'].x[:, _GEN_PMAX], nan=0.0).to(dtype)
        Pmin_g = torch.nan_to_num(data['generator'].x[:, _GEN_PMIN], nan=0.0).to(dtype)
        Qmax_g = torch.nan_to_num(data['generator'].x[:, _GEN_QMAX], nan=0.0).to(dtype)
        Qmin_g = torch.nan_to_num(data['generator'].x[:, _GEN_QMIN], nan=0.0).to(dtype)
        sum_Pmax_per_g.scatter_add_(0, gen_batch, (Pmax_g - Pmin_g).abs())
        sum_Qrange_per_g.scatter_add_(0, gen_batch, (Qmax_g - Qmin_g).abs())
    util_P = (sum_Pd_per_g / sum_Pmax_per_g.clamp_min(cap_eps)).clamp_max(OUTPUT_CAP)
    over_P = ((sum_Pd_per_g - sum_Pmax_per_g).clamp_min(0.0) / sum_Pmax_per_g.clamp_min(cap_eps)).clamp_max(OUTPUT_CAP)
    util_Q = (sum_Qd_abs_per_g / sum_Qrange_per_g.clamp_min(cap_eps)).clamp_max(OUTPUT_CAP)
    over_Q = ((sum_Qd_abs_per_g - sum_Qrange_per_g).clamp_min(0.0) / sum_Qrange_per_g.clamp_min(cap_eps)).clamp_max(OUTPUT_CAP)
    cap_stats = torch.stack([util_P, over_P, util_Q, over_Q], dim=-1)

    violation = torch.cat([
        V_stats, KCL_P_stats, KCL_Q_stats,
        thermal_stats, angle_stats, KVL_stats,
    ], dim=-1)
    assert violation.size(-1) == STRESS_VIOL_DIMS, (
        f"violation prefix width {violation.size(-1)} != STRESS_VIOL_DIMS "
        f"({STRESS_VIOL_DIMS}); the cat layout is what a consumer slices "
        f"`[:, :STRESS_VIOL_DIMS]` against; keep violations leading."
    )
    stress = torch.cat([violation, cap_stats, loading_margin_stats], dim=-1)
    assert stress.size(-1) == STRESS_DIM, (
        f"stress feature width {stress.size(-1)} != STRESS_DIM ({STRESS_DIM}); "
        f"update STRESS_DIM if components change."
    )
    return stress


