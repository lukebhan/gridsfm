"""Self-supervised elastic AC-OPF loss on the CLOSED state.

Evaluates the elastic relaxation objective at the power-flow-closed operating
point produced by ``closure.close_batch``:

    L = sum_i c_i(Pg_i)                        # generation cost
      + rho_g * sum_j log(1 + |F_j|)           # equality slack  s^g = |F|  (two-sided)
      + rho_h * sum   log(1 + relu(h~))        # inequality slack s^h = relu(h~) (one-sided)

``F`` is the nodal power-balance residual (active at non-slack buses, reactive
at PQ buses) -- bounded by the closure's curriculum tolerance, so this term is
the feasibility gap that shrinks as tol tightens. ``h~`` collects the gen-P /
gen-Q / voltage / thermal inequality residuals of the closed state. All physics
uses the model's own ``_pi_model_flows_*`` convention, so it is consistent with
both the closure and ``compute_physics_stress``.

Operates on the flattened block-diagonal batch (the same graph the model
forward consumed), so cost and every residual are one vectorised pass.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from gridsfm.model import GridTransformerBackbone as _GTB
from gridsfm.schema import (
    BUS_TYPE_IDX, BUS_VMIN_IDX, BUS_VMAX_IDX,
    GEN_PMIN_IDX, GEN_PMAX_IDX, GEN_QMIN_IDX, GEN_QMAX_IDX, GEN_AVAIL_IDX,
    GEN_CP2_IDX, GEN_CP1_IDX, GEN_CP0_IDX,
    LOAD_PD_IDX, LOAD_QD_IDX, SHUNT_GS_IDX, SHUNT_BS_IDX,
    AC_LINE_KEY, TRANSFORMER_KEY, AC_LINE_RATE_A_IDX, TR_RATE_A_IDX,
)

_GEN_LINK = ("generator", "generator_link", "bus")
_LOAD_LINK = ("load", "load_link", "bus")
_SHUNT_LINK = ("shunt", "shunt_link", "bus")
_EPS = 1e-9


def _scatter(n, idx, val, dtype, dev):
    out = torch.zeros(n, dtype=dtype, device=dev)
    if idx.numel():
        out.scatter_add_(0, idx, val)
    return out


def _branch_flows(data, key, V, th):
    ei = data[key].edge_index
    ea = data[key].edge_attr.to(V.dtype)
    fn = (_GTB._pi_model_flows_ac_line if key == AC_LINE_KEY
          else _GTB._pi_model_flows_transformer)
    Pij, Qij, Pji, Qji = fn(ei, ea, V, th)
    return ei, ea, Pij, Qij, Pji, Qji


def elastic_loss(data: HeteroData, Vm: Tensor, theta: Tensor, pg: Tensor,
                 rho_g: float, rho_h: float,
                 cost_scale: float = 1.0) -> Tuple[Tensor, Dict[str, float]]:
    """Elastic loss on the closed state.

    Args:
      data: the batched HeteroData the model forward consumed.
      Vm, theta: closed voltages, flattened [n_bus_total] (from close_batch).
      pg: generator active controls, flattened [n_gen_total].
      cost_scale: multiplies the cost term. Cost coefficients (e.g. cp1~3000)
        dwarf the log(1+.) penalties, so a normaliser is needed for rho~10 to
        balance feasibility against cost; pass 1/nominal-cost.
    Returns (loss, diagnostics).
    """
    dev = Vm.device; dt = Vm.dtype
    bx = data["bus"].x.to(dt)
    n_bus = bx.size(0)
    bus_type = bx[:, BUS_TYPE_IDX].long()

    # --- nodal branch out-flows + per-branch apparent power (thermal) ---
    P_out = torch.zeros(n_bus, dtype=dt, device=dev)
    Q_out = torch.zeros(n_bus, dtype=dt, device=dev)
    thermal = []
    for key, rate_idx in ((AC_LINE_KEY, AC_LINE_RATE_A_IDX),
                          (TRANSFORMER_KEY, TR_RATE_A_IDX)):
        if key not in data.edge_types or data[key].edge_index.size(1) == 0:
            continue
        ei, ea, Pij, Qij, Pji, Qji = _branch_flows(data, key, Vm, theta)
        P_out.scatter_add_(0, ei[0], Pij); P_out.scatter_add_(0, ei[1], Pji)
        Q_out.scatter_add_(0, ei[0], Qij); Q_out.scatter_add_(0, ei[1], Qji)
        # rate_a <= 0 means UNLIMITED (the MATPOWER convention), not "a limit of
        # zero". Clamping it to _EPS instead turns every unrated branch into a
        # ~1e9 phantom violation: on ACTIVSg10k 2459 of 9726 ac_lines are unrated,
        # and at the GT optimum they contributed the WHOLE of ineq_pen (7.4e6,
        # against 0.0 from the genuinely rated branches). Dividing by 1.0 on those
        # branches keeps the intermediate finite, and the `where` zeroes both the
        # value and the gradient.
        rate = ea[:, rate_idx]
        limited = rate > 0
        denom = torch.where(limited, rate, torch.ones_like(rate)).clamp_min(_EPS)
        S_ij = torch.sqrt(Pij**2 + Qij**2 + 1e-12)
        S_ji = torch.sqrt(Pji**2 + Qji**2 + 1e-12)
        over = (torch.maximum(S_ij, S_ji) / denom - 1.0).clamp_min(0.0)
        thermal.append(torch.where(limited, over, torch.zeros_like(over)))
    thermal_viol = torch.cat(thermal) if thermal else torch.zeros(0, device=dev, dtype=dt)

    # --- bus shunt + loads ---
    gsV2 = torch.zeros(n_bus, dtype=dt, device=dev)
    bsV2 = torch.zeros(n_bus, dtype=dt, device=dev)
    if "shunt" in data.node_types and data["shunt"].x.size(0) > 0:
        s2b = data[_SHUNT_LINK].edge_index[1]
        shx = data["shunt"].x.to(dt)
        gsV2 = _scatter(n_bus, s2b, shx[:, SHUNT_GS_IDX] * Vm[s2b]**2, dt, dev)
        bsV2 = _scatter(n_bus, s2b, shx[:, SHUNT_BS_IDX] * Vm[s2b]**2, dt, dev)
    Pd = Qd = torch.zeros(n_bus, dtype=dt, device=dev)
    if "load" in data.node_types and data["load"].x.size(0) > 0:
        l2b = data[_LOAD_LINK].edge_index[1]
        lx = data["load"].x.to(dt)
        Pd = _scatter(n_bus, l2b, lx[:, LOAD_PD_IDX], dt, dev)
        Qd = _scatter(n_bus, l2b, lx[:, LOAD_QD_IDX], dt, dev)

    # S(V): matches the closure's Ybus (branch + bus shunt on the diagonal)
    S_real = P_out + gsV2
    S_imag = Q_out - bsV2

    # --- generator roles / injections ---
    gx = data["generator"].x.to(dt)
    online = gx[:, GEN_AVAIL_IDX] > 0.5
    g2b = data[_GEN_LINK].edge_index[1]
    has_gen = torch.zeros(n_bus, dtype=torch.bool, device=dev)
    has_gen[g2b[online]] = True
    slack = bus_type == 3
    pv = (bus_type == 2) & has_gen
    pq = ~slack & ~pv
    pvpq = ~slack

    Pg_bus = _scatter(n_bus, g2b, pg * online.to(dt), dt, dev)   # control injection

    # --- equality residual F  (two-sided |F|) ---
    F_P = S_real - (Pg_bus - Pd)                 # active balance, all buses
    F_Q = S_imag + Qd                            # reactive balance (PQ: no gen Q)
    absF = torch.cat([F_P[pvpq].abs(), F_Q[pq].abs()])
    eq_pen = rho_g * torch.log1p(absF).sum()

    # --- cost (sum c_i(pg_i) over online gens) ---
    cp2, cp1, cp0 = gx[:, GEN_CP2_IDX], gx[:, GEN_CP1_IDX], gx[:, GEN_CP0_IDX]
    cost_per = (cp2 * pg**2 + cp1 * pg + cp0) * online.to(dt)
    cost_raw = cost_per.sum()
    cost = cost_scale * cost_raw

    # --- inequalities h~ (one-sided relu) ---
    Pmin, Pmax = gx[:, GEN_PMIN_IDX], gx[:, GEN_PMAX_IDX]
    Qmin, Qmax = gx[:, GEN_QMIN_IDX], gx[:, GEN_QMAX_IDX]
    genP_viol = ((Pmin - pg).clamp_min(0.0) + (pg - Pmax).clamp_min(0.0)) * online.to(dt)

    # gen Q at pv/slack buses vs summed online-gen [Qmin,Qmax] (bus-level)
    Qg_bus = S_imag + Qd
    Qmin_bus = _scatter(n_bus, g2b, Qmin * online.to(dt), dt, dev)
    Qmax_bus = _scatter(n_bus, g2b, Qmax * online.to(dt), dt, dev)
    ctrl = pv | slack
    genQ_viol = ((Qmin_bus[ctrl] - Qg_bus[ctrl]).clamp_min(0.0)
                 + (Qg_bus[ctrl] - Qmax_bus[ctrl]).clamp_min(0.0))

    Vmin, Vmax = bx[:, BUS_VMIN_IDX], bx[:, BUS_VMAX_IDX]
    V_viol = (Vmin - Vm).clamp_min(0.0) + (Vm - Vmax).clamp_min(0.0)

    # --- RELATIVE violations: each class divided by its own physical scale, so the
    # four become commensurable percentages that can be averaged into one number.
    #   genP  / (Pmax-Pmin)   fraction of the unit's active capability exceeded
    #   genQ  / (Qmax-Qmin)   fraction of the bus's reactive capability exceeded
    #   V     / 1 p.u.        deviation as a fraction of nominal voltage
    #   therm / rate_a        already a fraction of rating by construction
    # Reported as sums + counts so the caller can form per-element means (intensive,
    # so grids of different size are comparable) without this function choosing one.
    # A unit or bus with no capability in a class cannot be scored as a FRACTION of
    # that capability: the ratio divides ~0 by ~0 and a rounding-level overshoot turns
    # into a large relative number. Both classes that normalise by a range are
    # filtered for it. Elements dropped here still contribute to the absolute penalty
    # above, they are excluded only from the relative statistic.
    #
    # THE ACTIVE-POWER FILTER WAS MISSING AND THE ERROR WAS LARGE. ACTIVSg10k carries
    # ~880 fixed-output units per case (Pmin == Pmax), so clamp_min(1e-6) supplied the
    # denominator and 1325 of ~1940 online generators registered as violating while
    # the total overshoot summed to 5e-5 MW. That reported the reference AC-OPF
    # dispatch -- which is feasible -- at 0.436% mean active-power violation, and
    # every model row was inflated the same way. Filtered, it reads 1e-6%.
    _pr_all = Pmax - Pmin
    _has_p = _pr_all > 1e-3                      # 0.1 MW on a 100 MVA base
    _qr_all = (Qmax_bus - Qmin_bus)[ctrl]
    _has_q = _qr_all > 1e-3                      # 0.1 MVAr on a 100 MVA base
    rel_P = genP_viol[_has_p] / _pr_all[_has_p].clamp_min(1e-6)
    rel_Q = genQ_viol[_has_q] / _qr_all[_has_q].clamp_min(1e-6)
    rel_V = V_viol                      # nominal is 1 p.u., so this is already relative
    rel_T = thermal_viol                # already S/rate_a - 1

    h_all = torch.cat([genP_viol, genQ_viol, V_viol, thermal_viol])
    ineq_pen = rho_h * torch.log1p(h_all).sum()

    loss = cost + eq_pen + ineq_pen
    # PER-CASE DIAGNOSTICS. cost/eq_pen/ineq_pen/loss are SUMS over the batch, so their
    # raw values scale with batch size and cannot be compared between two things
    # measured at different batch sizes. That silently corrupted every cross-source
    # comparison: the training trace runs at train.batch (4) while the validation and
    # zero-shot numbers run at eval.batch (16), so a flat cost looked like a 74.5% drop
    # and the train/val gap in the loss panel was inflated 4x -- purely from the ratio.
    #
    # The returned `loss` tensor is left as the SUM: dividing it would rescale every
    # gradient and change the effective learning rate. Only what is reported is
    # normalised.
    n_graphs = (int(data["bus"].batch.max().item()) + 1
                if hasattr(data["bus"], "batch") and data["bus"].batch is not None
                else 1)
    _pc = 1.0 / max(n_graphs, 1)
    diag = dict(
        loss=float(loss.detach()) * _pc, cost=float(cost_raw.detach()) * _pc,
        eq_pen=float(eq_pen.detach()) * _pc, ineq_pen=float(ineq_pen.detach()) * _pc,
        n_graphs=n_graphs,
        max_absF=float(absF.max().detach()) if absF.numel() else 0.0,
        max_Vviol=float(V_viol.max().detach()) if V_viol.numel() else 0.0,
        max_thermal=float(thermal_viol.max().detach()) if thermal_viol.numel() else 0.0,
        max_Qviol=float(genQ_viol.max().detach()) if genQ_viol.numel() else 0.0,
        # Counts + populations, so a caller can report the INTENSIVE 'fraction of
        # elements violating' rather than only a size-biased max: a max over more
        # elements is larger by chance alone and cannot be compared across grids.
        n_Vviol=int((V_viol > 1e-6).sum()), n_V=int(V_viol.numel()),
        n_thermal=int((thermal_viol > 1e-6).sum()), n_branch=int(thermal_viol.numel()),
        n_Qviol=int((genQ_viol > 1e-6).sum()), n_Q=int(genQ_viol.numel()),
        rel_P_sum=float(rel_P.sum()), rel_P_n=int(rel_P.numel()),
        rel_Q_sum=float(rel_Q.sum()), rel_Q_n=int(rel_Q.numel()),
        rel_V_sum=float(rel_V.sum()), rel_V_n=int(rel_V.numel()),
        rel_T_sum=float(rel_T.sum()), rel_T_n=int(rel_T.numel()),
        rel_P_max=float(rel_P.max()) if rel_P.numel() else 0.0,
        rel_Q_max=float(rel_Q.max()) if rel_Q.numel() else 0.0,
        rel_V_max=float(rel_V.max()) if rel_V.numel() else 0.0,
        rel_T_max=float(rel_T.max()) if rel_T.numel() else 0.0,
    )
    return loss, diag


def curriculum_tol(epoch: int, horizon: int, tol_start: float, tol_end: float,
                   schedule: str = "geometric") -> float:
    """Closure tolerance for a 1-indexed epoch, annealed tol_start -> tol_end over
    `horizon` epochs (a FIXED schedule, decoupled from the total epoch count so
    early stopping / equal-steps scaling do not change how fast tol tightens).
    Past the horizon it stays at tol_end."""
    if horizon <= 1:
        return tol_end
    f = min(1.0, max(0.0, (epoch - 1) / (horizon - 1)))
    if schedule == "linear":
        return tol_start + (tol_end - tol_start) * f
    import math
    return float(math.exp(math.log(tol_start) + (math.log(tol_end) - math.log(tol_start)) * f))


def curriculum_weight(epoch: int, horizon: int, w_start: float, w_end: float,
                      schedule: str = "geometric") -> float:
    """GT-anchor weight for a 1-indexed epoch, annealed w_start -> w_end over
    `horizon` epochs (same machinery as ``curriculum_tol``). Heavy-anchor-first:
    start high so the RMSE-to-GT term pulls the controls into the PF-solvable
    region (where the closure converges), then decay to a balanced end weight so
    the feasibility + cost objective shapes the final solution. Constant if
    w_start == w_end. Past the horizon it stays at w_end."""
    if w_start == w_end:
        return w_end
    return curriculum_tol(epoch, horizon, w_start, w_end, schedule)
