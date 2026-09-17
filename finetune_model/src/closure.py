"""Differentiable single-slack AC power-flow closure.

The base model emits two controls -- generator active setpoints ``p_g`` and
control-bus voltage magnitudes ``v_m``. ``CLOSE`` resolves the remaining state
(``theta`` at non-slack buses, ``V`` at PQ buses; slack ``P_G``/``Q_G`` and PV
``Q_G`` fall out as outputs) by a Newton solve of the classic single-slack AC
power-flow equations, so the reported operating point satisfies the network
physics exactly (to the Newton tolerance).

Numerics match the model's own branch-flow convention. The complex bus
admittance ``Ybus`` is built with the standard MATPOWER branch stamp
(``Yff=(y+jb_fr)/tap^2``, ``Yft=-y/conj(t)``, ``Ytf=-y/t``, ``Ytt=y+jb_to`` with
``t=tap*e^{j*shift}``); ``build_ybus`` is unit-tested to reproduce the nodal
injections obtained by summing ``gridsfm.model._pi_model_flows_*`` (see
``tests/``).

Autograd: ``ACPFClosure`` runs Newton detached in the forward, caches the sparse
LU factor of the final Jacobian, and in the backward solves the adjoint system
with the cached factor transposed -- O(one sparse solve), no unrolling. Only
``V_m``/``theta`` come through the custom Function; every downstream quantity
(branch flows, ``Q_g``, slack ``P_g``, all inequality residuals) is a plain
differentiable torch function of ``(V_m, theta)`` and is left to ordinary
autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, SuperLU
import torch
from torch_geometric.data import HeteroData

from gridsfm.schema import (
    BUS_TYPE_IDX, BUS_VMIN_IDX, BUS_VMAX_IDX,
    GEN_PMIN_IDX, GEN_PMAX_IDX, GEN_QMIN_IDX, GEN_QMAX_IDX, GEN_AVAIL_IDX,
    GEN_CP2_IDX, GEN_CP1_IDX, GEN_CP0_IDX,
    LOAD_PD_IDX, LOAD_QD_IDX, SHUNT_GS_IDX, SHUNT_BS_IDX,
    AC_LINE_KEY, TRANSFORMER_KEY,
    AC_LINE_BFR_IDX, AC_LINE_BTO_IDX, AC_LINE_R_IDX, AC_LINE_X_IDX,
    TR_R_IDX, TR_X_IDX, TR_TAP_IDX, TR_SHIFT_IDX, TR_BFR_IDX, TR_BTO_IDX,
    PI_Z2_EPS, PI_TAP_EPS,
)

_GEN_LINK = ("generator", "generator_link", "bus")
_LOAD_LINK = ("load", "load_link", "bus")
_SHUNT_LINK = ("shunt", "shunt_link", "bus")


# ------------------------------------------------------------------ extraction

@dataclass
class Case:
    """Static per-topology + per-scenario arrays for one grid (float64/int64)."""
    n_bus: int
    n_gen: int
    # branch stamp inputs (concatenated ac_line ++ transformer)
    f: np.ndarray            # from-bus index per branch
    t: np.ndarray            # to-bus index per branch
    y: np.ndarray            # series admittance (complex)
    b_fr: np.ndarray         # from-side shunt susceptance
    b_to: np.ndarray         # to-side shunt susceptance
    tap: np.ndarray          # complex tap ratio t = tap*e^{j shift}
    ysh: np.ndarray          # per-bus shunt admittance gs + j bs (complex)
    # scenario injections
    Pd: np.ndarray           # per-bus active load
    Qd: np.ndarray           # per-bus reactive load
    gen2bus: np.ndarray      # bus index per generator
    gen_online: np.ndarray   # bool per generator
    # bus roles
    ref: np.ndarray          # slack bus indices
    pv: np.ndarray           # PV (regulating) bus indices
    pq: np.ndarray           # PQ bus indices
    pvpq: np.ndarray         # non-slack bus indices (pv ++ pq), the theta unknowns
    ctrl_bus: np.ndarray     # buses whose V is a control (pv ++ ref), sorted
    # limits / cost (per gen unless noted)
    Vmin: np.ndarray
    Vmax: np.ndarray
    Pmin: np.ndarray
    Pmax: np.ndarray
    Qmin: np.ndarray
    Qmax: np.ndarray
    cp2: np.ndarray
    cp1: np.ndarray
    cp0: np.ndarray
    Ybus: sp.csr_matrix = None


def _agg_to_bus(n_bus, link_ei, vals):
    out = np.zeros(n_bus, dtype=np.float64)
    if link_ei is not None and link_ei.shape[1] > 0:
        np.add.at(out, link_ei[1], vals)
    return out


def extract_case(d: HeteroData) -> Case:
    """Pull a single-graph HeteroData into numpy arrays for the PF closure."""
    bx = d["bus"].x.detach().cpu().numpy().astype(np.float64)
    n_bus = bx.shape[0]
    bus_type = bx[:, BUS_TYPE_IDX].astype(np.int64)
    Vmin = bx[:, BUS_VMIN_IDX].copy()
    Vmax = bx[:, BUS_VMAX_IDX].copy()

    gx = d["generator"].x.detach().cpu().numpy().astype(np.float64)
    n_gen = gx.shape[0]
    gen_ei = d[_GEN_LINK].edge_index.detach().cpu().numpy()
    gen2bus = np.zeros(n_gen, dtype=np.int64)
    gen2bus[gen_ei[0]] = gen_ei[1]
    gen_online = gx[:, GEN_AVAIL_IDX] > 0.5

    # loads / shunts aggregated to bus
    Pd = Qd = np.zeros(n_bus)
    if "load" in d.node_types and d["load"].x.size(0) > 0:
        lx = d["load"].x.detach().cpu().numpy().astype(np.float64)
        lei = d[_LOAD_LINK].edge_index.detach().cpu().numpy()
        Pd = _agg_to_bus(n_bus, lei, lx[:, LOAD_PD_IDX])
        Qd = _agg_to_bus(n_bus, lei, lx[:, LOAD_QD_IDX])
    gs = bs = np.zeros(n_bus)
    if "shunt" in d.node_types and d["shunt"].x.size(0) > 0:
        shx = d["shunt"].x.detach().cpu().numpy().astype(np.float64)
        sei = d[_SHUNT_LINK].edge_index.detach().cpu().numpy()
        gs = _agg_to_bus(n_bus, sei, shx[:, SHUNT_GS_IDX])
        bs = _agg_to_bus(n_bus, sei, shx[:, SHUNT_BS_IDX])
    ysh = gs + 1j * bs

    # branch stamp inputs, ac_line ++ transformer
    fs, ts, ys, bfrs, btos, taps = [], [], [], [], [], []
    for key, r_i, x_i, bfr_i, bto_i, tap_i, sh_i in (
        (AC_LINE_KEY, AC_LINE_R_IDX, AC_LINE_X_IDX, AC_LINE_BFR_IDX,
         AC_LINE_BTO_IDX, None, None),
        (TRANSFORMER_KEY, TR_R_IDX, TR_X_IDX, TR_BFR_IDX, TR_BTO_IDX,
         TR_TAP_IDX, TR_SHIFT_IDX),
    ):
        if key not in d.edge_types or d[key].edge_index.size(1) == 0:
            continue
        ei = d[key].edge_index.detach().cpu().numpy()
        ea = d[key].edge_attr.detach().cpu().numpy().astype(np.float64)
        r = ea[:, r_i]; x = ea[:, x_i]
        denom = r * r + x * x + PI_Z2_EPS
        yb = (r - 1j * x) / denom                       # = 1/(r + jx)
        if tap_i is None:
            tp = np.ones(ei.shape[1], dtype=np.complex128)
        else:
            mag = ea[:, tap_i]
            mag = np.where(np.abs(mag) < PI_TAP_EPS, 1.0, mag)
            tp = mag * np.exp(1j * ea[:, sh_i])
        fs.append(ei[0]); ts.append(ei[1]); ys.append(yb)
        bfrs.append(ea[:, bfr_i]); btos.append(ea[:, bto_i]); taps.append(tp)

    f = np.concatenate(fs) if fs else np.zeros(0, np.int64)
    t = np.concatenate(ts) if ts else np.zeros(0, np.int64)
    y = np.concatenate(ys) if ys else np.zeros(0, np.complex128)
    b_fr = np.concatenate(bfrs) if bfrs else np.zeros(0)
    b_to = np.concatenate(btos) if btos else np.zeros(0)
    tap = np.concatenate(taps) if taps else np.zeros(0, np.complex128)

    # bus roles: slack = type 3, PV = type 2 with an online regulating gen, else PQ
    has_online_gen = np.zeros(n_bus, dtype=bool)
    np.logical_or.at(has_online_gen, gen2bus[gen_online], True)
    ref = np.where(bus_type == 3)[0]
    pv = np.where((bus_type == 2) & has_online_gen)[0]
    is_ref = np.zeros(n_bus, dtype=bool); is_ref[ref] = True
    is_pv = np.zeros(n_bus, dtype=bool); is_pv[pv] = True
    pq = np.where(~is_ref & ~is_pv)[0]
    pvpq = np.sort(np.concatenate([pv, pq]))
    ctrl_bus = np.sort(np.concatenate([pv, ref]))

    case = Case(
        n_bus=n_bus, n_gen=n_gen, f=f, t=t, y=y, b_fr=b_fr, b_to=b_to, tap=tap,
        ysh=ysh, Pd=Pd, Qd=Qd, gen2bus=gen2bus, gen_online=gen_online,
        ref=ref, pv=pv, pq=pq, pvpq=pvpq, ctrl_bus=ctrl_bus,
        Vmin=Vmin, Vmax=Vmax,
        Pmin=gx[:, GEN_PMIN_IDX].copy(), Pmax=gx[:, GEN_PMAX_IDX].copy(),
        Qmin=gx[:, GEN_QMIN_IDX].copy(), Qmax=gx[:, GEN_QMAX_IDX].copy(),
        cp2=gx[:, GEN_CP2_IDX].copy(), cp1=gx[:, GEN_CP1_IDX].copy(),
        cp0=gx[:, GEN_CP0_IDX].copy(),
    )
    case.Ybus = build_ybus(case)
    return case


# ------------------------------------------------------------------ admittance

def build_ybus(c: Case) -> sp.csr_matrix:
    """Complex bus admittance matrix, standard MATPOWER branch stamp + bus shunts."""
    n = c.n_bus
    if c.f.size == 0:
        return sp.csr_matrix((sp.eye(n) * 0).astype(np.complex128)) + sp.diags(c.ysh)
    tap = c.tap
    Yff = (c.y + 1j * c.b_fr) / (np.abs(tap) ** 2)
    Yft = -c.y / np.conj(tap)
    Ytf = -c.y / tap
    Ytt = c.y + 1j * c.b_to
    # 4 stamps per branch: (f,f),(f,t),(t,f),(t,t)
    rows = np.concatenate([c.f, c.f, c.t, c.t])
    cols = np.concatenate([c.f, c.t, c.f, c.t])
    vals = np.concatenate([Yff, Yft, Ytf, Ytt])
    Ybus = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128)
    Ybus = Ybus + sp.diags(c.ysh)
    return Ybus.tocsr()


def _dSbus_dV(Ybus: sp.csr_matrix, V: np.ndarray):
    """MATPOWER polar dSbus_dVm, dSbus_dVa (complex sparse)."""
    Ibus = Ybus @ V
    diagV = sp.diags(V)
    diagIbus = sp.diags(Ibus)
    diagVnorm = sp.diags(V / np.abs(V))
    dS_dVm = diagV @ (Ybus @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
    dS_dVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conjugate()
    return dS_dVm.tocsr(), dS_dVa.tocsr()


# ------------------------------------------------------------------ Newton PF

@dataclass
class PFResult:
    V: np.ndarray             # complex bus voltage at the solution
    converged: bool
    iters: int
    norm_hist: list
    lu: SuperLU               # cached factor of the final Jacobian J = dF/dy
    dS_dVm: sp.csr_matrix     # cached at the solution (for the adjoint's ctrl cols)
    Ssp: np.ndarray           # specified complex injection used
    J: sp.csc_matrix = None   # the final Jacobian itself (picklable, SuperLU is not)
    gpu_factor: object = None  # live cuDSS factor (cudss backend only, cached by the
                               # caller across epochs -- never pickled/crosses processes)


def _mismatch(Ybus, V, Ssp, pvpq, pq):
    S = V * np.conj(Ybus @ V)
    mis = S - Ssp
    return np.concatenate([mis[pvpq].real, mis[pq].imag])


def newton_pf(c: Case, Ssp: np.ndarray, V0: np.ndarray, tol: float,
              max_iter: int = 30, backtrack: bool = True,
              backend: str = "superlu", gpu_factor=None) -> PFResult:
    """Single-slack Newton power flow with step-halving line search. Unknowns:
    Va[pvpq], Vm[pq].

    ``backend="superlu"`` factors each Jacobian with scipy SuperLU (CPU) and caches
    the factor for the adjoint. ``backend="cudss"`` factors on the GPU via cuDSS,
    reusing the case's cached symbolic plan (``gpu_factor``) across epochs; the live
    factor is returned in ``res.gpu_factor`` for the caller to cache. A singular
    Jacobian (voltage collapse under bad controls) marks the case non-converged
    rather than crashing -- its (non-root) adjoint gradient is dropped and the
    anchor drives it back toward feasibility. ``res.J``/``res.dS_dVm`` carry the
    solution-point data for the adjoint.
    """
    pvpq, pq = c.pvpq, c.pq
    npv_pq = pvpq.size
    Vm = np.abs(V0).copy()
    Va = np.angle(V0).copy()
    V = Vm * np.exp(1j * Va)

    F = _mismatch(c.Ybus, V, Ssp, pvpq, pq)
    norm_hist = [float(np.max(np.abs(F))) if F.size else 0.0]
    lu = None
    dS_dVm = None
    Jfinal = None
    gpu_fac = gpu_factor                          # reuse the cached plan if provided
    singular = False
    it = 0
    converged = norm_hist[-1] <= tol
    while not converged and it < max_iter:
        dS_dVm, dS_dVa = _dSbus_dV(c.Ybus, V)
        j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        j12 = dS_dVm[np.ix_(pvpq, pq)].real
        j21 = dS_dVa[np.ix_(pq, pvpq)].imag
        j22 = dS_dVm[np.ix_(pq, pq)].imag
        J = sp.bmat([[j11, j12], [j21, j22]], format="csc")
        Jfinal = J
        try:
            if backend == "cudss":
                from gpu_solver import CudssFactor
                if gpu_fac is None:
                    gpu_fac = CudssFactor(J)      # plan once, cached by the caller
                gpu_fac.refactor(J)               # refactor in place (values only)
                dx = gpu_fac.solve(-F)
            else:
                lu = splu(J)
                dx = lu.solve(-F)
        except RuntimeError:
            singular = True
            break
        step = 1.0
        base = norm_hist[-1]
        while True:
            Va_n = Va.copy(); Vm_n = Vm.copy()
            Va_n[pvpq] += step * dx[:npv_pq]
            Vm_n[pq] += step * dx[npv_pq:]
            V_n = Vm_n * np.exp(1j * Va_n)
            F_n = _mismatch(c.Ybus, V_n, Ssp, pvpq, pq)
            nn = float(np.max(np.abs(F_n)))
            if (not backtrack) or nn < base or step < 1e-4:
                break
            step *= 0.5
        Va, Vm, V, F = Va_n, Vm_n, V_n, F_n
        it += 1
        norm_hist.append(nn)
        converged = nn <= tol
    if singular:
        converged = False                          # non-root: skipped in the adjoint
    if Jfinal is None:  # already converged at the warm start: still emit J / dS_dVm
        dS_dVm, dS_dVa = _dSbus_dV(c.Ybus, V)
        j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        j12 = dS_dVm[np.ix_(pvpq, pq)].real
        j21 = dS_dVa[np.ix_(pq, pvpq)].imag
        j22 = dS_dVm[np.ix_(pq, pq)].imag
        Jfinal = sp.bmat([[j11, j12], [j21, j22]], format="csc")
    if backend != "cudss" and lu is None and not singular:  # CPU single-Function factor
        try:
            lu = splu(Jfinal)
        except RuntimeError:
            singular = True; converged = False
    # The cuDSS factor is NOT freed here: the ClosurePool caches it per case and
    # reuses its symbolic plan across epochs (freed on pool.shutdown()).
    return PFResult(V=V, converged=converged, iters=it, norm_hist=norm_hist,
                    lu=lu, dS_dVm=dS_dVm, Ssp=Ssp, J=Jfinal,
                    gpu_factor=(gpu_fac if backend == "cudss" else None))


def build_Ssp(c: Case, pg: np.ndarray, qg_bus: Optional[np.ndarray] = None) -> np.ndarray:
    """Specified complex injection Ssp = (Pg_bus - Pd) + j(-Qd). Slack P/Q and PV Q
    entries are placeholders (not used in the pvpq/pq mismatch)."""
    Pg_bus = np.zeros(c.n_bus)
    np.add.at(Pg_bus, c.gen2bus, pg * c.gen_online)
    Psp = Pg_bus - c.Pd
    Qsp = -c.Qd
    return Psp + 1j * Qsp


def dc_angles(c: Case, Ssp: np.ndarray) -> np.ndarray:
    """Standard (MATPOWER-style) DC power-flow bus angles: solve B' theta = P for
    the non-slack buses, slack fixed at 0. B' uses the series susceptance 1/x
    scaled by the tap magnitude; phase-shifters enter as fixed power injections.

    A flat V=1 start does not converge AC Newton on large grids (~6k+ buses), but
    a flat magnitude with THESE angles lands in the basin (~4-5 iters). Topology-
    only (independent of the operating point except P), so it is a cheap, robust
    cold-start initialiser."""
    if c.f.size == 0:
        return np.zeros(c.n_bus)
    x = np.imag(1.0 / c.y)                     # series reactance from 1/(r+jx)
    tapmag = np.abs(c.tap).copy(); tapmag[tapmag < 1e-6] = 1.0
    b = 1.0 / (x * tapmag)                     # DC branch susceptance
    shift = np.angle(c.tap)
    n = c.n_bus
    rows = np.concatenate([c.f, c.f, c.t, c.t])
    cols = np.concatenate([c.f, c.t, c.f, c.t])
    vals = np.concatenate([b, -b, -b, b])
    Bp = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Pshift = np.zeros(n)
    np.add.at(Pshift, c.f, -b * shift)
    np.add.at(Pshift, c.t, b * shift)
    rhs = (np.real(Ssp) - Pshift)[c.pvpq]
    th = np.zeros(n)
    nn = c.pvpq
    Bnn = Bp[np.ix_(nn, nn)].tocsc()
    try:
        th[nn] = splu(Bnn).solve(rhs)
    except Exception:
        th[nn] = 0.0                           # singular B' (islanded) -> flat fallback
    return th


def dc_init_V(c: Case, pg: np.ndarray, vm: Optional[np.ndarray] = None) -> np.ndarray:
    """Cold-start voltage guess: the model's magnitude ``vm`` (flat 1.0 if None)
    with DC power-flow angles. Used ONCE per case to bootstrap the first solve;
    every later epoch warm-starts from that epoch's cached converged point."""
    th = dc_angles(c, build_Ssp(c, pg))
    mag = np.ones(c.n_bus) if vm is None else np.asarray(vm, dtype=np.float64)
    return mag * np.exp(1j * th)


def flat_init_V(c: Case) -> np.ndarray:
    """TEXTBOOK FLAT START: V = 1.0 angle 0 at every bus. Ignores both the model's
    voltage magnitudes and the DC angles.

    Provided for ``closure.cold_init: flat``. Be aware what it costs on a large grid:
    the DC angles are load-bearing here. Measured on case6470_rte, feeding the model's
    controls into this start converged 0 of 60 cases at every tolerance down to 1e-01,
    against 3% for DC angles + model magnitudes and 94% for DC angles + flat magnitudes
    at PQ buses. The angle spread across 6470 buses is simply too large for Newton's
    basin from a uniform zero.
    """
    return np.ones(c.n_bus, dtype=np.complex128)


# ------------------------------------------------------------------ autograd

def _adjoint_grads(c: Case, solve_T, dS_dVm: sp.csr_matrix,
                   gVm: np.ndarray, gVa: np.ndarray):
    """Map upstream grads on (Vm, Va) to grads on (pg, vm_ctrl) via the adjoint.

    ``solve_T(b)`` returns the solution of ``J^T x = b`` -- a cached-factor
    transpose-solve on CPU (SuperLU) or a fresh cuDSS ``J^T`` factor on GPU. Then:
      grad_pg[g]   = mu_P at gen g's (pvpq) bus         (pg enters Ssp.real)
      grad_vm[ctrl]= gVm[ctrl] - (dF/dvm_ctrl)^T mu     (direct + implicit)
    """
    pvpq, pq, ctrl = c.pvpq, c.pq, c.ctrl_bus
    npv_pq = pvpq.size
    gy = np.concatenate([gVa[pvpq], gVm[pq]])
    mu = solve_T(gy)
    mu_P = mu[:npv_pq]

    pos = -np.ones(c.n_bus, dtype=np.int64)
    pos[pvpq] = np.arange(npv_pq)
    grad_pg = np.zeros(c.n_gen)
    sel = c.gen_online & (pos[c.gen2bus] >= 0)
    grad_pg[sel] = mu_P[pos[c.gen2bus[sel]]]

    blk = sp.vstack([dS_dVm[np.ix_(pvpq, ctrl)].real,
                     dS_dVm[np.ix_(pq, ctrl)].imag]).tocsr()
    grad_vmc = gVm[ctrl] - (blk.T @ mu)
    return grad_pg, grad_vmc


def _merit_grads(c: Case, dS_dVm: sp.csr_matrix, F: np.ndarray):
    """Gradient of the merit ``m(u) = 0.5 * mean(F(y,u)^2)`` w.r.t. the controls, at a
    NON-ROOT iterate: ``(dF/du)^T F / n``.

    This is what to use where the implicit-function adjoint is INVALID. The IFT adjoint
    assumes ``F(y*,u) = 0``; when the Newton solve stalls it is linearising about a point
    that is not a solution, so the old code dropped those cases and returned zero. On a
    hard grid that is not a rare rescue path -- measured on case6470_rte, the base
    checkpoint stalls on 60/60 training cases at every tolerance down to 1e-01 (and still
    0/20 at a 300-iteration cap), so EVERY case contributed zero closure gradient and
    training degenerated to pure GT regression with no signal from the physics at all.

    m is well defined at any iterate. Because step-halving stalled, ``y`` is approximately
    stationary in ``y``, so by the envelope theorem the total derivative collapses to the
    partial one -- no linear solve, no factorisation, one sparse matvec:

      * ``dF_P/dpg = -1`` on the generator's own mismatch row (pg enters Ssp.real), and
      * ``dF/dvm_ctrl`` is the control-voltage column slice of the polar Jacobian that
        was already formed for the Newton step.

    Normalised by the residual count, matching the forward value in ACPFClosure -- the
    two MUST use the same convention, since dividing one and not the other silently
    rescales the term by F.size.

    It SELF-ANNEALS: as a case approaches feasibility F -> 0 and the term
    vanishes on its own, which is the job the hand-tuned w_control curriculum kept
    failing to do.
    """
    pvpq, pq, ctrl = c.pvpq, c.pq, c.ctrl_bus
    npv_pq = pvpq.size
    n = max(F.size, 1)
    Fp = F[:npv_pq]

    pos = -np.ones(c.n_bus, dtype=np.int64)
    pos[pvpq] = np.arange(npv_pq)
    grad_pg = np.zeros(c.n_gen)
    sel = c.gen_online & (pos[c.gen2bus] >= 0)
    grad_pg[sel] = -Fp[pos[c.gen2bus[sel]]] / n          # dF_P/dpg = -1

    blk = sp.vstack([dS_dVm[np.ix_(pvpq, ctrl)].real,
                     dS_dVm[np.ix_(pq, ctrl)].imag]).tocsr()
    grad_vmc = (blk.T @ F) / n
    return grad_pg, grad_vmc


class ACPFClosure(torch.autograd.Function):
    """Differentiable single-slack Newton PF. Inputs: pg (n_gen), vm_ctrl
    (n_ctrl, aligned to case.ctrl_bus). Outputs: Vm (n_bus), Va (n_bus)."""

    @staticmethod
    def forward(ctx, pg: torch.Tensor, vm_ctrl: torch.Tensor,
                case: Case, V0: np.ndarray, tol: float, max_iter: int,
                backtrack: bool):
        dev, dt = pg.device, pg.dtype
        pg_np = pg.detach().cpu().numpy().astype(np.float64)
        vmc = vm_ctrl.detach().cpu().numpy().astype(np.float64)

        Vm0 = np.abs(V0).copy()
        Vm0[case.ctrl_bus] = vmc                       # controls fix V at pv+ref
        V0f = Vm0 * np.exp(1j * np.angle(V0))
        Ssp = build_Ssp(case, pg_np)

        res = newton_pf(case, Ssp, V0f, tol=tol, max_iter=max_iter,
                        backtrack=backtrack)

        ctx.case = case
        ctx.res = res
        ctx.dev = dev
        ctx.dt = dt
        Vm = torch.as_tensor(np.abs(res.V), device=dev, dtype=dt)
        Va = torch.as_tensor(np.angle(res.V), device=dev, dtype=dt)
        # stash convergence for the caller (not a differentiable output)
        ctx.converged = res.converged
        ctx.iters = res.iters
        return Vm, Va

    @staticmethod
    def backward(ctx, gVm: torch.Tensor, gVa: torch.Tensor):
        res: PFResult = ctx.res
        solve_T = lambda b: res.lu.solve(b, trans="T")
        grad_pg, grad_vmc = _adjoint_grads(
            ctx.case, solve_T, res.dS_dVm,
            gVm.detach().cpu().numpy().astype(np.float64),
            gVa.detach().cpu().numpy().astype(np.float64))
        gp = torch.as_tensor(grad_pg, device=ctx.dev, dtype=ctx.dt)
        gv = torch.as_tensor(grad_vmc, device=ctx.dev, dtype=ctx.dt)
        return gp, gv, None, None, None, None, None


def close(pg: torch.Tensor, vm_ctrl: torch.Tensor, case: Case,
          V0: np.ndarray, tol: float, max_iter: int = 30, backtrack: bool = True):
    """Convenience wrapper returning (Vm, Va) with autograd wired through."""
    return ACPFClosure.apply(pg, vm_ctrl, case, V0, tol, max_iter, backtrack)


# ------------------------------------------------------- parallel batched path

def _forward_worker(args):
    """Top-level (picklable) forward solve. Returns only picklable pieces -- the
    factor cannot cross the process boundary (and cuDSS runs in-process anyway),
    so the backward re-factors the returned sparse ``J``. Non-convergence is
    handled downstream: its state is NOT cached as a warm start and its (non-root)
    adjoint gradient is dropped, so the anchor drives such cases toward the
    feasible region. ``backend`` selects the linear solver (superlu / cudss)."""
    case, pg, vm_ctrl, V0, tol, max_iter, backtrack, backend = args
    Vm0 = np.abs(V0).copy()
    Vm0[case.ctrl_bus] = vm_ctrl
    res = newton_pf(case, build_Ssp(case, pg), Vm0 * np.exp(1j * np.angle(V0)),
                    tol=tol, max_iter=max_iter, backtrack=backtrack, backend=backend)
    return (np.abs(res.V), np.angle(res.V), res.J, res.dS_dVm,
            res.iters, res.converged)


class ClosurePool:
    """Persistent pool for the per-scenario forward solves.

    ``backend="superlu"`` (default) fans independent scipy SuperLU solves across
    ``workers`` CPU cores (``workers <= 1`` runs serially in-process). ``backend
    ="cudss"`` runs the solves SERIALLY in the main process on a single GPU cuDSS
    context (``workers`` is ignored) -- a process pool would fork the CUDA
    context, and for large grids the GPU factorization is the win, not cross-case
    CPU parallelism.

    For the cudss backend the pool caches one live ``CudssFactor`` per case (keyed
    by the ``Case`` object identity, which the trainer reuses across epochs) so the
    expensive symbolic *plan* is computed once per case for the whole run and only
    the numeric factorization is redone each epoch -- separately for the forward
    ``J`` and the adjoint ``J^T``. (This holds per-case GPU factors resident; for
    very large grids x large training sets that is real GPU memory.)
    """

    def __init__(self, workers: int = 0, backend: str = "superlu"):
        self.backend = str(backend)
        self.workers = int(workers)
        self.last_iters: list = []
        self.last_converged: list = []
        self._ex = None
        self._fwd_plans: dict = {}   # id(case) -> CudssFactor (forward J, plan cached)
        self._adj_plans: dict = {}   # id(case) -> CudssFactor (adjoint J^T, plan cached)
        if self.backend != "cudss" and self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            self._ex = ProcessPoolExecutor(max_workers=self.workers)

    def solve(self, tasks):
        if self.backend == "cudss":
            return self._solve_cudss(tasks)
        if self._ex is None:
            return [_forward_worker(t) for t in tasks]
        return list(self._ex.map(_forward_worker, tasks))

    def _solve_cudss(self, tasks):
        """Serial in-process forward solves, reusing each case's cached cuDSS plan."""
        out = []
        for case, pg, vm_ctrl, V0, tol, max_iter, backtrack, _backend in tasks:
            Vm0 = np.abs(V0).copy()
            Vm0[case.ctrl_bus] = vm_ctrl
            res = newton_pf(case, build_Ssp(case, pg),
                            Vm0 * np.exp(1j * np.angle(V0)),
                            tol=tol, max_iter=max_iter, backtrack=backtrack,
                            backend="cudss", gpu_factor=self._fwd_plans.get(id(case)))
            if res.gpu_factor is not None:
                self._fwd_plans[id(case)] = res.gpu_factor   # keep the plan for next epoch
            out.append((np.abs(res.V), np.angle(res.V), res.J, res.dS_dVm,
                        res.iters, res.converged))
        return out

    def adjoint_factor(self, case: Case, J):
        """Return the case's cached ``J^T`` cuDSS factor, refactored for this J
        (plans once per case; reused across epochs). cudss backend only."""
        from gpu_solver import CudssFactor
        JT = J.transpose().tocsr()
        fac = self._adj_plans.get(id(case))
        if fac is None:
            fac = CudssFactor(JT)
            self._adj_plans[id(case)] = fac
        fac.refactor(JT)
        return fac

    def shutdown(self):
        for cache in (self._fwd_plans, self._adj_plans):
            for fac in cache.values():
                fac.free()
            cache.clear()
        if self._ex is not None:
            self._ex.shutdown()
            self._ex = None


class ACPFClosureBatch(torch.autograd.Function):
    """Batched closure over B scenarios (same topology, but the PV/slack control
    set may differ per scenario with generator availability).

    Inputs are the FLAT model outputs: ``pg_flat`` [sum n_gen], ``vm_flat``
    [sum n_bus] (per-bus voltage). Each case is sliced by its own node offsets
    and ``ctrl_bus``, so a ragged control set is handled naturally. Outputs are
    ``Vm_flat`` / ``Va_flat`` [sum n_bus], matching the block-diagonal batch the
    model consumed. Forward solves fan across ``pool``; backward re-factors each
    returned Jacobian and runs the adjoint per case.
    """

    @staticmethod
    def forward(ctx, pg_flat, vm_flat, pool, cases, V0s, tol, max_iter, backtrack,
                merit_log=False):
        boff = np.concatenate([[0], np.cumsum([c.n_bus for c in cases])])
        goff = np.concatenate([[0], np.cumsum([c.n_gen for c in cases])])
        pg_np = pg_flat.detach().cpu().numpy().astype(np.float64)
        vm_np = vm_flat.detach().cpu().numpy().astype(np.float64)
        tasks = []
        for k, c in enumerate(cases):
            pg_k = pg_np[goff[k]:goff[k+1]]
            vm_ctrl_k = vm_np[boff[k]:boff[k+1]][c.ctrl_bus]
            tasks.append((c, pg_k, vm_ctrl_k, V0s[k], tol, max_iter, backtrack,
                          pool.backend))
        results = pool.solve(tasks)
        Vm = np.concatenate([r[0] for r in results])
        Va = np.concatenate([r[1] for r in results])
        # Per-case power-balance residual at the returned iterate, and the merit
        # m = 0.5 * mean(F^2). For a converged case F ~ 0 and both the value and its
        # gradient are ~0, so this costs one sparse matvec and changes nothing, for a
        # STALLED case it is the only well-defined gradient available (see _merit_grads).
        Fs, merits, merit_scale = [], [], []
        for k, c in enumerate(cases):
            Vk = results[k][0] * np.exp(1j * results[k][1])
            Fk = _mismatch(c.Ybus, Vk, build_Ssp(c, pg_np[goff[k]:goff[k+1]]),
                           c.pvpq, c.pq)
            Fs.append(Fk)
            # MEAN over residual entries. NOTE this makes merit the one INTENSIVE term
            # in an objective whose other penalties are sums (elastic_loss.py:
            # `log1p(absF).sum()`, `log1p(h_all).sum()`), so it carries an ~F.size
            # handicap against them -- ~900x on case500_goc, ~18,000x on activsg10k --
            # and that handicap is grid-size dependent, which is why a w_merit tuned on
            # one grid does not transfer to another. Summing it instead was tried and
            # REVERTED: at a 50% merit gradient share the closure collapsed from 0.70 to
            # 0.000-0.333 and test Pg went from 12.8% to 23.6%. The mean convention is
            # what every shipped w_merit was tuned under, keep them consistent.
            m_k = 0.5 * float(Fk @ Fk) / max(Fk.size, 1)
            # LOG-COMPRESSED MERIT (merit_log): rho_r * log(1 + ||r||^2/(2 n_r)), i.e.
            # log1p of the plain half-mean-square, matching how elastic_loss already
            # penalises the equality and inequality residuals (log1p(absF).sum(),
            # log1p(h_all).sum()).
            #
            # WHY. The plain merit spans an absurd dynamic range -- measured across one
            # run, 2.4e-16 to 5.6e+03, a factor of 2.4e19 -- so a handful of badly
            # stalled cases dominate it entirely, and those are exactly the cases where
            # the envelope-theorem gradient is least trustworthy (it assumes the Newton
            # has stalled with y stationary, and at ||F|| ~ 1e3 the returned iterate is
            # nowhere near a solution). log1p caps that: d/dm log(1+m) = 1/(1+m), which
            # at the observed maximum is 1.8e-4, so a hopeless case can no longer own
            # the update. Near convergence log(1+m) ~ m, so the self-annealing that
            # makes w_control a constant is preserved exactly.
            #
            # The chain factor is applied in backward() rather than by differentiating
            # through log1p, because _merit_grads returns dm/du analytically.
            merits.append(float(np.log1p(m_k)) if merit_log else m_k)
            merit_scale.append(1.0 / (1.0 + m_k) if merit_log else 1.0)
        ctx.merit_scale = merit_scale
        ctx.Fs = Fs
        ctx.results = results; ctx.cases = cases
        ctx.boff = boff; ctx.goff = goff
        ctx.backend = pool.backend
        ctx.pool = pool          # for the cudss adjoint's per-case cached J^T plan
        ctx.dev, ctx.dt = pg_flat.device, pg_flat.dtype
        pool.last_iters = [r[4] for r in results]
        pool.last_converged = [r[5] for r in results]
        return (torch.as_tensor(Vm, device=ctx.dev, dtype=ctx.dt),
                torch.as_tensor(Va, device=ctx.dev, dtype=ctx.dt),
                torch.as_tensor(np.asarray(merits), device=ctx.dev, dtype=ctx.dt))

    @staticmethod
    def backward(ctx, gVm, gVa, gmerit):
        gVm_np = gVm.detach().cpu().numpy().astype(np.float64)
        gVa_np = gVa.detach().cpu().numpy().astype(np.float64)
        boff, goff = ctx.boff, ctx.goff
        grad_pg = np.zeros(int(goff[-1]))
        grad_vm = np.zeros(int(boff[-1]))
        backend = getattr(ctx, "backend", "superlu")
        gm = (np.zeros(len(ctx.results)) if gmerit is None
              else gmerit.detach().cpu().numpy().astype(np.float64))
        for k, (_, _, J, dS_dVm, _, converged) in enumerate(ctx.results):
            c = ctx.cases[k]
            # MERIT TERM, on every case. For a converged case F ~ 0 so this is ~0 and the
            # adjoint below carries the gradient as before. For a STALLED case the adjoint
            # is invalid and skipped, and this is the only gradient the case contributes
            # -- previously it contributed nothing at all.
            if gm[k] != 0.0 and dS_dVm is not None:
                mp, mv = _merit_grads(c, dS_dVm, ctx.Fs[k])
                # d/du log(1+m) = (1/(1+m)) dm/du. Scale is 1.0 when merit_log is off.
                _ms = ctx.merit_scale[k]
                grad_pg[ctx.goff[k]:ctx.goff[k+1]] += gm[k] * _ms * mp
                grad_vm[ctx.boff[k]:ctx.boff[k+1]][c.ctrl_bus] += gm[k] * _ms * mv
            if not converged:
                # non-root: the implicit-function adjoint assumes F(y*,u) = 0 and is
                # invalid here, so it is still NOT propagated. The merit gradient above
                # replaces it with a valid feasibility-descent direction.
                continue
            try:
                if backend == "cudss":
                    fac = ctx.pool.adjoint_factor(c, J)  # cached J^T plan, refactored
                    solve_T = fac.solve                  # pool owns it, do not free
                else:
                    lu = splu(J.tocsc())                 # re-factor for the adjoint
                    solve_T = lambda b, lu=lu: lu.solve(b, trans="T")
            except RuntimeError:
                continue                                 # singular adjoint -> drop this
                                                         # case's grad (anchor handles it)
            gp, gv = _adjoint_grads(c, solve_T, dS_dVm,
                                    gVm_np[boff[k]:boff[k+1]],
                                    gVa_np[boff[k]:boff[k+1]])
            grad_pg[goff[k]:goff[k+1]] = gp
            grad_vm[boff[k]:boff[k+1]][c.ctrl_bus] = gv   # scatter into control slots
        return (torch.as_tensor(grad_pg, device=ctx.dev, dtype=ctx.dt),
                torch.as_tensor(grad_vm, device=ctx.dev, dtype=ctx.dt),
                None, None, None, None, None, None, None)   # +1 for merit_log


def close_batch(pg_flat, vm_flat, pool: ClosurePool, cases, V0s,
                tol: float, max_iter: int = 30, backtrack: bool = True,
                return_merit: bool = False, merit_log: bool = False):
    """Batched closure over the FLAT model outputs: ``pg_flat`` [sum n_gen],
    ``vm_flat`` [sum n_bus] -> (Vm_flat, Va_flat) [sum n_bus] each, forward
    solves fanned across ``pool``. Handles a per-scenario ragged control set.

    ``return_merit=True`` additionally returns the per-case merit
    ``0.5 * mean(F^2)``, differentiable w.r.t. the controls even where the Newton
    solve stalled (see ``_merit_grads``). Off by default so the many existing
    ``Vm, Va = close_batch(...)`` call sites are unaffected."""
    Vm, Va, merit = ACPFClosureBatch.apply(pg_flat, vm_flat, pool, cases, V0s,
                                           tol, max_iter, backtrack, merit_log)
    return (Vm, Va, merit) if return_merit else (Vm, Va)
