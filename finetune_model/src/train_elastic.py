"""Self-supervised ELASTIC fine-tune loop (loss.mode: elastic).

Reuses the control harness wholesale -- the same fixed splits, CaseStore, optimizer
/ scheduler builders, gradient clipping, eval_pred (Pg/V MAPE vs GT) and History.
What differs is the objective: the base model emits the two controls, a
differentiable single-slack Newton power flow (``closure.close_batch``, fanned
across CPU cores) closes the state, and the elastic AC-OPF loss
(``elastic_loss``) -- cost + equality/inequality feasibility on the CLOSED state
-- is minimised through the solve. The Newton tolerance follows a per-epoch
curriculum tol_start -> tol_end (loose -> tight).

Epoch 0 reports the ZERO-SHOT elastic loss: the base (un-fine-tuned) model's
controls closed at the tight tolerance, i.e. the feasible operating point the
released model already produces.
"""
from __future__ import annotations
import json, os, random, time
import numpy as np
import torch

from gridsfm.checkpoint import load_model
from gridsfm.schema import GEN_CP1_IDX, GEN_CP2_IDX, GEN_PMAX_IDX, GEN_AVAIL_IDX
from store import CaseStore, predict
from loss import control_loss, rmse_control_loss
from optim import build_optimizer, build_scheduler, warmup_lr, apply_freeze
from evaluate import eval_pred
from history import History
from plots import plot_elastic_training
from closure import (extract_case, close_batch, ClosurePool, dc_init_V,
                     flat_init_V)
from elastic_loss import elastic_loss, curriculum_tol, curriculum_weight
from train import set_seeds, set_runtime, preflight, _git_sha


def _cost_scale(store, keys) -> float:
    """Nominal-cost normaliser so rho~10 balances feasibility vs cost.
    nominal = mean over cases of sum(|cp1|*Pmax + cp2*Pmax^2) on online gens."""
    tot, n = 0.0, 0
    for k in keys:
        gx = store.get(k)[0]["generator"].x.double()
        on = gx[:, GEN_AVAIL_IDX] > 0.5
        pmax = gx[:, GEN_PMAX_IDX]
        nom = float(((gx[:, GEN_CP1_IDX].abs() * pmax
                      + gx[:, GEN_CP2_IDX] * pmax**2) * on).sum())
        tot += nom; n += 1
    return 1.0 / max(tot / max(n, 1), 1e-9)


def run_elastic(cfg: dict, man: dict) -> dict:
    dev = cfg["runtime"]["device"]
    T, L, CL = cfg["train"], cfg["loss"], cfg["closure"]
    D = cfg["_derived"]
    # Which GT-anchor shape closed_forward uses. Bound here with `wm` for the same
    # reason: closed_forward closes over it and the zero-shot eval runs first.
    CTRL_FORM = str(L.get("control_form", "maxaligned")).lower()
    if CTRL_FORM not in ("maxaligned", "squared"):
        raise ValueError(f"loss.control_form must be maxaligned|squared, got {CTRL_FORM!r}")
    # Bound here, not at the schedule block below: closed_forward closes over `wm` and
    # the zero-shot eval calls it before that block runs.
    wm = float(L.get("w_merit", 0.0))
    # CONVERGENCE-DRIVEN MERIT WEIGHT (closed loop). 0 disables -> wm stays constant.
    #
    # WHY A CONTROLLER AND NOT A CONSTANT. merit = 0.5*mean(F^2) self-anneals: its
    # gradient scales with F, so it has authority only while the closure is bad. Measured
    # on case6470_rte at a fixed w_merit=2e3, merit's share of the step direction went
    # 37% -> 2.4% -> 0.04% over three epochs as frac_converged climbed 71% -> 99%. The
    # anchor then owns the direction (98%) and, because it sits at cos = -0.89 to the
    # equality residual on this grid, walks the closure back out -- convergence decays and
    # merit is by then too small to arrest it. A CONSTANT weight cannot fix this: merit's
    # gradient per unit weight measured 173 in the failing regime and 0.741 at a trained
    # checkpoint, a 233x swing, so any fixed value is either inert or overwhelming. At
    # w_merit=5e5 merit held ~99% of the direction permanently, the closure never
    # converged, so merit never annealed -- a self-sustaining failure that took
    # frac_converged from 55% to 0% in five epochs.
    #
    # So: hold merit at a TARGET GRADIENT SHARE whenever the closure is unhealthy, and
    # let it anneal away once it is healthy. The weight is re-solved each epoch from the
    # MEASURED per-unit gradient, which makes it self-calibrating across regimes.
    # W_MERIT RAMP (loss.w_merit_start / _end / _epochs / _schedule). Reuses the same
    # curriculum_weight machinery as w_control. Set _end to ramp, leave it unset and
    # w_merit stays at the constant `w_merit`.
    #
    # WHY RAMP UP rather than hold or decay. merit only has authority while the closure
    # is bad, and it is least trustworthy exactly then -- its gradient comes from the
    # envelope theorem, which assumes a stalled, stationary Newton iterate, and at
    # ||F|| ~ 1e3 the returned point is nowhere near a solution. Starting small lets the
    # anchor and the inequality terms establish a sane operating point first, and raising
    # it slowly gives merit growing authority over the residual as the states it is
    # scoring become ones where its gradient actually means something.
    # ANCHOR ANNEALED BY GRADIENT SHARE (loss.w_control_share_start / _end / _epochs).
    # Schedules the anchor's SHARE OF THE GRADIENT NORM rather than its raw weight, and
    # re-solves w_control each epoch from the MEASURED per-term gradients to hit it.
    #
    # WHY SHARE AND NOT WEIGHT. The weight is not the quantity that matters and does not
    # map onto it stably: measured on this grid the anchor held 53.7% of the gradient at
    # w_control=85 but only 5-15% at w_control=1, and the anchor's gradient per unit
    # weight differs by 5.218x between the squared and max-aligned forms alone. A
    # geometric sweep of the WEIGHT therefore sweeps the share non-linearly and
    # unpredictably, scheduling the share directly is what the gated config's own sweep
    # was actually reasoning in ("93% -> PF broke", "75% <- chosen", "25% -> controls
    # lost"). Requires train.grad_probe_every >= 1.
    WC_SHARE_END = L.get("w_control_share_end", None)
    WC_SHARE_START = float(L.get("w_control_share_start", 0.75) or 0.75)
    WC_SHARE_EPOCHS = int(L.get("w_control_share_epochs", 100) or 100)
    WC_SHARE_SCHED = str(L.get("w_control_share_schedule", "geometric"))
    WC_SHARE_DAMP = float(L.get("w_control_damping", 0.35))
    WC_SHARE_STEP = float(L.get("w_control_max_step", 2.0))
    WM_RAMP_END = L.get("w_merit_end", None)
    WM_RAMP_START = float(L.get("w_merit_start", wm) or wm)
    WM_RAMP_EPOCHS = int(L.get("w_merit_epochs", 100) or 100)
    WM_RAMP_SCHED = str(L.get("w_merit_schedule", "geometric"))
    WM_BASE = wm
    WM_CONV_TARGET = float(L.get("w_merit_conv_target", 0.0) or 0.0)
    WM_BOOST_SHARE = float(L.get("w_merit_boost_share", 0.40))
    WM_MAX = float(L.get("w_merit_max", 5.0e4))
    # DAMPING. The controller inverts merit's measured gradient-per-unit-weight to hit
    # the target share, but that plant gain is wildly non-stationary: measured on
    # case6470_rte it moved 8.3e-5 -> 1.3e-6 -> 8.6e-5 over three consecutive epochs, a
    # 65x swing. A deadbeat solve against that oscillates -- observed 25% -> 5.6% ->
    # 61.6% achieved share against a 40% target, with w_merit swinging 2e3 -> 4.5e4 ->
    # 1.9e4. So move only part of the way, geometrically (a first-order lag in log w),
    # and rate-limit each step. Applies to the DOWN move too: snapping straight back to
    # the base weight is itself a ~9x jump.
    # PCGRAD (Yu et al. 2020, "Gradient Surgery for Multi-Task Learning"). Treats each
    # loss term as a task and, where two conflict, removes the conflicting component of
    # one before summing -- so neither has to be throttled for the other to make
    # progress. Motivated by measurement, not fashion: on case6470_rte
    # cos(anchor, eq_pen) = -0.892, and the per-epoch projections show the anchor at
    # +170% of the step direction while merit sits at -46% simultaneously. The whole
    # w_control / w_merit search has been an attempt to referee that conflict by
    # reweighting, PCGrad instead removes it geometrically.
    #
    # Requires the per-term gradients, i.e. train.grad_probe_every >= 1 -- which the
    # probe already computes, so the marginal cost is only the pairwise dot products.
    # loss.merit_log: use rho_r * log(1 + ||r||^2/(2 n_r)) instead of the raw
    # half-mean-square, i.e. log1p-compress merit the same way elastic_loss already
    # compresses the equality and inequality residuals. See closure.ACPFClosureBatch.
    MERIT_LOG = bool(L.get("merit_log", False))
    PCGRAD = bool(L.get("pcgrad", False))
    WM_DAMP = float(L.get("w_merit_damping", 0.35))      # 1.0 = deadbeat, 0 = frozen
    WM_STEP = float(L.get("w_merit_max_step", 2.0))      # max ratio change per epoch
    outdir = cfg["_meta"]["results_dir"]; ckdir = cfg["_meta"]["ckpt_dir"]
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckdir, exist_ok=True)
    ckpt_out = os.path.join(ckdir, "control_model.pt")

    set_seeds(T["seed"], T["deterministic"]); set_runtime(cfg); preflight(cfg)

    # ---- the fixed splits, exactly as the control harness reads them ----
    from subset import train_subset
    all_train = list(man["splits"]["train"]["keys"])
    train_modes = man["splits"]["train"].get("modes") or [None] * len(all_train)
    n_sub = int(cfg["data"].get("train_subset", 0))
    train_keys = train_subset(all_train, train_modes, n_sub,
                              int(cfg["data"].get("train_subset_seed", 0)))
    val_keys = list(man["splits"]["val"]["keys"])
    # The held-out split. Read but NEVER trained or selected on: it is scored once, at
    # the end, on the shipped weights. The sweep driver's scaling curve reads the
    # test_metrics.json written from it.
    test_keys = list(man["splits"].get("test", {}).get("keys", []))
    _solver = CL.get("solver", "superlu")
    print(f"run_label={cfg['run_label']}  ELASTIC  train={len(train_keys)} "
          f"val={len(val_keys)}  rho_g={L['rho_g']} rho_h={L['rho_h']}  "
          f"tol {CL['tol_start']:g}->{CL['tol_end']:g}  solver={_solver}  "
          f"workers={CL['workers'] if _solver != 'cudss' else 'serial(GPU)'}", flush=True)
    print(f"  anchor: control_form={CTRL_FORM}"
          + (f"  mean={L['mean_weight']} topk={L['topk_weight']} max={L['max_weight']}"
             f"  k_pg={D['topk_pg']} k_v={D['topk_v']}"
             f"  per_grid_topk={L.get('per_grid_topk', True)}"
             if CTRL_FORM == "maxaligned" else "  0.5*mean((d/tol)^2)"), flush=True)

    store = CaseStore(cfg["cache"]["dir"], cfg["_derived"]["lru_cases"], dev)
    # model.from_scratch: build a RANDOMLY INITIALISED backbone instead of loading the
    # released checkpoint. This is the no-pretraining ablation -- same architecture and
    # same objective as the fine-tuned arms, trained on one grid's data only, so the gap
    # to a fine-tuned run measures what the pretraining is worth and nothing else.
    #
    # model.hidden_dim / model.num_blocks size it. `load_model` builds the released
    # backbone from `metadata.arch`, which is null in gridsfm_open_v2.pt, so the released
    # model is the library default (hidden_dim 128, num_blocks 8, 15.15M params). A
    # from-scratch run on 1,000 scenarios of a single grid cannot support that capacity,
    # which is why this path takes its own sizes. Input widths come from the library's
    # DEFAULT_INPUT_DIMS, so nothing here depends on the dataset.
    if bool(cfg["model"].get("from_scratch", False)):
        from gridsfm.model import GridTransformerBackbone
        arch = dict(hidden_dim=int(cfg["model"].get("hidden_dim", 64)),
                    num_blocks=int(cfg["model"].get("num_blocks", 4)))
        model = GridTransformerBackbone(**arch).to(dev)
        _np = sum(q.numel() for q in model.parameters())
        print(f"  model FROM SCRATCH (no pretraining): hidden_dim={arch['hidden_dim']} "
              f"num_blocks={arch['num_blocks']}  {_np/1e6:.2f}M params", flush=True)
    else:
        model = load_model(cfg["base_checkpoint"], dev)
    model.clip_mode = cfg["model"]["clip_mode"]
    n_train_p, n_tot_p = apply_freeze(model, cfg["model"]["freeze"])
    opt = build_optimizer(model, T); sched = build_scheduler(opt, T)
    pool = ClosurePool(int(CL["workers"]), backend=CL.get("solver", "superlu"))
    # cost_scale: explicit config value wins, else the 1/nominal-cost auto-derivation.
    # The auto value normalises cost to O(1) PER CASE, but eq_pen/ineq_pen are
    # UNNORMALISED SUMS over the grid, so on a large grid the cost term is left three
    # orders of magnitude below the penalties -- measured on case6470_rte, cost carried
    # 0.0% of the gradient (8.49 against ineq_pen's 1.15e+04) and the objective was doing
    # no economic dispatch at all. An explicit value set by gradient balance fixes that,
    # the auto-derivation is kept for grids where it has not been measured.
    if L.get("cost_scale") is not None:
        cost_scale = float(L["cost_scale"])
        print(f"cost_scale={cost_scale:.6g} (from config)", flush=True)
    else:
        cost_scale = _cost_scale(store, train_keys[:min(50, len(train_keys))])
        print(f"cost_scale={cost_scale:.6g} (auto 1/nominal)", flush=True)

    # per-key Case (topology + this scenario's loads) and warm-start voltage
    case_cache: dict = {}
    warm: dict = {}
    # closure.warm_start: carry each case's converged point across epochs (the default,
    # and a large speed win), or COLD-START every batch from dc_init_V exactly as the
    # evaluation's completion does.
    #
    # WHY THE OPTION EXISTS. With the cache on, a case pays dc_init_V once and thereafter
    # resumes from its own previous AC solution, so `frac_converged` is a WARM-STARTED
    # statistic. The eval has no such cache -- completion.complete_case always calls
    # dc_init_V -- so the two measure different things on a grid where the cold start is
    # marginal. Measured on case6470_rte's val split: warm 100% converged, cold 4-51%.
    # That gap is why training reported no violations while the CPU eval reported plenty,
    # and why select_min_frac_converged could not discriminate between checkpoints.
    # train.grad_probe_every: every N epochs, measure each loss term's share of the
    # gradient (see grad_probe). 0 = off. Costs one extra closure solve + five backward
    # passes on ONE batch, so ~1 batch of overhead per probed epoch.
    GRAD_PROBE = int(T.get("grad_probe_every", 0) or 0)
    WARM_START = bool(CL.get("warm_start", True))
    # closure.cold_init: 'dc' (default) | 'flat'. Which bootstrap a case that has no
    # cached point gets. 'flat' is the textbook V=1 angle 0, on case6470_rte that start
    # converged 0/60 cases when it was measured, because the DC angles carry the solve.
    COLD_INIT = str(CL.get("cold_init", "dc")).lower()
    if COLD_INIT not in ("dc", "flat"):
        raise SystemExit(f"closure.cold_init must be dc|flat, got {COLD_INIT!r}")
    if not WARM_START:
        print(f"  closure warm_start=False: every batch cold-starts from "
              f"{'flat_init_V (V=1 angle 0)' if COLD_INIT == 'flat' else 'dc_init_V'} "
              f"(cold_init={COLD_INIT}; matches the eval's completion when dc; "
              f"slower, and frac_converged becomes the cold-start number)", flush=True)

    def cases_for(keys):
        cs = []
        for k in keys:
            c = case_cache.get(k)
            if c is None:
                c = extract_case(store.get(k)[0]); case_cache[k] = c
            cs.append(c)
        return cs

    def closed_forward(keys, tol, wc, train_mode, return_terms=False):
        """model -> controls -> closure -> elastic loss. ``wc`` is this step's
        GT-anchor weight (scheduled). Returns (loss, diag), or (loss, diag, terms) when
        ``return_terms`` -- each term still attached to THIS batch's graph so the caller
        can backpropagate them separately."""
        S = store.batch(keys)
        cases = cases_for(keys)
        boff = np.concatenate([[0], np.cumsum([c.n_bus for c in cases])])
        goff = np.concatenate([[0], np.cumsum([c.n_gen for c in cases])])
        Pg, V = predict(model, S)                       # [sum NG], [sum NB], with grad
        # Warm start: the cached converged point once a case has solved, otherwise a
        # cold-start bootstrap = the model's vm profile with DC power-flow angles (a
        # flat start does not converge AC Newton on large grids). One DC solve per
        # case, ever -- the cache carries it forward every later epoch.
        #
        # TRAINING ONLY. The cache used to be read on the validation path too, which
        # made val_frac_converged -- and therefore the SEL_MIN_FRAC selection gate that
        # exists to stop a non-converging checkpoint from shipping -- a WARM-STARTED
        # statistic: each val case was seeded from its own converged point from the
        # previous epoch, so the number measured "does this still solve from last
        # epoch's answer", not "does this solve". Measured on the shipped case6470_rte
        # set, val split, model controls from dc_init_V at tol 1e-4:
        #
        #     ckpt    reported val_frac_conv    true cold-start
        #     n0100          1.00                    98%
        #     n0200          0.98                    30%     <- shipped at ep18
        #     n0500          1.00                    61%
        #     n0025          0.96                    29%
        #     GT controls      --                   100%  (median 3 Newton iters)
        #
        # so the gate passed a checkpoint whose power flow solves on 30% of held-out
        # cases, and the DC cold start was never the limitation. Validation now always
        # bootstraps from dc_init_V, which is also what evaluate_model/completion.py
        # does, so the training-time number and the eval-time number finally agree.
        pg_np = Pg.detach().cpu().numpy(); vm_np = V.detach().cpu().numpy()
        V0s = []
        for k in range(len(keys)):
            cached = warm.get(keys[k]) if (WARM_START and train_mode) else None
            if cached is not None:
                V0s.append(cached)
            else:
                c = cases[k]
                # closure.cold_init selects the bootstrap. 'dc' (default) = the model's
                # voltage magnitudes with DC power-flow angles, 'flat' = the textbook
                # V = 1.0 angle 0. See flat_init_V for what flat costs on this grid.
                if COLD_INIT == "flat":
                    V0s.append(flat_init_V(c))
                else:
                    V0s.append(dc_init_V(c, pg_np[goff[k]:goff[k+1]],
                                         vm_np[boff[k]:boff[k+1]]))
        Vm, Va, merit = close_batch(Pg, V, pool, cases, V0s,
                                    tol=tol, max_iter=int(CL["max_newton"]),
                                    backtrack=bool(CL["backtrack"]),
                                    return_merit=True, merit_log=MERIT_LOG)
        loss, diag = elastic_loss(S["data"], Vm, Va, Pg,
                                  L["rho_g"], L["rho_h"], cost_scale=cost_scale)
        # GT-ANCHOR term (tol-normalised, same tol_pg/tol_v/w_pg/w_v as the control
        # loss). Always computed for logging/selection, weighted by the scheduled ``wc``
        # in the training loss -- heavy early to pull the controls into the PF-solvable
        # region, annealed to a balanced end weight.
        #
        # `loss.control_form` picks the shape. `maxaligned` is the v2 training loss and
        # the default: mean + top-k mean + 0.3*max on |d|/tol, so the anchor tracks the
        # WORST controls rather than the average. Under a plain mean (or its square) a
        # handful of badly-placed generators hide behind ~1900 good ones, and those are
        # exactly the ones the downstream corrector pays for. `squared` is the previous
        # 0.5*mean(e^2), kept so a run trained under it can be reproduced.
        if CTRL_FORM == "squared":
            ctrl, _, _ = rmse_control_loss(Pg, S["pg_gt"], V, S["vm_gt"],
                                           S["av"], S["vctrl"], L)
        else:
            ctrl, _, _ = control_loss(Pg, S["pg_gt"], V, S["vm_gt"],
                                      S["av"], S["vctrl"], L,
                                      D["topk_pg"], D["topk_v"],
                                      gen_seg=S["gen_seg"], bus_seg=S["bus_seg"],
                                      nseg=S["nseg"])
        if wc > 0:
            loss = loss + wc * ctrl
        # MERIT (feasibility-descent) term: 0.5 * mean(F^2) per case, SUMMED over the
        # batch. Its whole purpose is the STALLED cases -- the implicit-function adjoint
        # is invalid at a non-root, so those cases used to contribute exactly zero
        # gradient, which on case6470_rte was every case (60/60 stalled at the base
        # checkpoint, at every tolerance down to 1e-01). This gives each of them a valid
        # direction toward feasibility computed from its OWN physics rather than from
        # proximity to the GT label.
        #
        # It needs no schedule: as a case approaches feasibility F -> 0 and the term
        # vanishes on its own. That is why w_control can now be a constant.
        #
        # SUM OVER THE BATCH, not mean. cost/eq_pen/ineq_pen are already sums over the
        # flattened block-diagonal batch (elastic_loss: `log1p(absF).sum()`), so a batch
        # MEAN made merit the only intensive term in the objective and handed it a 1/B
        # handicap against every other term. Worse, the handicap fell on exactly the
        # cases merit exists for: merit is ~0 on a CONVERGED case by construction, so at
        # frac_converged=0.9 one stalled case in a batch of 4 was divided by 4 rather
        # than by the 1 case that actually had a gradient, while ineq_pen drew on all 3
        # converged ones through the closure adjoint. Measured on case6470_rte n0025 at
        # w_merit=1e3: merit held 0.00% of the step direction at frac_converged=1.0 and
        # 16.7% once every case stalled, i.e. it only spoke after the drift it was meant
        # to prevent. Summing makes each stalled case's contribution independent of how
        # many of its batch-mates happened to converge.
        #
        # NOTE this changes the SCALE of the logged `merit` column by ~B against every
        # run recorded before it -- history rows are only comparable within one
        # convention. It is the same convention cost/eq_pen/ineq_pen were always logged
        # under. The WITHIN-case mean over residual entries in closure.py is untouched,
        # that one was tried as a sum and reverted (see closure.py: the note at
        # `m_k = 0.5 * float(Fk @ Fk)`).
        merit_batch = merit.sum()
        if wm > 0:
            loss = loss + wm * merit_batch
        diag["merit"] = float(merit_batch.detach())
        diag["control_term"] = float(ctrl.detach())
        # WORST single control error in this batch, in PHYSICAL p.u. -- not
        # tol-normalised, so the number means the same thing whatever tol_pg/tol_v are
        # set to and can be read against a generator's own rating. The anchor's whole
        # premise is that the worst controls are what the downstream corrector pays
        # for, so the training figure has to show them: mean MAPE cannot distinguish a
        # model that halved every error from one that halved the median and doubled the
        # worst.
        with torch.no_grad():
            dP = (Pg[S["av"]] - S["pg_gt"][S["av"]]).abs()
            dV = (V[S["vctrl"]] - S["vm_gt"][S["vctrl"]]).abs()
            diag["max_pg_err"] = float(dP.max()) if dP.numel() else 0.0
            diag["max_v_err"] = float(dV.max()) if dV.numel() else 0.0
        diag["newton_iters"] = float(np.mean(pool.last_iters)) if pool.last_iters else 0.0
        # Fraction of the batch whose closure actually converged. Was never recorded --
        # only max_absF, a MAX over the batch, so one bad case masked everything and
        # there was no way to tell 5% failing from 100% failing.
        diag["frac_converged"] = (float(np.mean(pool.last_converged))
                                  if pool.last_converged else 1.0)
        # Carry the converged state across epochs for BOTH train and val, so every
        # case pays the DC bootstrap solve only once, drop non-converged cases so
        # they re-bootstrap (never warm-start from a bad basin).
        if WARM_START and train_mode:          # never seed val from val (see above)
            Vm_np = Vm.detach().cpu().numpy(); Va_np = Va.detach().cpu().numpy()
            conv = pool.last_converged or [True] * len(keys)
            for k in range(len(keys)):
                sl = slice(int(boff[k]), int(boff[k+1]))
                if conv[k]:
                    warm[keys[k]] = Vm_np[sl] * np.exp(1j * Va_np[sl])
                else:
                    warm.pop(keys[k], None)
        if return_terms:
            # Each term ISOLATED, by zeroing the other rho/cost_scale rather than by
            # re-deriving the algebra -- so what is measured is exactly what the
            # optimiser sees. elastic_loss does no closure work, it only scores the
            # (Vm, Va, Pg) already computed, so these three calls are cheap.
            from elastic_loss import elastic_loss as _el
            terms = {
                "cost":     _el(S["data"], Vm, Va, Pg, 0.0, 0.0, cost_scale=cost_scale)[0],
                "eq_pen":   _el(S["data"], Vm, Va, Pg, L["rho_g"], 0.0, cost_scale=0.0)[0],
                "ineq_pen": _el(S["data"], Vm, Va, Pg, 0.0, L["rho_h"], cost_scale=0.0)[0],
                "anchor":   wc * ctrl,
                "merit":    wm * merit_batch,
            }
            return loss, diag, terms
        return loss, diag

    @torch.no_grad()
    def eval_elastic(keys, tol, bs):
        model.eval()
        agg = {}
        nb = 0
        for i in range(0, len(keys), bs):
            _, d = closed_forward(keys[i:i+bs], tol, wc=0.0, train_mode=False)
            for kk, vv in d.items():
                agg[kk] = agg.get(kk, 0.0) + vv
            nb += 1
        return {kk: vv / max(nb, 1) for kk, vv in agg.items()}

    H = History(outdir)
    BATCH = max(1, min(int(T["batch"]), len(train_keys)))
    EB = int(cfg["eval"]["batch"])
    val_sel = val_keys[:min(48, len(val_keys))]      # subset used for per-epoch val loss

    @torch.no_grad()
    def gt_cost(keys, bs):
        """Cost of the SOLVER's dispatch on `keys` -- the optimum the model is chasing.

        Reported PER CASE, matching elastic_loss's per-case diagnostics. A batch sum
        would be comparable only against a trace taken at the same batch size, and it is
        not: the training trace runs at train.batch while this and the zero-shot numbers
        run at eval.batch. Online generators only -- the solver never charges c0 for a
        unit that is out.
        """
        from gridsfm.schema import (GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX,
                                    GEN_AVAIL_IDX)
        tot, nb = 0.0, 0
        for i in range(0, len(keys), bs):
            S = store.batch(keys[i:i+bs])
            gx = S["data"]["generator"].x.double()
            pg = S["pg_gt"].double()
            on = (gx[:, GEN_AVAIL_IDX] > 0.5).double()
            c = ((gx[:, GEN_CP2_IDX] * pg**2 + gx[:, GEN_CP1_IDX] * pg
                  + gx[:, GEN_CP0_IDX]) * on).sum()
            nb_cases = len(keys[i:i+bs])
            tot += float(c) / max(nb_cases, 1); nb += 1
        return tot / max(nb, 1)

    # ---- ZERO-SHOT: base model, closure at the tight tol (same subset as selection) ----
    zs = eval_elastic(val_sel, tol=float(CL["tol_end"]), bs=EB)
    zs["cost_opt"] = gt_cost(val_sel, EB)
    b_pg, b_v = eval_pred(model, store, val_keys, "val@ep0", bs=EB)
    # Carried into the dashboard so epoch 0 -- the state the run STARTS from -- is a
    # plotted point rather than only a reference line. Without it every curve begins at
    # epoch 1 and the first epoch's move, which is usually the largest of the run, is
    # invisible.
    zs["val_pg"], zs["val_v"] = b_pg, b_v
    print(f"[ZERO-SHOT ep0] elastic_loss={zs['loss']:.4f}  cost={zs['cost']:.3g}  "
          f"(optimal {zs['cost_opt']:.3g}, gap {100*(zs['cost']-zs['cost_opt'])/max(abs(zs['cost_opt']),1e-9):+.1f}%)  "
          f"eq_pen={zs['eq_pen']:.3f} ineq_pen={zs['ineq_pen']:.3f}  "
          f"maxViol(V/therm/Q)={zs['max_Vviol']:.3f}/{zs['max_thermal']:.3f}/"
          f"{zs['max_Qviol']:.3f}  ||  MAPE Pg={b_pg:.2f}% V={b_v:.4f}pu", flush=True)

    # selection + early stopping on the VAL elastic loss (the actual objective),
    # evaluated at the tight tolerance so it is comparable to the zero-shot point.
    # MAPE-vs-GT is NOT the selection metric here (the elastic objective legitimately
    # moves off the GT control point).
    PAT = int(T["early_stop_patience"]); MIN_D = float(T["early_stop_min_delta"])
    MIN_EPOCHS = int(T.get("min_epochs", 0))     # early stopping is inert below this
    # GT-anchor weight SCHEDULE: heavy early (pull controls into the PF-solvable
    # region), annealed to a balanced end weight so feasibility+cost shape the
    # final solution. Constant if only w_control is set (tx2k/case500 behaviour).
    wc_end = float(L.get("w_control_end", L.get("w_control", 0.0)))
    wc_start = float(L.get("w_control_start", wc_end))
    wc_epochs = int(L.get("w_control_epochs", CL.get("tol_epochs", 25)))
    wc_sched = L.get("w_control_schedule", "geometric")
    WC = wc_end                          # fixed balanced weight for val selection
    print(f"  w_control schedule: {wc_start:g} -> {wc_end:g} over {wc_epochs} epochs "
          f"({wc_sched}); selection weight = {WC:g}", flush=True)
    if WC_SHARE_END is not None:
        print(f"  w_control by GRADIENT SHARE: {100*WC_SHARE_START:g}% -> "
              f"{100*float(WC_SHARE_END):g}% over {WC_SHARE_EPOCHS} epochs "
              f"({WC_SHARE_SCHED}); w_control re-solved each epoch from measured grads "
              f"(damp {WC_SHARE_DAMP}, max step {WC_SHARE_STEP}x)", flush=True)
    if L.get("w_merit_end", None) is not None:
        print(f"  w_merit schedule: {WM_RAMP_START:g} -> {float(L['w_merit_end']):g} over "
              f"{WM_RAMP_EPOCHS} epochs ({WM_RAMP_SCHED})"
              + ("  [log1p form]" if MERIT_LOG else "  [raw half-mean-square]"), flush=True)
    else:
        print(f"  w_merit: {wm:g} (feasibility-descent term on stalled closures)"
              + ("  [log1p form]" if MERIT_LOG else ""), flush=True)
    # Minimum fraction of VALIDATION closures that must converge for an epoch to be
    # eligible for selection. 1.0 would be brittle (a single hard case blocks every
    # epoch), the default demands essentially all of them.
    SEL_MIN_FRAC = float(T.get("select_min_frac_converged", 0.99))
    best_vsel = float("inf"); stale = 0; best_ep = 0
    t_run = time.time()
    wc_share_next = None
    for ep in range(1, T["epochs"] + 1):
        t_ep = time.time()
        tol = curriculum_tol(ep, int(CL.get("tol_epochs", 25)), float(CL["tol_start"]),
                             float(CL["tol_end"]), CL.get("tol_schedule", "geometric"))
        wc = curriculum_weight(ep, wc_epochs, wc_start, wc_end, wc_sched)
        if WC_SHARE_END is not None and wc_share_next is not None:
            wc = wc_share_next          # the share controller owns w_control
        if WM_RAMP_END is not None:
            # The ramp OWNS w_merit, the convergence controller (w_merit_conv_target)
            # would fight it, so the two are mutually exclusive by construction --
            # whichever is configured takes effect, the ramp winning if both are.
            wm = curriculum_weight(ep, WM_RAMP_EPOCHS, WM_RAMP_START,
                                   float(WM_RAMP_END), WM_RAMP_SCHED)
            WM_BASE = wm
        wlr = warmup_lr(T, ep)
        if wlr is not None:
            for g in opt.param_groups:
                g["lr"] = wlr
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        random.shuffle(train_keys)
        model.train(); opt.zero_grad()
        acc = {}; acc_max = {}; nb = 0; gns = []; iters = []; n_clip = 0
        # PER-TERM GRADIENT ACCUMULATORS. Measured over EVERY batch of the epoch, not a
        # sample: one batch is 4 of 100 cases and train_keys is reshuffled each epoch, so
        # a single-batch probe swings wildly between epochs (observed: merit 99.2% ->
        # 0.00% -> 99.5% on consecutive epochs) and reports sampling noise as a trend.
        # Summing the gradient VECTORS across batches, then taking norms and projections
        # once at the end, is the same convention grad_decompose.py uses.
        probe_now = (bool(GRAD_PROBE) and (ep % GRAD_PROBE == 0)) or PCGRAD
        gnorms = None
        _flat = {}
        pcg_cos = []; pcg_conf = []
        gacc = {} if probe_now else None
        gparams = [q for q in model.parameters() if q.requires_grad] if probe_now else None
        for i in range(0, len(train_keys), BATCH):      # include the partial last batch
            keys = train_keys[i:i+BATCH]
            opt.zero_grad()
            if probe_now:
                loss, diag, terms = closed_forward(keys, tol, wc, train_mode=True,
                                                   return_terms=True)
                # Differentiate each term on the graph training ALREADY built, so the
                # probe costs backward passes only -- no second closure solve. Every
                # call retains the graph so the real loss.backward() below still works.
                if torch.isfinite(loss):
                    for _nm, _t in terms.items():
                        if not (torch.is_tensor(_t) and _t.requires_grad):
                            continue
                        _gs = torch.autograd.grad(_t, gparams, retain_graph=True,
                                                  allow_unused=True)
                        _f = torch.cat([(x if x is not None else torch.zeros_like(q)).reshape(-1)
                                        for x, q in zip(_gs, gparams)])
                        gacc[_nm] = _f if _nm not in gacc else gacc[_nm] + _f
                        _flat[_nm] = _f
                        del _gs
                del terms
            else:
                loss, diag = closed_forward(keys, tol, wc, train_mode=True)
            if PCGRAD and _flat and torch.isfinite(loss):
                # ---- gradient surgery -------------------------------------------
                # For each term, drop the component of it that opposes any other term,
                # then sum. Random pairwise order per the paper (the operation is not
                # order-invariant). The result is written straight into .grad, so
                # loss.backward() must NOT also run or the naive sum is double counted.
                _names = [k for k, v in _flat.items() if v is not None and float(v.dot(v)) > 0]
                _proj, _nconf = {}, 0
                for _i in _names:
                    _gi = _flat[_i].clone()
                    _order = _names[:]; random.shuffle(_order)
                    for _j in _order:
                        if _j == _i:
                            continue
                        _gj = _flat[_j]
                        _n2 = float(_gj.dot(_gj))
                        if _n2 <= 0.0:
                            continue
                        _d = float(_gi.dot(_gj))
                        if _d < 0.0:
                            _gi -= (_d / _n2) * _gj
                            _nconf += 1
                    _proj[_i] = _gi
                _tot = None
                for _g in _proj.values():
                    _tot = _g if _tot is None else _tot + _g
                # how much did surgery change the step? cosine of naive vs projected
                _naive = None
                for _g in _flat.values():
                    if _g is not None:
                        _naive = _g.clone() if _naive is None else _naive + _g
                _dn = float(_naive.norm()) * float(_tot.norm())
                pcg_cos.append(float(_naive.dot(_tot)) / _dn if _dn > 0 else 0.0)
                pcg_conf.append(_nconf)
                _off = 0
                for _p in gparams:
                    _n = _p.numel()
                    _p.grad = _tot[_off:_off + _n].view_as(_p).detach().clone()
                    _off += _n
                del _proj, _tot, _naive
            elif torch.isfinite(loss):
                loss.backward()
            _flat = {}
            # smart clipping, identical policy to the control harness
            if T["clip_grad"]:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), T["clip_grad"],
                                                    norm_type=float(T["clip_norm_type"]))
            else:
                gn = torch.nn.utils.get_total_norm(
                    [p.grad for p in model.parameters() if p.grad is not None],
                    norm_type=float(T["clip_norm_type"]))
            gnf = float(gn)
            if np.isfinite(gnf):
                opt.step(); gns.append(gnf)
                if T["clip_grad"] and gnf > T["clip_grad"]:
                    n_clip += 1
            for kk, vv in diag.items():
                if kk.startswith("max_"):
                    # These are maxima over a batch, the epoch value is the max over
                    # batches, not their mean. Summing then dividing would report a
                    # number no batch ever saw and would hide the worst one.
                    acc_max[kk] = max(acc_max.get(kk, float("-inf")), vv)
                else:
                    acc[kk] = acc.get(kk, 0.0) + vv
            iters.append(diag["newton_iters"]); nb += 1
        if sched is not None and warmup_lr(T, ep + 1) is None:
            sched.step()

        ep_mean = {kk: vv / max(nb, 1) for kk, vv in acc.items()}
        ep_mean.update(acc_max)          # maxima pass through unaveraged
        v_pg, v_v = eval_pred(model, store, val_keys, f"val@ep{ep}", bs=EB)   # full-val MAPE (plot)
        vd = eval_elastic(val_sel, tol=float(CL["tol_end"]), bs=EB)
        vel = vd["loss"]                                          # elastic-only (plotted)
        vsel = vel + WC * vd.get("control_term", 0.0)            # combined objective (selection)
        # CONVERGENCE GATE on selection. The elastic loss is computed on the CLOSED
        # state, and a case whose closure stalls is dropped -- so a model that fails to
        # close MORE cases is scored on fewer, easier ones and can post a LOWER val
        # loss while its power flow does not solve. Measured on case6470_rte: n0200 hit
        # its best val at ep61 with max_absF = 6.0 and shipped that, discarding an ep81
        # state at 5.5e-07, n0500 recorded three "best" epochs while sitting at
        # max_absF 9-27. Requiring the validation closure to actually converge before an
        # epoch can be selected makes that impossible.
        v_frac = float(vd.get("frac_converged", 1.0))
        converged_enough = v_frac >= SEL_MIN_FRAC
        improved = vsel < best_vsel - MIN_D and converged_enough
        is_best = vsel < best_vsel and converged_enough
        if is_best:
            best_vsel = vsel; best_ep = ep
            torch.save(model.state_dict(), os.path.join(ckdir, "best_val.pt"))
        elif vsel < best_vsel and not converged_enough:
            print(f"  [gate] ep{ep} val loss {vsel:.4f} beats best {best_vsel:.4f} but "
                  f"only {100*v_frac:.1f}% of val closures converged "
                  f"(< {100*SEL_MIN_FRAC:.0f}%); NOT selected", flush=True)
        torch.save(model.state_dict(), os.path.join(ckdir, "last.pt"))

        # PER-TERM GRADIENT SHARE, on the first batch of the shuffled epoch. Measured
        # rather than inferred: loss-value share and gradient share have repeatedly
        # disagreed here (w_merit once carried 22% of the value and 1.8% of the
        # gradient), so the dashboard plots the gradient.
        gp = {}
        if probe_now and gacc:
            try:
                tot_g = None
                for _g in gacc.values():
                    tot_g = _g.clone() if tot_g is None else tot_g + _g
                t2 = float(tot_g.dot(tot_g))
                gnorms = {_nm: float(_g.norm()) for _nm, _g in gacc.items()}
                nsum = sum(gnorms.values()) or 1.0
                for _nm, _g in gacc.items():
                    gp[f"gshare_{_nm}"] = 100.0 * gnorms[_nm] / nsum
                    gp[f"gproj_{_nm}"] = (100.0 * float(_g.dot(tot_g)) / t2) if t2 > 0 else 0.0
                del tot_g
            except Exception as e:                       # never let a probe kill a run
                print(f"  [grad_probe] ep{ep} failed: {type(e).__name__}: {e}", flush=True)
        gacc = None

        # --- ANCHOR WEIGHT FROM ITS TARGET GRADIENT SHARE ----------------------------
        # Solve for the w_control that puts the anchor at this epoch's scheduled share,
        # using the norms just measured, then damp and rate-limit exactly as the merit
        # controller does -- the same 65x plant-gain non-stationarity applies here, and
        # an undamped solve oscillated by 11x per epoch when it was tried on w_merit.
        if WC_SHARE_END is not None and gnorms is not None:
            tgt = curriculum_weight(ep, WC_SHARE_EPOCHS, WC_SHARE_START,
                                    float(WC_SHARE_END), WC_SHARE_SCHED)
            apu = gnorms.get("anchor", 0.0) / max(wc, 1e-30)      # per UNIT weight
            if apu > 0.0:
                others = sum(v for k, v in gnorms.items() if k != "anchor")
                sh = min(max(tgt, 1e-6), 0.99)
                want = (sh / (1.0 - sh)) * others / apu
                ratio = (want / max(wc, 1e-30)) ** WC_SHARE_DAMP
                ratio = min(max(ratio, 1.0 / WC_SHARE_STEP), WC_SHARE_STEP)
                wc_next = float(wc * ratio)
                got = 100.0 * gnorms.get("anchor", 0.0) / max(sum(gnorms.values()), 1e-30)
                if abs(wc_next - wc) / max(wc, 1e-30) > 0.01:
                    print(f"  [w_control] share {got:.1f}%->{100*tgt:.1f}%  "
                          f"w {wc:.4g} -> {wc_next:.4g} (solved {want:.4g})", flush=True)
                wc_share_next = wc_next

        # --- CONVERGENCE-DRIVEN MERIT WEIGHT -----------------------------------------
        # Re-solve wm for the NEXT epoch from this epoch's measured gradients. The
        # control signal is the gradient-NORM share, not the projection share: the norm
        # is positive and monotone in wm, so it inverts cleanly, whereas the projection
        # can go negative when terms fight and is not invertible.
        if WM_CONV_TARGET > 0.0 and gnorms is not None and WM_BASE > 0.0:
            fc = float(ep_mean.get("frac_converged", 1.0))
            mpu = gnorms.get("merit", 0.0) / max(wm, 1e-30)   # per UNIT weight
            if fc < WM_CONV_TARGET and mpu > 0.0:
                others = sum(v for k, v in gnorms.items() if k != "merit")
                sh = min(max(WM_BOOST_SHARE, 0.0), 0.95)
                want = (sh / (1.0 - sh)) * others / mpu
            else:
                want = WM_BASE                                # healthy: let it anneal
            want = float(min(max(want, WM_BASE), WM_MAX))
            # geometric damping, then rate limit -- both in ratio space, because w_merit
            # spans orders of magnitude and a linear step means nothing across that range
            ratio = (want / max(wm, 1e-30)) ** WM_DAMP
            ratio = min(max(ratio, 1.0 / WM_STEP), WM_STEP)
            wm_new = float(min(max(wm * ratio, WM_BASE), WM_MAX))
            if abs(wm_new - wm) / max(wm, 1e-30) > 0.01:
                got = 100.0 * gnorms.get("merit", 0.0) / max(sum(gnorms.values()), 1e-30)
                print(f"  [w_merit] conv={100*fc:.0f}% "
                      f"{'<' if fc < WM_CONV_TARGET else '>='} {100*WM_CONV_TARGET:.0f}%  "
                      f"share {got:.1f}%->{100*WM_BOOST_SHARE:.0f}%  "
                      f"w {wm:.4g} -> {wm_new:.4g} (solved {want:.4g}, damp {WM_DAMP})",
                      flush=True)
            wm = wm_new

        H.add_epoch(
            epoch=ep, elastic_loss=ep_mean["loss"], val_elastic_loss=vel,
            cost=ep_mean["cost"], eq_pen=ep_mean["eq_pen"], ineq_pen=ep_mean["ineq_pen"],
            control_term=ep_mean.get("control_term", 0.0), w_control=wc,
            merit=ep_mean.get("merit", 0.0), w_merit=wm,
            # Fraction of closures that actually converged. Without this the only
            # convergence signal was max_absF, a MAX over the batch -- one stalled case
            # pinned it and 5%-failing looked identical to 100%-failing.
            frac_converged=ep_mean.get("frac_converged", 1.0),
            pcgrad_cos=(float(np.mean(pcg_cos)) if pcg_cos else None),
            pcgrad_conflicts=(float(np.mean(pcg_conf)) if pcg_conf else None),
            **gp,
            val_frac_converged=v_frac,
            closure_tol=tol, newton_iters=float(np.mean(iters)) if iters else 0.0,
            max_pg_err=ep_mean.get("max_pg_err", float("nan")),
            max_v_err=ep_mean.get("max_v_err", float("nan")),
            max_absF=ep_mean["max_absF"], max_Vviol=ep_mean["max_Vviol"],
            max_thermal=ep_mean["max_thermal"], max_Qviol=ep_mean["max_Qviol"],
            val_pg=v_pg, val_v=v_v, val_score=vsel, lr=opt.param_groups[0]["lr"],
            gnorm_mean=float(np.mean(gns)) if gns else 0.0,
            gnorm_max=float(np.max(gns)) if gns else 0.0,
            gnorm_p99=float(np.percentile(gns, 99)) if gns else 0.0,
            clip_frac=n_clip / max(len(gns), 1),
            epoch_sec=time.time() - t_ep, is_best=int(is_best),
            gpu_peak_gb=(torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else 0.0),
        )
        if ep % max(1, int(cfg["logging"]["plot_every_epochs"])) == 0 or ep == T["epochs"]:
            plot_elastic_training(H.epoch, outdir, zero_shot=zs,
                                  sel_min_frac=SEL_MIN_FRAC,
                                  wm_conv_target=(WM_CONV_TARGET or None),
                                  wm_boost_share=WM_BOOST_SHARE)
        print(f"  ep{ep} tol={tol:.1e} L={ep_mean['loss']:.4f} valL={vel:.4f}"
              f"{' *' if is_best else '  '} cost={ep_mean['cost']:.3g} "
              f"ineq={ep_mean['ineq_pen']:.3f} iters={np.mean(iters):.1f} "
              f"gnorm={np.mean(gns) if gns else 0:.0f} MAPE Pg={v_pg:.2f}% "
              f"{time.time()-t_ep:.0f}s", flush=True)
        if PAT:
            stale = 0 if improved else stale + 1
            # MIN_EPOCHS floor. Early stopping asks "has val improved lately", which on a
            # fast-converging grid answers "no" long before the model has finished
            # improving: case500_goc stopped at 54-68 epochs with its best epoch at 34-48
            # and only 6.30% -> 3.3% Pg MAPE. Patience alone cannot express "do not even
            # ask yet", so the floor is explicit. Below it the counter still ADVANCES, so
            # once the floor is passed a genuinely plateaued run stops promptly rather
            # than serving a fresh patience window.
            if stale >= PAT and ep >= MIN_EPOCHS:
                print(f"  early stop: {PAT} epochs w/o val-loss improvement "
                      f"(best={best_vsel:.4f})", flush=True)
                break
            if stale >= PAT and ep < MIN_EPOCHS:
                print(f"  [min_epochs] would early-stop (stale={stale}) but ep{ep} < "
                      f"min_epochs={MIN_EPOCHS}; continuing", flush=True)

    plot_elastic_training(H.epoch, outdir, zero_shot=zs,
                                  sel_min_frac=SEL_MIN_FRAC,
                                  wm_conv_target=(WM_CONV_TARGET or None),
                                  wm_boost_share=WM_BOOST_SHARE)
    # ship the SELECTED epoch (best val elastic loss), like the control harness
    bp = os.path.join(ckdir, "best_val.pt")
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=dev))
    torch.save(model.state_dict(), ckpt_out)
    v_pg, v_v = eval_pred(model, store, val_keys, "val@ship", bs=EB)
    # HELD-OUT SCORE, on the shipped weights, once. Same metric definitions the control
    # harness and v2_ckpt_eval/eval_v2.py use -- aggregate Pg MAPE over AVAILABLE
    # generators, V MAE over control buses -- so an elastic point on the scaling curve
    # is comparable to a control point and to a zero-shot number.
    t_pg, t_v = (eval_pred(model, store, test_keys, "TEST", bs=EB)
                 if test_keys else (float("nan"), float("nan")))
    pool.shutdown()
    summary = dict(run_label=cfg["run_label"], grid_id=cfg["grid_id"],
                   train_subset=n_sub, n_train=len(train_keys),
                   zero_shot_loss=zs["loss"], final_loss=ep_mean["loss"],
                   zero_shot_mape_pg=b_pg, final_mape_pg=v_pg, final_mape_v=v_v,
                   epochs=T["epochs"], train_hours=(time.time()-t_run)/3600,
                   git_sha=_git_sha())
    summary.update(test_pg_pct=t_pg, test_v_pu=t_v, n_test=len(test_keys),
                   val_pg_pct=v_pg, val_v_pu=v_v, ship_epoch=best_ep,
                   baseline_val_pg_pct=b_pg, baseline_val_v_pu=b_v)
    with open(os.path.join(outdir, "elastic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # Written under the control harness's name as well: sweep.collect_result reads
    # results/<grid>/<label>/test_metrics.json to assemble the scaling curve, and
    # without it every elastic point is reported FAIL despite having trained fine.
    with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== DONE {cfg['run_label']} in {summary['train_hours']*3600:.0f}s | "
          f"zero-shot L={zs['loss']:.4f} -> final L={ep_mean['loss']:.4f} | "
          f"MAPE Pg {b_pg:.2f}%->{v_pg:.2f}% ===", flush=True)
    return summary
