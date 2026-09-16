"""The MAX-ALIGNED control loss.

    L_q  = mean_weight*mean(e) + topk_weight*mean(topk(e)) + max_weight*max(e)
    tot  = w_pg * L_pg + w_v * L_v,            e = |pred - gt| / tol

Why this shape, and why the pieces are not interchangeable:
  * tol-normalised ABSOLUTE error, not range-normalised: range-norm degenerates on
    fixed generators (Pmax == Pmin) and silently stops scoring them.
  * the top-k and max terms are what make the loss track the WORST control error,
    which is what drives the downstream corrector's cost. A plain mean lets a few
    bad generators hide behind 1900 good ones.
  * because of that, top-k must be taken WITHIN a case. Pooled across a batch it
    selects the worst k in the batch, and cases whose worst error is merely
    average contribute nothing to the max-aligned part. `segmented` handles this
    so batch size stays a free knob; at batch=1 the two paths are identical.
"""
from __future__ import annotations
import torch


def _terms(e: torch.Tensor, k: int, wm: float, wt: float, wx: float) -> torch.Tensor:
    """wm*mean + wt*mean(topk) + wx*max. Zero weights drop their term entirely."""
    if e.numel() == 0:
        return e.new_zeros(())
    out = e.new_zeros(())
    if wm:
        out = out + wm * e.mean()
    if wt:
        out = out + wt * e.topk(min(k, e.numel())).values.mean()
    if wx:
        out = out + wx * e.amax()
    return out


def _segmented(e: torch.Tensor, seg: torch.Tensor, nseg: int, k: int,
               wm: float, wt: float, wx: float, frac: float = 0.0) -> torch.Tensor:
    """Same as `_terms` but computed per case and averaged, so a batch behaves like
    the mean of its single-case losses. `seg` maps each element to its case index.

    `k` IS PER SEGMENT WHEN `frac` IS GIVEN. One absolute k is a different quantile on
    every case it meets: the worst 9 of ~100 scored generators is a broad mean, the
    worst 9 of ~2000 is a near-max. Sizing k from each segment's OWN scored count keeps
    the term the same quantile everywhere.

    This is not only a multi-grid concern. Under composed perturbations `killgen` takes
    a different number of units offline in every scenario, so the scored count varies
    case to case inside a single-grid batch too, and a fixed k drifts in quantile across
    them. `frac=0` keeps the old fixed-k behaviour.
    """
    if e.numel() == 0:
        return e.new_zeros(())
    tot = e.new_zeros(())
    n = 0
    for s in range(nseg):
        m = (seg == s)
        if m.any():
            ks = max(1, round(frac * int(m.sum()))) if frac else k
            tot = tot + _terms(e[m], ks, wm, wt, wx)
            n += 1
    return tot / max(n, 1)


def control_loss(pg_pred, pg_gt, v_pred, v_gt, av, vctrl, cfg,
                 topk_pg: int, topk_v: int,
                 gen_seg=None, bus_seg=None, nseg: int = 1):
    """Returns (total, L_pg, L_v). `av`/`vctrl` are the scored-element masks."""
    eP = ((pg_pred[av] - pg_gt[av]) / cfg["tol_pg"]).abs()
    eV = ((v_pred[vctrl] - v_gt[vctrl]) / cfg["tol_v"]).abs()
    wm, wt, wx = cfg["mean_weight"], cfg["topk_weight"], cfg["max_weight"]
    if cfg["segmented_topk"] and nseg > 1:
        # Per-segment k when the config gives a fraction: see _segmented.
        fp = cfg.get("topk_pg_frac", 0.0) if cfg.get("per_grid_topk", True) else 0.0
        fv = cfg.get("topk_v_frac", 0.0) if cfg.get("per_grid_topk", True) else 0.0
        L_pg = _segmented(eP, gen_seg[av], nseg, topk_pg, wm, wt, wx, fp)
        L_v = _segmented(eV, bus_seg[vctrl], nseg, topk_v, wm, wt, wx, fv)
    else:
        L_pg = _terms(eP, topk_pg, wm, wt, wx)
        L_v = _terms(eV, topk_v, wm, wt, wx)
    return cfg["w_pg"] * L_pg + cfg["w_v"] * L_v, L_pg, L_v


def rmse_control_loss(pg_pred, pg_gt, v_pred, v_gt, av, vctrl, cfg):
    """Squared-error anchor against the GT controls.

        L = w_pg * 0.5*mean((dPg/tol_pg)^2) + w_v * 0.5*mean((dV/tol_v)^2)

    This is EXACTLY the GT-anchor term the elastic objective already carries
    (`train_elastic.closed_forward`), factored out here so the ablation optimises the
    identical functional form with the physics -- cost, the equality/inequality
    penalties on the PF-closed state, and the merit term -- removed. The two arms then
    differ in one thing and one thing only, which is what makes the comparison an
    ablation rather than two different recipes.

    Note it is deliberately NOT the max-aligned control loss (`control_loss` above):
    that one is a third objective, and mixing it in would confound "elastic vs plain
    regression" with "max-aligned vs mean-aligned".

    Returns (total, L_pg, L_v), same signature shape as `control_loss`.

    SQUARED form, the exact analogue of the merit term:

        L_pg = ||dPg||^2 / (2 * tol_pg^2 * n_G)  =  0.5 * mean((dPg/tol_pg)^2)
        L_v  = ||dV ||^2 / (2 * tol_v^2  * n_PV) =  0.5 * mean((dV /tol_v )^2)

    The merit (power-flow restoration) term is `0.5 * mean(F^2)`; these are the same
    half-mean-square taken over the TOLERANCE-NORMALISED control error, so the anchor
    and the merit term sit on one footing inside the elastic objective.

    The tolerance enters SQUARED because it is the normalised error `dPg/tol_pg` that
    gets squared -- this term is literally the square of the previous
    `||d||/(tol*sqrt(n))`. The anchor therefore grows QUADRATICALLY once errors exceed
    the tolerance where before it grew linearly, so a `w_control` derived under the old
    form does not carry over and is re-derived per grid.
    """
    eP = (pg_pred[av] - pg_gt[av]) / cfg["tol_pg"]
    eV = (v_pred[vctrl] - v_gt[vctrl]) / cfg["tol_v"]
    L_pg = (0.5 * (eP ** 2).mean() if eP.numel() else pg_pred.new_zeros(()))
    L_v = (0.5 * (eV ** 2).mean() if eV.numel() else v_pred.new_zeros(()))
    return cfg["w_pg"] * L_pg + cfg["w_v"] * L_v, L_pg, L_v
