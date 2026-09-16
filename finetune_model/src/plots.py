"""Training / evaluation figures.

Written against a validated categorical palette (slots 1-3 clear the all-pairs CVD
and normal-vision floors on the light surface). The aqua slot sits at 2.74:1
contrast, below 3:1, so the relief rule applies: every series is DIRECT-LABELLED
and history_epoch.csv ships as the table view. Text wears ink tokens, never the
series colour.

One measure per axis -- never a dual-axis chart. Quantities on different scales
(Pg% vs V p.u., LR vs grad-norm) each get their own panel.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8880"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GOOD, CRIT = "#0ca30c", "#d03b3b"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=INK_MUTED, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    return ax


def _label_end(ax, x, y, text, color):
    if len(x) == 0:
        return
    ax.annotate(f" {text}", (x[-1], y[-1]), color=color, fontsize=8,
                va="center", ha="left", weight="bold", annotation_clip=False)


def _clean(xs, ys):
    """Drop None/NaN pairs so a partial run still plots."""
    out = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None
           and not (isinstance(b, float) and np.isnan(b))]
    return ([a for a, _ in out], [b for _, b in out])


def plot_training(hist: dict, outdir: str, step_csv: str | None = None) -> str:
    ep = hist["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), facecolor=SURFACE)

    ax = _style(axes[0][0], "Train loss per epoch (tol-normalised)", "epoch", "loss")
    for col, c, lab in (("loss", S1, "total"), ("loss_pg", S2, "Pg"), ("loss_v", S3, "V")):
        x, y = _clean(ep, hist[col])
        ax.plot(x, y, color=c, lw=1.8); _label_end(ax, x, y, lab, c)
    ax.set_yscale("log"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = _style(axes[0][1], "Train loss per optimizer step", "step", "loss")
    if step_csv and os.path.exists(step_csv):
        d = np.genfromtxt(step_csv, delimiter=",", names=True)
        if d.size > 1:
            s, l = d["step"], d["loss"]
            ax.plot(s, l, color=INK_MUTED, lw=0.5, alpha=0.55)
            w = max(1, len(l) // 200)                      # rolling mean over ~200 pts
            if w > 1:
                k = np.ones(w) / w
                ax.plot(s[w - 1:], np.convolve(l, k, mode="valid"), color=S1, lw=1.6)
            _label_end(ax, list(s), list(l), "per-step", S1)
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no step history", color=INK_MUTED, ha="center", transform=ax.transAxes)

    ax = _style(axes[0][2], "Validation control error", "epoch", "see labels")
    x, y = _clean(ep, hist["val_pg"])
    ax.plot(x, y, color=S1, lw=1.8); _label_end(ax, x, y, "Pg %", S1)
    x2, y2 = _clean(ep, hist["val_v"])
    if y2:                                                 # V in p.u. -> scale to share the axis
        sc = (max(y) / max(y2)) if max(y2) > 0 and y else 1.0
        ax.plot(x2, [v * sc for v in y2], color=S3, lw=1.8, ls="--")
        _label_end(ax, x2, [v * sc for v in y2], f"V p.u. (x{sc:.0f})", S3)
    best = [i for i, b in enumerate(hist["is_best"]) if b]
    if best:                                   # mark the epoch whose weights ship
        bi = best[-1]
        ax.axvline(ep[bi], color=GOOD, lw=1.0, ls=":")
        # anchored to the axis floor, not to the series: at max(y) it collides with the
        # direct end-labels, which is the whole relief mechanism for this palette
        ax.annotate(f" best ep{ep[bi]}", xy=(ep[bi], 0), xycoords=("data", "axes fraction"),
                    color=GOOD, fontsize=8, va="bottom")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = _style(axes[1][0], "Learning rate (cosine)", "epoch", "lr")
    x, y = _clean(ep, hist["lr"]); ax.plot(x, y, color=S1, lw=1.8)
    _label_end(ax, x, y, "lr", S1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = _style(axes[1][1], "Gradient norm, pre-clip", "epoch", "norm")
    for col, c, lab in (("gnorm_mean", S1, "mean"), ("gnorm_p99", S2, "p99"),
                        ("gnorm_max", INK_MUTED, "max")):
        x, y = _clean(ep, hist[col])
        ax.plot(x, y, color=c, lw=1.6 if c != INK_MUTED else 1.0)
        _label_end(ax, x, y, lab, c if c != INK_MUTED else INK_2)
    ax.axhline(1.0, color=CRIT, lw=1.0, ls=":")
    ax.annotate(" clip=1.0", (ep[0] if ep else 0, 1.0), color=CRIT, fontsize=8, va="bottom")
    ax.set_yscale("log"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = _style(axes[1][2], "Epoch wall time by phase", "epoch", "seconds")
    x = [e for e in ep if e is not None]
    dat = np.array([hist["data_sec"][i] or 0 for i in range(len(x))])
    fb = np.array([hist["fwd_bwd_sec"][i] or 0 for i in range(len(x))])
    ev = np.array([hist["eval_sec"][i] or 0 for i in range(len(x))])
    ax.bar(x, dat, color=SEQ[1], label="data")
    ax.bar(x, fb, bottom=dat, color=SEQ[4], label="fwd+bwd")
    ax.bar(x, ev, bottom=dat + fb, color=SEQ[6], label="eval")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    p = os.path.join(outdir, "training_curves.png")
    fig.savefig(p, dpi=140, facecolor=SURFACE); plt.close(fig)
    return p


def plot_test_diagnostics(d: dict, outdir: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), facecolor=SURFACE)
    err = np.asarray(d["pg_abs_err"]); ve = np.asarray(d["v_abs_err"])
    gt = np.asarray(d["pg_gt"]); pr = np.asarray(d["pg_pred"])

    ax = _style(axes[0][0], "Per-generator |Pg error|", "p.u.", "count")
    ax.hist(err, bins=80, color=S1, alpha=0.9)
    ax.set_yscale("log")
    for q, lab in ((d["pg_err_p90"], "p90"), (d["pg_err_p99"], "p99")):
        ax.axvline(q, color=S2, lw=1.0, ls="--")
        ax.annotate(f" {lab} {q:.3f}", (q, ax.get_ylim()[1] * 0.5), color=S2, fontsize=8)

    ax = _style(axes[0][1], "Per-bus |V error| at PV/slack buses", "p.u.", "count")
    if ve.size:
        ax.hist(ve, bins=80, color=S3, alpha=0.9); ax.set_yscale("log")
        ax.axvline(d["v_err_p99"], color=S2, lw=1.0, ls="--")
        ax.annotate(f" p99 {d['v_err_p99']:.4f}", (d["v_err_p99"], ax.get_ylim()[1] * 0.5),
                    color=S2, fontsize=8)

    ax = _style(axes[0][2], "Predicted vs GT Pg", "GT Pg (p.u.)", "predicted Pg (p.u.)")
    ax.hexbin(gt, pr, gridsize=55, cmap="Blues", mincnt=1, linewidths=0)
    lim = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
    ax.plot(lim, lim, color=INK_MUTED, lw=1.0, ls="--")
    ax.annotate(" y=x", (lim[1], lim[1]), color=INK_2, fontsize=8, va="top", ha="right")

    ax = _style(axes[1][0], "|Pg error| vs generator output", "GT Pg (p.u.)", "|err| (p.u.)")
    ax.scatter(gt, err, s=3, color=S1, alpha=0.18, edgecolors="none")
    if gt.size > 50:                                      # binned median: is error output-dependent?
        qs = np.quantile(gt, np.linspace(0, 1, 13))
        cx, cy = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (gt >= a) & (gt < b)
            if m.sum() > 5:
                cx.append(0.5 * (a + b)); cy.append(float(np.median(err[m])))
        ax.plot(cx, cy, color=S2, lw=1.8); _label_end(ax, cx, cy, "median", S2)
    ax.set_yscale("log")

    ax = _style(axes[1][1], "Worst-generator |Pg error| per case (sorted)",
                "case rank", "p.u.")
    w = np.sort(np.asarray(d["per_case_worst_pg"]))[::-1]
    ax.plot(np.arange(1, w.size + 1), w, color=S2, lw=1.6)
    _label_end(ax, list(range(1, w.size + 1)), list(w), "worst gen", S2)

    ax = _style(axes[1][2], "Per-case Pg error (sorted)", "case rank", "% of case |Pg|")
    pc = np.sort(np.asarray(d["per_case_pg_pct"]))[::-1]
    ax.plot(np.arange(1, pc.size + 1), pc, color=S1, lw=1.6)
    _label_end(ax, list(range(1, pc.size + 1)), list(pc), "case Pg %", S1)
    ax.axhline(float(np.mean(pc)), color=INK_MUTED, lw=1.0, ls=":")
    ax.annotate(f" mean {np.mean(pc):.2f}%", (pc.size, float(np.mean(pc))),
                color=INK_2, fontsize=8, va="bottom", ha="right")

    fig.suptitle(f"TEST diagnostics  |  {d.get('n_test', 0)} cases, "
                 f"Pg {d.get('test_pg_pct', 0):.2f}%  V {d.get('test_v_pu', 0):.4f} p.u.",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, "test_diagnostics.png")
    fig.savefig(p, dpi=140, facecolor=SURFACE); plt.close(fig)
    return p


def plot_elastic_training(hist: dict, outdir: str, zero_shot: dict | None = None,
                          sel_min_frac: float | None = None,
                          wm_conv_target: float | None = None,
                          wm_boost_share: float | None = None) -> str:
    """Live 12-panel dashboard for the elastic fine-tune.

    Row1: elastic loss | worst control err | learning rate | GRADIENT SHARE | WEIGHTS
    Row2: Pg & V MAPE  | time per epoch    | Newton iters  | ACHIEVED vs TOL | MERIT CTRL
    Row3: generation cost | loss components | feasibility   | CLOSURE CONV   | GT-ANCHOR LOSS

    The FIFTH column is the schedules, the merit controller's own loop, and the anchor:
    what the weights were set to, whether the controller hit the share it was aiming for,
    and the max-aligned GT-anchor loss both raw and weighted. w_merit is no longer a constant
    when loss.w_merit_conv_target > 0 -- it is re-solved each epoch from the measured
    gradients -- so a plot of the objective that omits it is not the objective.

    The fourth column answers three questions the first nine panels could not:
    WHAT IS STEERING (per-term share of the realised step direction), HOW WELL THE
    NEWTON SOLVER ACTUALLY CLOSED (max|F| against the tolerance it was asked for), and
    HOW MANY CASES CLOSED AT ALL. The last two matter because a run can report a
    converged closure while sitting an order of magnitude above the tolerance the
    downstream evaluation demands -- on case6470_rte every epoch of the n=500 run had
    max|F| between 6.6e-6 and 9.9e-6, inside the training tol of 1e-5 and outside the
    eval's 1e-6.
    """
    # EPOCH 0 = the zero-shot state, prepended so every series starts where the run
    # actually started. The first epoch's move is normally the largest of the whole run
    # (the closure converging), and beginning the axis at epoch 1 hides it. Only keys
    # the zero-shot eval actually measured are prepended; the rest (lr, w_control,
    # epoch_sec) have no epoch-0 value and keep their own, shorter x.
    if zero_shot is not None and hist.get("epoch"):
        _Z = {"elastic_loss": "loss", "val_elastic_loss": "loss", "cost": "cost",
              "val_pg": "val_pg", "val_v": "val_v", "newton_iters": "newton_iters",
              "control_term": "control_term"}
        hist = dict(hist)
        hist["epoch"] = [0] + list(hist["epoch"])
        for k, zk in _Z.items():
            if k in hist and zero_shot.get(zk) is not None:
                hist[k] = [zero_shot[zk]] + list(hist[k])
            elif k in hist:
                hist[k] = [None] + list(hist[k])
        for k, v in hist.items():
            if k not in ("epoch",) and isinstance(v, list) and len(v) == len(hist["epoch"]) - 1:
                hist[k] = [None] + list(v)          # no epoch-0 value: leave a gap
    ep = hist["epoch"]
    fig, ax = plt.subplots(3, 5, figsize=(27.5, 12.6), facecolor=SURFACE)

    # -- (0,0) elastic loss (train + val, vs zero-shot) --
    a = _style(ax[0][0], "Elastic loss per epoch", "epoch", "loss")
    x, y = _clean(ep, hist["elastic_loss"]); a.plot(x, y, color=S1, lw=1.9); _label_end(a, x, y, "train", S1)
    xv, yv = _clean(ep, hist.get("val_elastic_loss", []))
    if xv:
        a.plot(xv, yv, color=S2, lw=1.9); _label_end(a, xv, yv, "val", S2)
    if zero_shot is not None and (x or xv):
        a.axhline(zero_shot["loss"], color=INK_MUTED, lw=1.0, ls="--")
        a.annotate(" zero-shot", ((x or xv)[0], zero_shot["loss"]), color=INK_MUTED, fontsize=8, va="bottom")
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (0,1) WORST control error, Pg and V --
    # Replaced the w_control schedule, which was a constant line on every run that does
    # not anneal it and is in the resolved config anyway. The anchor is built to track
    # the WORST controls -- that is what the top-k and max terms are for -- and mean
    # MAPE cannot show whether they are working: a model that halves the median while
    # doubling its worst generator scores better on MAPE and is worse where it matters.
    # Physical p.u. on both axes, twinned because Pg and V differ by ~two decades.
    a = _style(ax[0][1], "Worst control error (max over epoch)", "epoch", "max |dPg| p.u.")
    x, y = _clean(ep, hist.get("max_pg_err", []))
    if x:
        a.plot(x, y, color=S1, lw=1.9); _label_end(a, x, y, "Pg", S1)
    a2 = a.twinx(); a2.set_ylabel("max |dV| p.u.", color=INK_2, fontsize=9)
    a2.tick_params(colors=INK_2, labelsize=8)
    xv, yv = _clean(ep, hist.get("max_v_err", []))
    if xv:
        a2.plot(xv, yv, color=S2, lw=1.6, ls="--"); _label_end(a2, xv, yv, "V", S2)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (0,2) learning rate --
    a = _style(ax[0][2], "Learning rate", "epoch", "lr")
    x, y = _clean(ep, hist["lr"]); a.plot(x, y, color=S3, lw=1.9); _label_end(a, x, y, "lr", S3)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (1,0) Pg / V MAPE (twin axis: distinct scales) --
    a = _style(ax[1][0], "Val MAPE: Pg (%) and V (p.u.)", "epoch", "Pg %")
    x, y = _clean(ep, hist["val_pg"]); a.plot(x, y, color=S1, lw=1.9); _label_end(a, x, y, "Pg%", S1)
    a2 = a.twinx(); a2.set_ylabel("V p.u.", color=INK_2, fontsize=9); a2.tick_params(colors=INK_2, labelsize=8)
    for s in ("top",): a2.spines[s].set_visible(False)
    xv, yv = _clean(ep, hist["val_v"]); a2.plot(xv, yv, color=S2, lw=1.9)
    _label_end(a2, xv, yv, "V", S2); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (1,1) time per epoch --
    a = _style(ax[1][1], "Time per epoch", "epoch", "seconds")
    x, y = _clean(ep, hist["epoch_sec"]); a.plot(x, y, color=S3, lw=1.9); _label_end(a, x, y, "s", S3)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (1,2) avg Newton iters --
    a = _style(ax[1][2], "Avg Newton iterations (closure)", "epoch", "iters")
    x, y = _clean(ep, hist["newton_iters"]); a.plot(x, y, color=S2, lw=1.9, marker="o", ms=3)
    _label_end(a, x, y, "it", S2); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (2,0) generation cost, against zero-shot and the solver optimum --
    # Cost, not gradient norm: gradient norm says how hard the optimiser is pushing,
    # which is only interesting when something is broken. Cost says whether the run is
    # buying anything, and it is only readable against two references -- where the base
    # checkpoint started, and what the solver achieved on the same cases. The gap
    # between the trace and `optimal` is the actual quantity of interest.
    a = _style(ax[2][0], "Generation cost vs zero-shot and optimum", "epoch", "cost")
    x, y = _clean(ep, hist.get("cost", []))
    if x:
        a.plot(x, y, color=S1, lw=1.9); _label_end(a, x, y, "train", S1)
    if zero_shot is not None and x:
        zc = zero_shot.get("cost")
        if zc is not None:
            a.axhline(zc, color=INK_MUTED, lw=1.0, ls="--")
            a.annotate(" zero-shot", (x[0], zc), color=INK_MUTED, fontsize=8, va="bottom")
        opt = zero_shot.get("cost_opt")
        if opt is not None:
            a.axhline(opt, color=GOOD, lw=1.2, ls="-.")
            a.annotate(" optimal (solver)", (x[0], opt), color=GOOD, fontsize=8, va="top")
            if y:
                gap = 100.0 * (y[-1] - opt) / max(abs(opt), 1e-9)
                a.annotate(f"gap {gap:+.1f}%", (x[-1], y[-1]), color=S1, fontsize=8,
                           va="bottom", ha="right")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (2,1) loss components (the GT anchor is excluded; noted in the title) --
    a = _style(ax[2][1], "Loss components", "epoch", "value")
    # `cost` in the history is cost_RAW, not cost*cost_scale -- elastic_loss records
    # cost_raw in its diagnostics while the loss uses the scaled value. Labelling it
    # "cost*sc" invited reading the panel as if the three series were the loss terms in
    # proportion, which they are not: at cost_scale 3.82e-4 the scaled cost is ~4e3,
    # not the 1e7 plotted here.
    comps = [("cost", S1, "cost (raw)"), ("eq_pen", S2, "eq |F|"), ("ineq_pen", S3, "ineq h~")]
    for col, c, lab in comps:
        x, y = _clean(ep, hist.get(col, [])); a.plot(x, y, color=c, lw=1.7); _label_end(a, x, y, lab, c)
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # -- (2,2) closed-state feasibility (max violations) --
    a = _style(ax[2][2], "Closed-state max violations", "epoch", "p.u. / ratio")
    for col, c, lab in (("max_absF", INK_2, "|F|"), ("max_Vviol", S1, "V"),
                        ("max_thermal", S2, "therm"), ("max_Qviol", S3, "Q")):
        x, y = _clean(ep, hist[col]); a.plot(x, y, color=c, lw=1.6); _label_end(a, x, y, lab, c)
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- COLUMN 4 -----------------------------------------------------------------
    # (0,3) per-term share OF THE STEP DIRECTION. gproj_X = (g_X . g_tot)/||g_tot||^2,
    # so the series sum to 100% and a term pulling AGAINST the consensus plots negative
    # -- which is the whole point on a grid where the GT anchor sits at cos = -0.89 to
    # the equality residual. Populated only when train.grad_probe_every > 0.
    a = _style(ax[0][3], "Gradient share of step direction", "epoch", "% of direction")
    _TERMS = (("anchor", S1), ("merit", S2), ("eq_pen", S3), ("ineq_pen", INK_2),
              ("cost", INK_MUTED))
    any_g = False
    for name, c in _TERMS:
        x, y = _clean(ep, hist.get(f"gproj_{name}", []))
        if not x:
            continue
        any_g = True
        a.plot(x, y, color=c, lw=1.7); _label_end(a, x, y, name, c)
    if any_g:
        a.axhline(0.0, color=INK_MUTED, lw=0.9, ls=":")
    else:
        a.text(0.5, 0.5, "set train.grad_probe_every > 0\nto record gradient shares",
               transform=a.transAxes, ha="center", va="center",
               color=INK_MUTED, fontsize=9)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (1,3) what the Newton solver ACHIEVED against what it was asked for. Both on one
    # log axis: the gap between them is the quantity that made training report a
    # converged closure while the evaluation reported violations.
    a = _style(ax[1][3], "Newton residual achieved vs target tol", "epoch", "max |F|  (p.u.)")
    x, y = _clean(ep, hist.get("max_absF", []))
    if x:
        a.plot(x, y, color=INK_2, lw=1.8); _label_end(a, x, y, "achieved", INK_2)
    xt, yt = _clean(ep, hist.get("closure_tol", []))
    if xt:
        a.plot(xt, yt, color=S2, lw=1.5, ls="--"); _label_end(a, xt, yt, "target tol", S2)
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (2,3) how many closures converged, train and val, against the selection gate.
    # With closure.warm_start off this is the COLD-START number -- the same statistic
    # the eval's completion measures -- so it is directly comparable to it.
    a = _style(ax[2][3], "Closure convergence", "epoch", "fraction of cases")
    for col, c, l2 in (("frac_converged", S1, "train"),
                       ("val_frac_converged", S2, "val")):
        x, y = _clean(ep, hist.get(col, []))
        if x:
            a.plot(x, y, color=c, lw=1.8); _label_end(a, x, y, l2, c)
    g = sel_min_frac
    if g is not None:
        a.axhline(float(g), color=INK_MUTED, lw=1.0, ls="--")
        a.annotate(f" selection gate {float(g):.2f}", (ep[0], float(g)),
                   color=INK_MUTED, fontsize=8, va="bottom")
    a.set_ylim(-0.02, 1.05); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- COLUMN 5: schedules + the merit controller --------------------------------
    # (0,4) the two weights that actually move. Log axis: w_merit spans orders of
    # magnitude once the controller is live, and w_control is scheduled.
    a = _style(ax[0][4], "Weight schedules", "epoch", "weight")
    for col, c, l2 in (("w_control", S1, "w_control"), ("w_merit", S2, "w_merit")):
        x, y = _clean(ep, hist.get(col, []))
        if x:
            a.plot(x, y, color=c, lw=1.8); _label_end(a, x, y, l2, c)
    a.set_yscale("log"); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (1,4) did the controller hit the share it aimed for? Both series are the NORM
    # share, which is what the control law inverts -- the projection share is not
    # monotone in w_merit and cannot be targeted.
    a = _style(ax[1][4], "Merit controller: share achieved", "epoch", "% of gradient norm")
    x, y = _clean(ep, hist.get("gshare_merit", []))
    if x:
        a.plot(x, y, color=S2, lw=1.8); _label_end(a, x, y, "merit", S2)
    xa, ya = _clean(ep, hist.get("gshare_anchor", []))
    if xa:
        a.plot(xa, ya, color=S1, lw=1.5, ls="--"); _label_end(a, xa, ya, "anchor", S1)
    if wm_boost_share is not None:
        a.axhline(100.0 * float(wm_boost_share), color=INK_MUTED, lw=1.0, ls=":")
        a.annotate(f" target {100*float(wm_boost_share):.0f}%",
                   ((x or xa or [0])[0], 100.0 * float(wm_boost_share)),
                   color=INK_MUTED, fontsize=8, va="bottom")
    a.set_ylim(0, 100); a.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (2,4) THE GT-ANCHOR LOSS. Every other panel showed the physics; this term was in
    # the objective and plotted nowhere, which is why the figure used to carry the note
    # "objective includes GT anchor (not plotted)".
    #
    # Two series, because they answer different questions. RAW is the anchor loss itself
    # -- under control_form: maxaligned that is mean(e) + mean(topk(e)) + 0.3*max(e) on
    # the tol-normalised control error, so it tracks the WORST controls rather than the
    # average and is the quantity the fine-tune is actually minimising. WEIGHTED is
    # w_control * raw, its contribution to the objective, which moves even when the raw
    # term does not because w_control is scheduled (and, with
    # loss.w_control_share_end set, re-solved every epoch from the measured gradients).
    a = _style(ax[2][4], "GT-anchor loss (max-aligned)", "epoch", "loss")
    xr, yr = _clean(ep, hist.get("control_term", []))
    if xr:
        a.plot(xr, yr, color=S1, lw=1.9); _label_end(a, xr, yr, "raw", S1)
    wcs = hist.get("w_control", [])
    if xr and wcs:
        wl = [(c * w if (c is not None and w) else None)
              for c, w in zip(hist.get("control_term", []), wcs)]
        xw, yw = _clean(ep, wl)
        if xw:
            a.plot(xw, yw, color=S2, lw=1.6, ls="--")
            _label_end(a, xw, yw, "x w_control", S2)
    if xr:
        a.set_yscale("log")
    else:
        a.text(0.5, 0.5, "no control_term recorded", transform=a.transAxes,
               ha="center", va="center", color=INK_MUTED, fontsize=9)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    lab = hist.get("_run_label", "")
    # Name the anchor that is actually in the objective. It said "RMSE→GT" on every
    # figure, which stopped being true when the default became the max-aligned
    # mean+top-k+max form -- a plot that misnames the objective it is describing is
    # worse than one that says nothing.
    # The anchor used to be absent from the figure; panel (2,4) plots it now, so the
    # old "(not plotted)" disclaimer would be wrong.
    note = ""
    fig.suptitle("Elastic fine-tune  |  live" + note, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(outdir, "elastic_training.png")
    fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    return p
