"""Run history at two granularities.

The previous harness recorded EPOCH means only, which hides exactly the things you
need per-iteration data for: loss spikes, the clipping regime (the 10k run's
pre-clip gradient norm peaked at 8069 against clip=1.0 -- invisible in an epoch
mean), and warmup pathologies. So both are written:

  history_step.csv   one row per optimizer step
  history_epoch.csv  one row per epoch, plus history_epoch.json for plotting
  timing.json        where the wall time went
"""
from __future__ import annotations
import json, os

EPOCH_COLS = ["epoch", "loss", "loss_pg", "loss_v", "val_pg", "val_v", "val_score",
              "lr", "gnorm_mean", "gnorm_max", "gnorm_p99", "clip_frac",
              "epoch_sec", "data_sec", "fwd_bwd_sec", "eval_sec",
              "gpu_peak_gb", "cache_hit_frac", "is_best",
              # --- elastic mode (loss.mode: elastic) ---
              "elastic_loss", "val_elastic_loss", "cost", "eq_pen", "ineq_pen",
              "control_term", "w_control", "closure_tol", "newton_iters", "max_absF",
              "max_Vviol", "max_thermal", "max_Qviol",
              # --- per-term GRADIENT share (train.grad_probe_every > 0) ---
              # gshare_X = ||g_X|| / sum_i ||g_i||     : relative magnitude.
              # gproj_X  = (g_X . g_tot)/||g_tot||^2   : share of the REALISED step
              #   direction; sums to 100% across terms, and is NEGATIVE for a term
              #   pulling against the consensus. Both are recorded because they
              #   disagree whenever terms fight each other -- on case6470_rte the GT
              #   anchor sits at cos = -0.89 to the equality residual, so magnitude
              #   share overstates what it actually wins.
              # NOTE add_epoch() iterates THIS list and silently drops any kwarg not in
              # it, so a new diagnostic must be registered here or it is written nowhere.
              "gshare_anchor", "gshare_merit", "gshare_eq_pen", "gshare_ineq_pen",
              "gshare_cost",
              "gproj_anchor", "gproj_merit", "gproj_eq_pen", "gproj_ineq_pen",
              "gproj_cost",
              # --- PCGrad (loss.pcgrad) ---
              # pcgrad_cos: cosine between the NAIVE summed gradient and the projected
              #   one. 1.0 = surgery changed nothing (no conflicts); lower = the terms
              #   were fighting and the step was redirected.
              # pcgrad_conflicts: mean number of conflicting ORDERED term pairs per
              #   batch (max = k*(k-1) for k terms).
              "pcgrad_cos", "pcgrad_conflicts",
              # merit = 0.5*mean(F^2), the feasibility-descent term that gives STALLED
              # closures a gradient (the IFT adjoint is invalid at a non-root, so they
              # used to contribute zero). frac_converged is the fraction of the batch
              # whose closure actually converged -- max_absF is a MAX and hides this.
              "merit", "w_merit", "frac_converged",
              # val_frac_converged is the SELECTION GATE's actual input, and it was
              # missing here while train_elastic passed it -- add_epoch dropped it
              # silently, so a run that shipped ship_epoch=0 could not be diagnosed from
              # its own history and the train-side frac_converged got read in its place.
              # max_pg_err / max_v_err are the worst single control errors in the epoch,
              # in physical p.u.; the training figure plots them.
              "val_frac_converged", "max_pg_err", "max_v_err"]
STEP_COLS = ["step", "epoch", "loss", "loss_pg", "loss_v", "gnorm", "lr", "wall_sec"]


class History:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.epoch: dict[str, list] = {k: [] for k in EPOCH_COLS}
        self._step_path = os.path.join(outdir, "history_step.csv")
        with open(self._step_path, "w") as f:                    # header once
            f.write(",".join(STEP_COLS) + "\n")
        self._step_buf: list[str] = []

    def add_step(self, **kw):
        self._step_buf.append(",".join(f"{kw.get(c, '')!s}" for c in STEP_COLS))
        if len(self._step_buf) >= 64:                            # cheap, still tail-able
            self.flush_steps()

    def flush_steps(self):
        if self._step_buf:
            with open(self._step_path, "a") as f:
                f.write("\n".join(self._step_buf) + "\n")
            self._step_buf.clear()

    def add_epoch(self, **kw):
        for c in EPOCH_COLS:
            self.epoch[c].append(kw.get(c))
        self.flush_steps()
        self.write()

    def write(self):
        with open(os.path.join(self.outdir, "history_epoch.json"), "w") as f:
            json.dump(self.epoch, f, indent=2)
        with open(os.path.join(self.outdir, "history_epoch.csv"), "w") as f:
            f.write(",".join(EPOCH_COLS) + "\n")
            n = len(self.epoch["epoch"])
            for i in range(n):
                f.write(",".join("" if self.epoch[c][i] is None else str(self.epoch[c][i])
                                 for c in EPOCH_COLS) + "\n")

    def write_timing(self, extra: dict):
        e = self.epoch
        tot = sum(x for x in e["epoch_sec"] if x)
        d = dict(
            total_train_seconds=tot, total_train_hours=tot / 3600,
            data_seconds=sum(x for x in e["data_sec"] if x),
            fwd_bwd_seconds=sum(x for x in e["fwd_bwd_sec"] if x),
            eval_seconds=sum(x for x in e["eval_sec"] if x),
            mean_epoch_seconds=tot / max(len(e["epoch"]), 1),
            gpu_peak_gb=max([x for x in e["gpu_peak_gb"] if x] or [0]),
            **extra)
        with open(os.path.join(self.outdir, "timing.json"), "w") as f:
            json.dump(d, f, indent=2)
        return d
