"""Prediction-accuracy evaluation and TEST diagnostics (no solve in the loop).

Two metrics, both over the CONTROLS the model actually sets:
  pg_pct  -- sum|Pg err| / sum|Pg gt| over AVAILABLE generators, as a percent.
             Aggregate-relative, not mean-of-ratios: small units would otherwise
             dominate a per-unit percentage.
  v_pu    -- mean |V err| in p.u. over PV/slack buses.
"""
from __future__ import annotations
import numpy as np
import torch

from store import predict


@torch.no_grad()
def eval_pred(model, store, keys: list[str], tag: str, bs: int = 16) -> tuple[float, float]:
    model.eval()
    ne = de = vsum = 0.0
    vn = 0
    for i in range(0, len(keys), bs):
        S = store.batch(keys[i:i + bs])
        Pg, V = predict(model, S)
        av, vc = S["av"], S["vctrl"]
        ne += float((Pg[av] - S["pg_gt"][av]).abs().sum())
        de += float(S["pg_gt"][av].abs().sum())
        vsum += float((V[vc] - S["vm_gt"][vc]).abs().sum())
        vn += int(vc.sum())
    pg_pct = 100 * ne / max(de, 1e-9)
    v_pu = vsum / max(vn, 1)
    print(f"[{tag}] Pg={pg_pct:.2f}%  Vctrl={v_pu:.4f}pu", flush=True)
    return pg_pct, v_pu


@torch.no_grad()
def collect_diag(model, store, keys: list[str]) -> dict:
    """Per-element TEST errors for the diagnostics figure. One case at a time so
    per-case statistics (worst generator, per-case error) are exact."""
    model.eval()
    pg_pred, pg_gt, v_err, worst, per_case_pct = [], [], [], [], []
    for k in keys:
        S = store.batch([k])
        Pg, V = predict(model, S)
        av, vc = S["av"], S["vctrl"]
        p = Pg[av].cpu().numpy(); g = S["pg_gt"][av].cpu().numpy()
        ve = (V[vc] - S["vm_gt"][vc]).abs().cpu().numpy()
        pg_pred.append(p); pg_gt.append(g); v_err.append(ve)
        worst.append(float(np.abs(p - g).max()) if p.size else 0.0)
        per_case_pct.append(100 * float(np.abs(p - g).sum()) / max(float(np.abs(g).sum()), 1e-9))
    pg_pred = np.concatenate(pg_pred); pg_gt = np.concatenate(pg_gt)
    v_err = np.concatenate(v_err) if v_err else np.zeros(0)
    err = np.abs(pg_pred - pg_gt)
    return dict(
        pg_pred=pg_pred, pg_gt=pg_gt, pg_abs_err=err, v_abs_err=v_err,
        per_case_worst_pg=np.array(worst), per_case_pg_pct=np.array(per_case_pct),
        pg_err_p50=float(np.percentile(err, 50)), pg_err_p90=float(np.percentile(err, 90)),
        pg_err_p99=float(np.percentile(err, 99)), pg_err_max=float(err.max()),
        v_err_p50=float(np.percentile(v_err, 50)) if v_err.size else 0.0,
        v_err_p90=float(np.percentile(v_err, 90)) if v_err.size else 0.0,
        v_err_p99=float(np.percentile(v_err, 99)) if v_err.size else 0.0,
        v_err_max=float(v_err.max()) if v_err.size else 0.0,
    )
