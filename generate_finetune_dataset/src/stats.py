"""End-of-run dataset statistics, computed from the durable per-case records.

Records cover every attempted case, feasible or not, so infeasible cases are described
too. They are the expensive ones, since they run to Ipopt's iteration cap before failing,
and they are what determines the yield.
"""
from __future__ import annotations
import glob, json, os, statistics as st

# Severity scalar per mode, and the precision to print it at.
SEVERITY = {"loads": ("system_load_factor", 4), "costs": ("cost_shuffle_pct", 1),
            "killgen": ("n_gens_killed", 1), "derate": ("derate_factor_mean", 4),
            "vsqueeze": ("band_shrink_mean", 5)}


def load_records(recdir: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(recdir, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # tolerate a torn final line from a killed worker
    # de-duplicate on (mode, sidx): a resumed run may re-record a case
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r.get("mode", ""), r.get("sidx", 0))):
        k = (r.get("mode"), r.get("sidx"))
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def _agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return dict(n=len(v), min=min(v), max=max(v), mean=st.mean(v), median=st.median(v))


def _fmt(a, prec=2):
    if a is None:
        return "        -        -        -        -"
    return (f"{a['min']:>9.{prec}f}{a['max']:>9.{prec}f}"
            f"{a['mean']:>9.{prec}f}{a['median']:>9.{prec}f}")


def report(recs: list[dict], modes: list[str], max_iters: int) -> str:
    """Human-readable statistics block. Everything here is measured, nothing estimated."""
    L = []
    F = [r for r in recs if r.get("feasible")]
    I = [r for r in recs if not r.get("feasible")]
    tot_s = sum(r.get("seconds", 0) or 0 for r in recs)
    P = [r for r in recs if r.get("mode") != "base"]

    L.append("=" * 92)
    L.append("DATASET STATISTICS")
    L.append("=" * 92)
    L.append(f"  explored {len(recs)} scenarios -> {len(F)} feasible "
             f"({100*len(F)/max(len(recs),1):.1f}%), {len(I)} infeasible")
    L.append(f"  total solver runtime {tot_s/3600:.2f} h ({tot_s/max(len(recs),1):.1f} s/case avg)")
    fs = sum(r.get('seconds', 0) or 0 for r in F); isec = sum(r.get('seconds', 0) or 0 for r in I)
    L.append(f"    feasible {fs/3600:.2f} h | infeasible {isec/3600:.2f} h "
             f"({100*isec/max(tot_s,1e-9):.0f}% of runtime spent on cases that were discarded)")

    nmodes = _agg([r.get("n_modes_applied") for r in P])
    if nmodes:
        L.append(f"  perturbations per scenario: mean {nmodes['mean']:.2f} "
                 f"(min {nmodes['min']:.0f}, max {nmodes['max']:.0f})")

    L.append("")
    L.append("  MARGINAL YIELD BY MODE")
    L.append("  Scenarios compose their perturbations, so a case cannot be attributed to")
    L.append("  one mode. Each row is over the scenarios in which that mode FIRED, and the")
    L.append("  rows therefore overlap.")
    L.append(f"  {'mode':<10}{'fired':>9}{'share':>8}{'feasible':>10}{'yield':>8}")
    for m in modes:
        mr = [r for r in P if m in (r.get("modes_applied") or [])]
        if not mr:
            continue
        mf = sum(1 for r in mr if r.get("feasible"))
        L.append(f"  {m:<10}{len(mr):>9}{100*len(mr)/max(len(P),1):>7.0f}%"
                 f"{mf:>10}{100*mf/len(mr):>7.0f}%")

    L.append("")
    L.append("  SEVERITY BY MODE (over the scenarios in which the mode fired)")
    L.append("  loads=load factor  costs=% gens shuffled  killgen=#units tripped  "
             "derate=mean factor  vsqueeze=mean band shrink p.u.")
    L.append(f"  {'mode':<10}{'set':<10}{'min':>9}{'max':>9}{'mean':>9}{'median':>9}")
    for m in modes:
        key, prec = SEVERITY.get(m, (None, 2))
        if key is None:
            continue
        mr = [r for r in P if m in (r.get("modes_applied") or [])]
        for label, sub in (("feasible", [r for r in mr if r.get("feasible")]),
                           ("infeas", [r for r in mr if not r.get("feasible")])):
            L.append(f"  {m:<10}{label:<10}{_fmt(_agg([r.get(key) for r in sub]), prec)}")

    L.append("")
    L.append("  IPOPT ITERATIONS")
    L.append(f"  {'set':<10}{'':<10}{'min':>9}{'max':>9}{'mean':>9}{'median':>9}")
    for label, sub in (("feasible", F), ("infeasible", I)):
        a = _agg([r.get("iters") for r in sub if (r.get("iters") or -1) > 0])
        L.append(f"  {label:<10}{'':<10}{_fmt(a, 0)}")
    capped = sum(1 for r in recs if (r.get("iters") or 0) >= max_iters)
    L.append(f"  hit the {max_iters}-iteration cap: {capped}"
             + ("  <-- raise --max_iters if these should have converged" if capped else ""))

    L.append("")
    L.append("  SOLVE TIME (s)")
    L.append(f"  {'set':<10}{'':<10}{'min':>9}{'max':>9}{'mean':>9}{'median':>9}")
    for label, sub in (("feasible", F), ("infeasible", I)):
        L.append(f"  {label:<10}{'':<10}{_fmt(_agg([r.get('seconds') for r in sub]), 1)}")

    obj = _agg([r.get("objective") for r in F])
    if obj:
        L.append("")
        L.append("  OBJECTIVE (feasible cases) - spread shows how much the perturbations move cost")
        L.append(f"    min {obj['min']:,.0f}  max {obj['max']:,.0f}  mean {obj['mean']:,.0f}  "
                 f"median {obj['median']:,.0f}  (max/min = {obj['max']/max(obj['min'],1e-9):.2f}x)")

    st_counts: dict[str, int] = {}
    for r in I:
        st_counts[r.get("status", "?")] = st_counts.get(r.get("status", "?"), 0) + 1
    if st_counts:
        L.append("")
        L.append("  INFEASIBLE TERMINATION STATUSES")
        for k, v in sorted(st_counts.items(), key=lambda kv: -kv[1]):
            L.append(f"    {k:<40} {v}")
    L.append("=" * 92)
    return "\n".join(L)
