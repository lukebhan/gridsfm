"""Grid statistics and perturbation-parameter validation.

Setting the load-scale ceiling above what a grid can physically serve makes a large
fraction of `loads` scenarios infeasible by construction, and the cost is only visible
after the solve budget has been spent. ACTIVSg10k, for instance, runs at 89% of its
active generation capacity, so anything above 1.13x is unservable. This module computes
the headroom from the Matpower file and refuses configs that exceed it.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class GridStats:
    name: str
    n_bus: int
    n_gen: int
    n_gen_active: int
    n_branch: int
    n_branch_rated: int
    n_ctrl_bus: int          # PV + slack: the buses a control model actually sets
    total_pd: float          # MW
    total_pmax: float        # MW, active gens
    total_pmin: float        # MW, active gens
    base_mva: float

    @property
    def loading(self) -> float:
        return self.total_pd / self.total_pmax

    @property
    def lossless_sf_ceiling(self) -> float:
        """Highest uniform load scale factor the active fleet could serve with zero losses.
        A hard upper bound: real congestion and losses make the true ceiling lower."""
        return self.total_pmax / self.total_pd

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_bus} buses, {self.n_gen} gens ({self.n_gen_active} active), "
            f"{self.n_branch} branches ({self.n_branch_rated} rated), {self.n_ctrl_bus} control buses\n"
            f"  load {self.total_pd:,.0f} MW / Pmax {self.total_pmax:,.0f} MW = "
            f"{self.loading:.1%} loaded  ->  lossless sf ceiling {self.lossless_sf_ceiling:.2f}x\n"
            f"  Pmin/Pmax {self.total_pmin/self.total_pmax:.2f} (dispatch freedom), baseMVA {self.base_mva:g}"
        )


def _section(text: str, name: str):
    m = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\n\s*\];", text, re.S)
    if m is None:
        return []
    rows = []
    for line in m.group(1).strip().split("\n"):
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append(line.split())
    return rows


def read_matpower(path: str, name: str | None = None) -> GridStats:
    """Parse the fields needed for validation straight out of the .m file.

    Deliberately not a full Matpower parser: PowerModels does the real parsing at solve
    time. This only needs enough to judge whether the perturbation ranges are physical.
    """
    text = open(path).read()
    bus, gen, branch = _section(text, "bus"), _section(text, "gen"), _section(text, "branch")
    if not bus or not gen:
        raise ValueError(f"{path}: could not parse mpc.bus / mpc.gen")
    bm = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text)

    # Matpower column order: bus[1]=type bus[2]=Pd ; gen[7]=status gen[8]=Pmax gen[9]=Pmin
    active = [g for g in gen if float(g[7]) != 0.0]
    types = [int(float(b[1])) for b in bus]
    return GridStats(
        name=name or path.rsplit("/", 1)[-1],
        n_bus=len(bus), n_gen=len(gen), n_gen_active=len(active),
        n_branch=len(branch),
        n_branch_rated=sum(1 for b in branch if len(b) > 5 and float(b[5]) > 0),
        n_ctrl_bus=sum(1 for t in types if t in (2, 3)),
        total_pd=sum(float(b[2]) for b in bus),
        total_pmax=sum(float(g[8]) for g in active),
        total_pmin=sum(float(g[9]) for g in active),
        base_mva=float(bm.group(1)) if bm else 100.0,
    )


def validate_perturbations(stats: GridStats, p: dict, strict: bool = True) -> list[str]:
    """Check perturbation ranges against what the grid can physically do.
    Returns warnings; raises on hard errors when strict."""
    warn: list[str] = []
    ceil_ = stats.lossless_sf_ceiling
    lo, hi = p["load_sf_lo"], p["load_sf_hi"]

    if hi > ceil_:
        msg = (f"load_sf_hi={hi} exceeds the lossless capacity ceiling {ceil_:.2f}x "
               f"({stats.loading:.0%} loaded). Roughly "
               f"{100*(hi-ceil_)/(hi-lo):.0f}% of `loads` scenarios will be infeasible by "
               f"construction. Set load_sf_hi <= {ceil_:.2f} (allow margin for losses).")
        # Deliberate breach: the top of the load range is knowingly put above what the
        # intact fleet can serve, to buy diversity at the high end. It costs SOLVE TIME,
        # not correctness -- infeasible scenarios are discarded and never published, so
        # the dataset is unaffected; the run simply explores more cases to reach its
        # feasible target. Opt in per config so the cost is a recorded choice rather
        # than an accident, which is the failure this check exists to prevent.
        if p.get("allow_load_sf_above_ceiling", False):
            warn.append("ACCEPTED BREACH (allow_load_sf_above_ceiling): " + msg)
        elif strict:
            raise ValueError(msg)
        else:
            warn.append(msg)
    elif hi > 0.97 * ceil_:
        warn.append(f"load_sf_hi={hi} is within 3% of the ceiling {ceil_:.2f}x; expect a low `loads` yield.")

    # low end: the fleet cannot go below its own Pmin
    if lo * stats.total_pd < stats.total_pmin:
        warn.append(f"load_sf_lo={lo} puts load ({lo*stats.total_pd:,.0f} MW) below total Pmin "
                    f"({stats.total_pmin:,.0f} MW); expect infeasible low-load scenarios.")

    nk = max(p["killgen_nk"])
    if nk >= stats.n_gen_active - 2:
        raise ValueError(f"killgen_nk max {nk} leaves <2 active gens of {stats.n_gen_active}.")
    if nk / stats.n_gen_active < 0.001:
        warn.append(f"killgen_nk max {nk} is only {100*nk/stats.n_gen_active:.2f}% of "
                    f"{stats.n_gen_active} active gens - a weak perturbation at this scale.")

    if stats.n_branch_rated < stats.n_branch:
        warn.append(f"{stats.n_branch - stats.n_branch_rated} of {stats.n_branch} branches have "
                    f"rate_a=0 and are therefore never selected by `derate`.")
    return warn
