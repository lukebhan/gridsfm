"""Training-set subsetting for the data-scaling sweep.

Two properties matter, and a naive `random.sample` has neither:

  STRATIFIED -- each perturbation mode contributes in proportion to its share of the
  split. At n=10 over 5 modes a blind sample can easily omit a mode entirely, so the
  10-case model would be measuring "which modes did we happen to draw" rather than
  "how much does 10 cases buy".

  NESTED -- subset(10) is a subset of subset(100), which is a subset of subset(500),
  and so on. Without nesting, the difference between two sizes mixes the effect of
  MORE DATA with the effect of DIFFERENT DATA, and the scaling curve is unreadable.
  Achieved by fixing a per-mode order once from the seed and taking a prefix of it.
"""
from __future__ import annotations
import collections
import random


def _apportion(pop: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split of `total` across modes, proportional to population."""
    n = sum(pop.values())
    if n == 0 or total <= 0:
        return {m: 0 for m in pop}
    if total >= n:
        return dict(pop)
    exact = {m: total * c / n for m, c in pop.items()}
    base = {m: min(int(v), pop[m]) for m, v in exact.items()}
    left = total - sum(base.values())
    order = sorted(pop, key=lambda m: (-(exact[m] - int(exact[m])), m))
    i = 0
    while left > 0 and i < len(order) * 4:
        m = order[i % len(order)]
        if base[m] < pop[m]:
            base[m] += 1
            left -= 1
        i += 1
    return base


def train_subset(keys: list[str], modes: list[str], n: int, seed: int = 0) -> list[str]:
    """First `n` cases of a stratified, nested ordering. n<=0 or n>=len(keys) -> all."""
    if n <= 0 or n >= len(keys):
        return list(keys)
    by: dict[str, list[str]] = collections.defaultdict(list)
    for k, m in zip(keys, modes):
        by[m or "?"].append(k)
    for m in by:                                  # fixed order per mode -> prefixes nest
        by[m].sort()
        random.Random(f"{seed}|{m}").shuffle(by[m])
    quota = _apportion({m: len(v) for m, v in by.items()}, n)
    out: list[str] = []
    for m in sorted(by):
        out.extend(by[m][:quota[m]])
    return out


def subset_label(n: int) -> str:
    """Directory-safe run label: n0010, n0100, n1000, or `full`."""
    return "full" if n <= 0 else f"n{n:04d}"
