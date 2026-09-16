"""Per-bus control roles and per-case control counts.

`bus_masks` is lifted verbatim (behaviour-wise) from 10k/finetune.py: it is the
only thing training needs from the AC-PF toolkit, so the full PF solver is not a
dependency of this harness.

The DEMOTION matters and is easy to miss: a PV bus whose available generators have
no net reactive range cannot hold voltage, so it is scored as PQ. On ACTIVSg10k
that turns 1728 nominal control buses into ~1255 actually-scored ones -- which is
why the top-k widths are derived from THESE counts, not from a static grid stat.
"""
from __future__ import annotations
import torch
from gridsfm.schema import (GEN_AVAIL_IDX, GEN_QMIN_IDX, GEN_QMAX_IDX,
                            BUS_TYPE_IDX)


def bus_masks(data) -> dict:
    bt = data["bus"].x[:, BUS_TYPE_IDX].long()
    slack = (bt == 3); pv = (bt == 2); pq = (bt == 1)
    gkey = ("generator", "generator_link", "bus")
    if ("generator" in data.node_types and data["generator"].x.size(0) > 0
            and gkey in data.edge_types and data[gkey].edge_index.numel() > 0):
        n = data["bus"].x.size(0)
        gx = data["generator"].x
        gav = gx[:, GEN_AVAIL_IDX]
        qrange = torch.zeros(n, dtype=gx.dtype, device=gx.device)
        qrange.scatter_add_(0, data[gkey].edge_index[1],
                            (gx[:, GEN_QMAX_IDX] - gx[:, GEN_QMIN_IDX]) * gav)
        no_reg = pv & (qrange < 1e-6)
        pv = pv & (~no_reg)
        pq = pq | no_reg
    return dict(slack=slack, pv=pv, pq=pq, nonslack=(~slack))


def scored_counts(data) -> tuple[int, int]:
    """(available generators, V-control buses) for one prepared case -- the two
    populations the loss and the metrics are computed over."""
    m = bus_masks(data)
    gav = data["generator"].x[:, GEN_AVAIL_IDX]
    return int((gav > 0.5).sum()), int((m["pv"] | m["slack"]).sum())
