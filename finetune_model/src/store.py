"""LRU-backed access to the prepared cache, and batch collation.

The LRU capacity is derived from cache.mem_budget_gb and the measured bytes/case
rather than hardcoded per grid (the previous harness carried CASE_CAP=700 for 10k
and 2600 for tx2k as bare magic numbers, and it was the most likely OOM).
"""
from __future__ import annotations
import os
from collections import OrderedDict
import torch

from gridsfm.data import batch_data_list
from gridsfm.schema import (GEN_AVAIL_IDX, GEN_PMIN_IDX, GEN_PMAX_IDX,
                            BUS_VMIN_IDX, BUS_VMAX_IDX)
from masks import bus_masks


class CaseStore:
    def __init__(self, cache_dir: str, capacity: int, device: str = "cuda"):
        self.dir = cache_dir
        self.cap = max(1, int(capacity))
        self.device = device
        self._lru: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key + ".pt")

    def get(self, key: str):
        if key in self._lru:
            self._lru.move_to_end(key)
            self.hits += 1
            return self._lru[key]
        self.misses += 1
        o = torch.load(self._path(key), weights_only=False)
        rec = (o["data"],
               torch.tensor(o["pg"], dtype=torch.float64),
               torch.tensor(o["vm"], dtype=torch.float64))
        self._lru[key] = rec
        if len(self._lru) > self.cap:
            self._lru.popitem(last=False)
        return rec

    def batch(self, keys: list[str]) -> dict:
        """Collate same-topology cases into one block-diagonal batch, with GT and
        per-node normalisers concatenated in the SAME order, plus the segment ids
        the per-case top-k needs."""
        recs = [self.get(k) for k in keys]
        data = batch_data_list([r[0].clone() for r in recs]).to(self.device)
        pg = torch.cat([r[1] for r in recs]).to(self.device)
        vm = torch.cat([r[2] for r in recs]).to(self.device)
        m = bus_masks(data)
        gx = data["generator"].x.double()
        gav = gx[:, GEN_AVAIL_IDX]
        Vmin = data["bus"].x[:, BUS_VMIN_IDX].double()
        Vmax = data["bus"].x[:, BUS_VMAX_IDX].double()
        # segment ids: which case each generator / bus belongs to
        gen_seg = torch.cat([torch.full((r[0]["generator"].x.size(0),), i,
                                        dtype=torch.long) for i, r in enumerate(recs)]
                            ).to(self.device)
        bus_seg = torch.cat([torch.full((r[0]["bus"].x.size(0),), i,
                                        dtype=torch.long) for i, r in enumerate(recs)]
                            ).to(self.device)
        return dict(data=data, pg_gt=pg, vm_gt=vm, gav=gav, av=gav > 0.5,
                    vctrl=(m["pv"] | m["slack"]),
                    Prange=(gx[:, GEN_PMAX_IDX] - gx[:, GEN_PMIN_IDX]).clamp_min(1e-3),
                    Vband=(Vmax - Vmin).clamp_min(0.02),
                    gen_seg=gen_seg, bus_seg=bus_seg, nseg=len(recs))


def predict(model, S: dict):
    """Model controls: Pg per generator, V per bus (column 1 of the bus head)."""
    out = model(S["data"])
    return out["generator"].pred[:, 0].double(), out["bus"].pred[:, 1].double()
