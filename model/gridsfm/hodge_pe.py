"""Hodge positional encoding via signed Hodge diffusion."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData


def _signed_scatter(
    src: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    sign: Tensor,
    n_dst: int,
) -> Tensor:
    if src_idx.numel() == 0:
        return torch.zeros(n_dst, src.size(-1), dtype=src.dtype, device=src.device)
    weighted = sign.unsqueeze(-1) * src[src_idx]
    out = torch.zeros(n_dst, src.size(-1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(weighted), weighted)
    return out


class HodgePE(nn.Module):
    """Cross-grid positional encoding via signed Hodge diffusion."""

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        K: int,
        T: int,
        active_forms: tuple,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.T = int(T)
        self.active_forms = tuple(
            f for f in active_forms if f in node_feature_dims
        )

        self.init_mlp = nn.ModuleDict()
        for nt in self.active_forms:
            self.init_mlp[nt] = nn.Linear(node_feature_dims[nt], self.K)

        self.alpha = nn.ParameterDict()
        for nt in self.active_forms:
            self.alpha[nt] = nn.Parameter(torch.full((self.T, self.K), 0.05))

        self.mix_mlp = nn.ModuleDict()
        for nt in self.active_forms:
            self.mix_mlp[nt] = nn.Sequential(
                nn.Linear((self.T + 1) * self.K, 2 * self.K),
                nn.GELU(),
                nn.Linear(2 * self.K, self.K),
            )

        self.out_ln = nn.ModuleDict()
        for nt in self.active_forms:
            self.out_ln[nt] = nn.LayerNorm(self.K)

        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.init_mlp.values():
            nn.init.normal_(mod.weight, std=0.02)
            nn.init.zeros_(mod.bias)
        for seq in self.mix_mlp.values():
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)


    @staticmethod
    def _L0_bus(x_bus: Tensor, data: HeteroData) -> Tensor:
        n_bus = x_bus.size(0)
        branch_x = {}
        for nt_branch in ("branch_ac", "branch_tr"):
            et = ("bus", "endpoint_of", nt_branch)
            if et not in data.edge_types or data[et].edge_index.size(1) == 0:
                continue
            if nt_branch not in data.node_types or data[nt_branch].x.size(0) == 0:
                continue
            ei = data[et].edge_index
            ea = data[et].edge_attr
            sign = ea[:, 0].to(x_bus.dtype)
            n_branch = data[nt_branch].x.size(0)
            branch_x[nt_branch] = _signed_scatter(
                x_bus, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_branch,
            )
        bus_acc = torch.zeros_like(x_bus)
        for nt_branch in ("branch_ac", "branch_tr"):
            et = (nt_branch, "endpoint_of", "bus")
            if et not in data.edge_types or data[et].edge_index.size(1) == 0:
                continue
            if nt_branch not in branch_x:
                continue
            ei = data[et].edge_index
            ea = data[et].edge_attr
            sign = ea[:, 0].to(x_bus.dtype)
            bus_acc = bus_acc + _signed_scatter(
                branch_x[nt_branch], src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_bus,
            )
        return bus_acc

    @staticmethod
    def _L1_branch(x_branch: Tensor, nt_branch: str, data: HeteroData) -> Tensor:
        n_branch = x_branch.size(0)
        n_bus = data["bus"].x.size(0)
        n_cycle = data["cycle"].x.size(0) if "cycle" in data.node_types else 0

        bus_y = torch.zeros(n_bus, x_branch.size(-1),
                            dtype=x_branch.dtype, device=x_branch.device)
        et = (nt_branch, "endpoint_of", "bus")
        if et in data.edge_types and data[et].edge_index.size(1) > 0:
            ei = data[et].edge_index
            ea = data[et].edge_attr
            sign = ea[:, 0].to(x_branch.dtype)
            bus_y = _signed_scatter(
                x_branch, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_bus,
            )
        et2 = ("bus", "endpoint_of", nt_branch)
        down = torch.zeros_like(x_branch)
        if et2 in data.edge_types and data[et2].edge_index.size(1) > 0:
            ei = data[et2].edge_index
            ea = data[et2].edge_attr
            sign = ea[:, 0].to(x_branch.dtype)
            down = _signed_scatter(
                bus_y, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_branch,
            )

        up = torch.zeros_like(x_branch)
        if n_cycle > 0:
            et = (nt_branch, "in_cycle", "cycle")
            cycle_y = torch.zeros(n_cycle, x_branch.size(-1),
                                  dtype=x_branch.dtype, device=x_branch.device)
            if et in data.edge_types and data[et].edge_index.size(1) > 0:
                ei = data[et].edge_index
                ea = data[et].edge_attr
                sign = ea[:, 0].to(x_branch.dtype)
                cycle_y = _signed_scatter(
                    x_branch, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_cycle,
                )
            et2 = ("cycle", "in_cycle", nt_branch)
            if et2 in data.edge_types and data[et2].edge_index.size(1) > 0:
                ei = data[et2].edge_index
                ea = data[et2].edge_attr
                sign = ea[:, 0].to(x_branch.dtype)
                up = _signed_scatter(
                    cycle_y, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_branch,
                )

        return down + up

    @staticmethod
    def _L2_cycle(x_cycle: Tensor, data: HeteroData) -> Tensor:
        n_cycle = x_cycle.size(0)
        branch_y = {}
        for nt_branch in ("branch_ac", "branch_tr"):
            et = ("cycle", "in_cycle", nt_branch)
            if et not in data.edge_types or data[et].edge_index.size(1) == 0:
                continue
            n_branch = data[nt_branch].x.size(0)
            ei = data[et].edge_index
            ea = data[et].edge_attr
            sign = ea[:, 0].to(x_cycle.dtype)
            branch_y[nt_branch] = _signed_scatter(
                x_cycle, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_branch,
            )
        cycle_acc = torch.zeros_like(x_cycle)
        for nt_branch, y in branch_y.items():
            et = (nt_branch, "in_cycle", "cycle")
            if et not in data.edge_types or data[et].edge_index.size(1) == 0:
                continue
            ei = data[et].edge_index
            ea = data[et].edge_attr
            sign = ea[:, 0].to(x_cycle.dtype)
            cycle_acc = cycle_acc + _signed_scatter(
                y, src_idx=ei[0], dst_idx=ei[1], sign=sign, n_dst=n_cycle,
            )
        return cycle_acc


    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}

        for nt in self.active_forms:
            if nt not in data.node_types or data[nt].x.size(0) == 0:
                continue
            x_in = data[nt].x
            x = self.init_mlp[nt](x_in)
            states = [x]
            for t in range(self.T):
                if nt == "bus":
                    Lx = self._L0_bus(x, data)
                elif nt in ("branch_ac", "branch_tr"):
                    Lx = self._L1_branch(x, nt, data)
                elif nt == "cycle":
                    Lx = self._L2_cycle(x, data)
                else:
                    Lx = torch.zeros_like(x)
                x = x - self.alpha[nt][t].unsqueeze(0) * Lx
                states.append(x)
            stacked = torch.cat(states, dim=-1)
            pe = self.mix_mlp[nt](stacked)
            pe = self.out_ln[nt](pe)
            out[nt] = pe

        return out
