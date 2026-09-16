"""Signed-incidence message passing."""
from __future__ import annotations

from typing import Tuple, Union

from torch import Tensor, nn
from torch_geometric.nn import MessagePassing


class SignedIncidenceConv(MessagePassing):
    """Signed bipartite message passing with mean aggregation (see module docstring)."""

    _SMALL_INIT_STD = 0.02

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
    ):
        super().__init__(aggr='mean', flow='source_to_target', node_dim=0)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = int(out_channels)
        self.lin_msg  = nn.Linear(in_channels[0], out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels[1], out_channels, bias=True)
        self.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x
        else:
            x_src, x_dst = x, x

        h_src = self.lin_msg(x_src)

        size = (int(x_src.size(0)), int(x_dst.size(0)))
        out = self.propagate(edge_index, x=(h_src, None),
                             edge_attr=edge_attr, size=size)

        return out + self.lin_self(x_dst)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        sign = edge_attr[:, 0:1]
        return x_j * sign

    def reset_parameters(self):
        nn.init.normal_(self.lin_msg.weight,  mean=0.0, std=self._SMALL_INIT_STD)
        nn.init.normal_(self.lin_self.weight, mean=0.0, std=self._SMALL_INIT_STD)
        if self.lin_self.bias is not None:
            nn.init.zeros_(self.lin_self.bias)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.in_channels}, "
                f"{self.out_channels})")
