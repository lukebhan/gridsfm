"""GridSFM AC-OPF inference backbone."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from .blocks import (
    _G_CTX_DIM,
    EdgeType,
    DEFAULT_INPUT_DIMS,
    _nan_to_zero,
    _leaky_clip,
    _sigmoid_clip,
    _slack_or_mean_anchor,
    GridBlock,
    FusionLayer,
)
from .dc_prior import DCPriorCache, compute_dc_prior
from .stress_features import compute_physics_stress, STRESS_DIM
from .hodge_pe import HodgePE
from .schema import (
    AC_LINE_BFR_IDX, AC_LINE_BTO_IDX, AC_LINE_R_IDX, AC_LINE_X_IDX,
    BUS_VMIN_IDX, BUS_VMAX_IDX,
    GEN_PMIN_IDX, GEN_PMAX_IDX, GEN_QMIN_IDX, GEN_QMAX_IDX,
    PI_Z2_EPS, PI_TAP_EPS,
    TR_R_IDX, TR_X_IDX, TR_TAP_IDX, TR_SHIFT_IDX, TR_BFR_IDX, TR_BTO_IDX,
)


class GridTransformerBackbone(nn.Module):
    """Top-level GridSFM backbone: trunk + 5 heads. Writes predictions onto `data`."""

    NODE_TYPES = ['bus', 'generator', 'load', 'shunt', 'branch_ac', 'branch_tr', 'cycle']

    EDGE_TYPES = [
        ('bus', 'endpoint_of', 'branch_ac'),
        ('branch_ac', 'endpoint_of', 'bus'),
        ('bus', 'endpoint_of', 'branch_tr'),
        ('branch_tr', 'endpoint_of', 'bus'),
        ('cycle', 'in_cycle', 'branch_ac'),
        ('branch_ac', 'in_cycle', 'cycle'),
        ('cycle', 'in_cycle', 'branch_tr'),
        ('branch_tr', 'in_cycle', 'cycle'),
        ('generator', 'generator_link', 'bus'),
        ('bus', 'generator_link', 'generator'),
        ('load', 'load_link', 'bus'),
        ('bus', 'load_link', 'load'),
        ('shunt', 'shunt_link', 'bus'),
        ('bus', 'shunt_link', 'shunt'),
    ]

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 8,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        input_dims: Optional[Dict[str, int]] = None,
        leaky_alpha: float = 0.0,
        input_norm: bool = True,
        signed_fusion: bool = True,
        hodge_pe_dim: int = 8,
        hodge_pe_steps: int = 4,
        theta_res_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.leaky_alpha = float(leaky_alpha)
        # 'clamp' = released hard clamp (dead gradient at limits); 'sigmoid' = smooth
        # strictly-in-range squash (live gradient everywhere). Set externally for fine-tuning.
        self.clip_mode = 'clamp'
        self.theta_res_scale = float(theta_res_scale)
        self.NODE_TYPES = list(self.__class__.NODE_TYPES)
        self.EDGE_TYPES = list(self.__class__.EDGE_TYPES)
        self._dc_cache = DCPriorCache(max_cache=64)
        self.input_norm_enabled = bool(input_norm)
        self.signed_fusion = bool(signed_fusion)

        in_dims = dict(DEFAULT_INPUT_DIMS)
        if input_dims is not None:
            in_dims.update(input_dims)

        self.hodge_pe_dim = int(hodge_pe_dim)
        self.hodge_pe_steps = int(hodge_pe_steps)
        hodge_active = tuple(
            f for f in ("bus", "branch_ac", "branch_tr", "cycle")
            if f in self.NODE_TYPES
        )
        self.hodge_pe = HodgePE(
            node_feature_dims={f: in_dims[f] for f in hodge_active},
            K=self.hodge_pe_dim,
            T=self.hodge_pe_steps,
            active_forms=hodge_active,
        )
        for f in hodge_active:
            in_dims[f] = in_dims[f] + self.hodge_pe_dim
        self.in_dims = in_dims

        if self.input_norm_enabled:
            self.input_ln = nn.ModuleDict({
                nt: nn.LayerNorm(in_dims[nt]) for nt in self.NODE_TYPES
            })
        else:
            self.input_ln = None

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_dims[nt], hidden_dim) for nt in self.NODE_TYPES
        })

        self.blocks = nn.ModuleList([
            GridBlock(
                node_types=self.NODE_TYPES,
                edge_types=self.EDGE_TYPES,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.fusion = FusionLayer(hidden_dim, signed_fusion=self.signed_fusion)

        mid = max(1, hidden_dim // 2)

        def _make_head(in_dim: int, out_dim: int, bias_init: float) -> nn.Sequential:
            h = nn.Sequential(
                nn.Linear(in_dim, mid), nn.GELU(), nn.Linear(mid, out_dim)
            )
            nn.init.normal_(h[-1].weight, mean=0.0, std=0.02)
            with torch.no_grad():
                h[-1].bias.fill_(float(bias_init))
            return h

        head_bus_in = 2 * hidden_dim + _G_CTX_DIM
        head_gen_in = 2 * hidden_dim + _G_CTX_DIM
        self.head_theta = _make_head(head_bus_in,     1, bias_init=0.0)
        self.head_V     = _make_head(head_bus_in + 1, 1, bias_init=1.0)
        self.head_Pg    = _make_head(head_gen_in,     1, bias_init=0.0)
        self.head_Qg    = _make_head(head_gen_in,     1, bias_init=0.0)
        self.head_feas  = _make_head(hidden_dim + _G_CTX_DIM + STRESS_DIM, 1, bias_init=0.0)

    def forward(self, data: HeteroData) -> HeteroData:
        n_graphs = int(getattr(data, 'num_graphs', 1) or 1)

        pe_dict = self.hodge_pe(data)

        x_dict: Dict[str, Tensor] = {}
        batch_dict: Dict[str, Tensor] = {}
        for nt in self.NODE_TYPES:
            if nt in data.node_types and data[nt].x.size(0) > 0:
                x_in = data[nt].x
                if nt in pe_dict:
                    x_in = torch.cat([x_in, pe_dict[nt]], dim=-1)
                if self.input_norm_enabled and self.input_ln is not None:
                    x_in = self.input_ln[nt](x_in)
                x_dict[nt] = self.input_proj[nt](x_in)
                bb = getattr(data[nt], 'batch', None)
                if bb is None:
                    bb = torch.zeros(x_dict[nt].size(0), dtype=torch.long,
                                     device=x_dict[nt].device)
                    data[nt].batch = bb
                batch_dict[nt] = bb

        edge_index_dict: Dict[EdgeType, Tensor] = {}
        edge_attr_dict:  Dict[EdgeType, Tensor] = {}
        for et in self.EDGE_TYPES:
            if et in data.edge_types and data[et].edge_index.size(1) > 0:
                edge_index_dict[et] = data[et].edge_index
                if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
                    edge_attr_dict[et] = data[et].edge_attr

        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict, batch_dict, edge_attr_dict, n_graphs)

        h_grid_bus, h_grid_gen, h_grid_global = self.fusion(x_dict, data, n_graphs)
        h_grid_bus    = _nan_to_zero(h_grid_bus)
        h_grid_gen    = _nan_to_zero(h_grid_gen)
        h_grid_global = _nan_to_zero(h_grid_global)

        self._predict_and_write(data, h_grid_bus, h_grid_gen, h_grid_global, n_graphs)
        return data

    def _predict_and_write(
        self,
        data: HeteroData,
        h_grid_bus: Tensor,
        h_grid_gen: Tensor,
        h_grid_global: Tensor,
        n_graphs: int,
    ) -> None:
        xg = data['generator'].x
        Pmin = torch.nan_to_num(xg[:, GEN_PMIN_IDX], nan=-1e6)
        Pmax = torch.nan_to_num(xg[:, GEN_PMAX_IDX], nan=+1e6)
        Qmin = torch.nan_to_num(xg[:, GEN_QMIN_IDX], nan=-1e6)
        Qmax = torch.nan_to_num(xg[:, GEN_QMAX_IDX], nan=+1e6)
        Plo = torch.minimum(Pmin, Pmax); Phi = torch.maximum(Pmin, Pmax)
        Qlo = torch.minimum(Qmin, Qmax); Qhi = torch.maximum(Qmin, Qmax)

        bus_batch = data['bus'].batch
        g_per_bus = h_grid_global[bus_batch]

        g_ctx = getattr(data, 'g_ctx', None)
        if g_ctx is None:
            g_ctx = torch.zeros(
                n_graphs, _G_CTX_DIM,
                dtype=h_grid_global.dtype, device=h_grid_global.device,
            )
        else:
            if g_ctx.size(-1) != _G_CTX_DIM:
                raise RuntimeError(
                    f'data.g_ctx last dim {g_ctx.size(-1)} != expected {_G_CTX_DIM}.'
                )
            g_ctx = g_ctx.to(dtype=h_grid_global.dtype, device=h_grid_global.device)
        g_ctx_per_bus = g_ctx[bus_batch]

        h_bus_in = torch.cat([h_grid_bus, g_per_bus, g_ctx_per_bus], dim=-1)
        if 'generator' in data.node_types and data['generator'].x.size(0) > 0:
            gen_batch = data['generator'].batch
            g_per_gen = h_grid_global[gen_batch]
            g_ctx_per_gen = g_ctx[gen_batch]
            h_gen_in = torch.cat([h_grid_gen, g_per_gen, g_ctx_per_gen], dim=-1)
        else:
            h_gen_in = torch.cat(
                [h_grid_gen,
                 torch.zeros(h_grid_gen.size(0),
                             h_grid_global.size(-1) + _G_CTX_DIM,
                             dtype=h_grid_gen.dtype, device=h_grid_gen.device)],
                dim=-1,
            )

        _sig = self.clip_mode == 'sigmoid'
        Pg_raw = _nan_to_zero(self.head_Pg(h_gen_in).squeeze(-1))
        Pg_mid = _nan_to_zero(0.5 * (Plo + Phi))
        Pg = _nan_to_zero(_sigmoid_clip(Pg_raw, Plo, Phi) if _sig
                          else _leaky_clip(Pg_raw + Pg_mid, Plo, Phi, leak=self.leaky_alpha))

        Qg_raw = _nan_to_zero(self.head_Qg(h_gen_in).squeeze(-1))
        Qg_mid = _nan_to_zero(0.5 * (Qlo + Qhi))
        Qg = _nan_to_zero(_sigmoid_clip(Qg_raw, Qlo, Qhi) if _sig
                          else _leaky_clip(Qg_raw + Qg_mid, Qlo, Qhi, leak=self.leaky_alpha))

        bus_x = data['bus'].x
        theta_DC = compute_dc_prior(data, Pg, self._dc_cache)
        theta_DC = math.pi * torch.tanh(theta_DC / math.pi)
        theta_raw = self.theta_res_scale * torch.tanh(
            self.head_theta(h_bus_in).squeeze(-1) / self.theta_res_scale
        )
        theta = _nan_to_zero(_slack_or_mean_anchor(data, theta_DC + theta_raw, n_graphs=n_graphs))

        v_setpoint = getattr(data['bus'], 'v_setpoint', None)
        if v_setpoint is None:
            v_setpoint = torch.zeros(
                h_bus_in.size(0), 1, dtype=h_bus_in.dtype, device=h_bus_in.device,
            )
        else:
            v_setpoint = v_setpoint.to(dtype=h_bus_in.dtype, device=h_bus_in.device)
        V_raw = _nan_to_zero(self.head_V(torch.cat([h_bus_in, v_setpoint], dim=-1)).squeeze(-1))
        Vmin = torch.nan_to_num(bus_x[:, BUS_VMIN_IDX], nan=0.95)
        Vmax = torch.nan_to_num(bus_x[:, BUS_VMAX_IDX], nan=1.05)
        V_lo = torch.minimum(Vmin, Vmax); V_hi = torch.maximum(Vmin, Vmax)
        V = _nan_to_zero(_sigmoid_clip(V_raw, V_lo, V_hi) if _sig
                         else _leaky_clip(V_raw, V_lo, V_hi, leak=self.leaky_alpha), sub_value=1.0)

        data['bus'].pred = torch.stack([theta, V], dim=-1)
        data['generator'].pred = torch.stack([Pg, Qg], dim=-1)

        self._compute_flows(data)
        ac_key = ('bus', 'ac_line', 'bus')
        tr_key = ('bus', 'transformer', 'bus')
        ac_flows = data[ac_key].edge_flow_pred if ac_key in data.edge_types and \
            hasattr(data[ac_key], 'edge_flow_pred') else None
        tr_flows = data[tr_key].edge_flow_pred if tr_key in data.edge_types and \
            hasattr(data[tr_key], 'edge_flow_pred') else None
        stress_actual = _nan_to_zero(
            compute_physics_stress(
                data, V=V, theta=theta, Pg=Pg, Qg=Qg,
                ac_flows=ac_flows, tr_flows=tr_flows, n_graphs=n_graphs,
            ),
            sub_value=1e3,
        )
        feas_input = torch.cat([h_grid_global, g_ctx, stress_actual], dim=-1)
        data.feas_logit = self.head_feas(feas_input).squeeze(-1)

    def _compute_flows(self, data: HeteroData) -> None:
        theta = data['bus'].pred[:, 0]
        V = data['bus'].pred[:, 1]

        if ('bus', 'ac_line', 'bus') in data.edge_types:
            ei = data[('bus', 'ac_line', 'bus')].edge_index
            ea = data[('bus', 'ac_line', 'bus')].edge_attr
            if ei.size(1) > 0:
                Pij, Qij, Pji, Qji = self._pi_model_flows_ac_line(ei, ea, V, theta)
                data[('bus', 'ac_line', 'bus')].edge_flow_pred = torch.stack(
                    [Pij, Qij, Pji, Qji], dim=-1)

        if ('bus', 'transformer', 'bus') in data.edge_types:
            ei = data[('bus', 'transformer', 'bus')].edge_index
            ea = data[('bus', 'transformer', 'bus')].edge_attr
            if ei.size(1) > 0:
                Pij, Qij, Pji, Qji = self._pi_model_flows_transformer(ei, ea, V, theta)
                data[('bus', 'transformer', 'bus')].edge_flow_pred = torch.stack(
                    [Pij, Qij, Pji, Qji], dim=-1)

    @staticmethod
    def _pi_model_flows_ac_line(
        edge_index: Tensor, edge_attr: Tensor, V: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        i = edge_index[0]; j = edge_index[1]
        r = edge_attr[:, AC_LINE_R_IDX]
        x = edge_attr[:, AC_LINE_X_IDX]
        b_fr = edge_attr[:, AC_LINE_BFR_IDX]
        b_to = edge_attr[:, AC_LINE_BTO_IDX]
        denom = r * r + x * x + PI_Z2_EPS
        G = r / denom
        B = -x / denom
        Vi = V[i]; Vj = V[j]
        ti = theta[i]; tj = theta[j]
        cos_dt = torch.cos(ti - tj)
        sin_dt = torch.sin(ti - tj)
        Pij =  Vi * Vi * G          - Vi * Vj * (G * cos_dt + B * sin_dt)
        Qij = -Vi * Vi * (B + b_fr) - Vi * Vj * (G * sin_dt - B * cos_dt)
        Pji =  Vj * Vj * G          - Vi * Vj * (G * cos_dt - B * sin_dt)
        Qji = -Vj * Vj * (B + b_to) + Vi * Vj * (G * sin_dt + B * cos_dt)
        return Pij, Qij, Pji, Qji

    @staticmethod
    def _pi_model_flows_transformer(
        edge_index: Tensor, edge_attr: Tensor, V: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        i = edge_index[0]; j = edge_index[1]
        r = edge_attr[:, TR_R_IDX]
        x = edge_attr[:, TR_X_IDX]
        tap = edge_attr[:, TR_TAP_IDX]
        shift = edge_attr[:, TR_SHIFT_IDX]
        b_fr = edge_attr[:, TR_BFR_IDX]
        b_to = edge_attr[:, TR_BTO_IDX]
        tap = torch.where(tap.abs() < PI_TAP_EPS, torch.ones_like(tap), tap)
        denom = r * r + x * x + PI_Z2_EPS
        G = r / denom
        B = -x / denom
        Vi = V[i]; Vj = V[j]
        ti = theta[i]; tj = theta[j]
        Vi_t = Vi / tap
        ti_t = ti - shift
        cos_dt_ft = torch.cos(ti_t - tj)
        sin_dt_ft = torch.sin(ti_t - tj)
        cos_dt_tf = torch.cos(tj - ti_t)
        sin_dt_tf = torch.sin(tj - ti_t)
        Pij =  Vi_t * Vi_t * G          - Vi_t * Vj * (G * cos_dt_ft + B * sin_dt_ft)
        Qij = -Vi_t * Vi_t * (B + b_fr) - Vi_t * Vj * (G * sin_dt_ft - B * cos_dt_ft)
        Pji =  Vj   * Vj   * G          - Vi_t * Vj * (G * cos_dt_tf + B * sin_dt_tf)
        Qji = -Vj   * Vj   * (B + b_to) - Vi_t * Vj * (G * sin_dt_tf - B * cos_dt_tf)
        return Pij, Qij, Pji, Qji
