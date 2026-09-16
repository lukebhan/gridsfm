"""GridSFM AC-OPF inference package: the backbone, its schema and its data path."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch_geometric.data import HeteroData

from . import schema
from .checkpoint import load_from_hf, load_model
from .data import (
    batch_data_list,
    drop_offline_rows,
    load_opfdata,
    load_pyg_json,
    prepare_for_inference,
)
from .model import GridTransformerBackbone
from .schema import AC_LINE_KEY, TRANSFORMER_KEY

__version__ = "1.1.0"


@torch.no_grad()
def predict(
    model: GridTransformerBackbone,
    scenario: Union[str, Path, HeteroData],
    fmt: str = "auto",
) -> Dict[str, Any]:
    """Run single-graph inference and return CPU tensors + scalars.

    Args:
      model: a loaded `GridTransformerBackbone`.
      scenario: a `.pyg.json` / `.json` path, or a `HeteroData` already
        in the package's schema. The latter is `.clone()`'d before any
        in-place mutation by `prepare_for_inference`.
      fmt: `"auto"` (default), `"pyg"`, or `"opfdata"`. Auto detection
        chooses `pyg` for `*.pyg.json`, otherwise `opfdata`.

    Returns a dict with `theta`, `V`, `Pg`, `Qg` (per-node CPU tensors),
    branch flows `Pij` / `Qij` / `Pji` / `Qji` concatenated across edge
    families (`flow_edge_types` + `flow_edge_counts` document the
    layout), and feasibility outputs (`feas`, `feas_logit`).
    """
    if isinstance(scenario, HeteroData):
        data = scenario.clone()
    else:
        path = Path(scenario)
        if fmt == "auto":
            if path.suffixes[-2:] == [".pyg", ".json"]:
                fmt = "pyg"
            elif path.suffix == ".json":
                fmt = "opfdata"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for {path}. "
                    f"Pass fmt='pyg' or fmt='opfdata' explicitly."
                )
        if fmt == "pyg":
            data = load_pyg_json(path)
        elif fmt == "opfdata":
            data = load_opfdata(path)
        else:
            raise ValueError(f"Unknown fmt={fmt!r}. Expected 'auto', 'pyg', or 'opfdata'.")
    data = prepare_for_inference(data)

    device = next(model.parameters()).device
    data = data.to(device)
    if int(getattr(data, "num_graphs", 1) or 1) > 1:
        raise ValueError(
            "predict() handles a single scenario; for multi-graph batches use "
            "batch_data_list(prepared) and call model(batch) directly."
        )

    # model.forward attaches default per-node-type `batch` and reads
    # `num_graphs` itself; no need to pre-populate them here.
    out = model(data)

    bus_pred = out["bus"].pred
    gen_pred = out["generator"].pred
    flow_tensors: list[torch.Tensor] = []
    flow_edge_types: list[str] = []
    flow_edge_counts: list[int] = []
    for k in (AC_LINE_KEY, TRANSFORMER_KEY):
        if k in out.edge_types and hasattr(out[k], "edge_flow_pred"):
            e = out[k].edge_flow_pred
            flow_tensors.append(e)
            flow_edge_types.append(k[1])
            flow_edge_counts.append(int(e.size(0)))
    if flow_tensors:
        flows = torch.cat(flow_tensors, dim=0)
    else:
        flows = torch.zeros(0, 4, device=device, dtype=bus_pred.dtype)

    feas_logit = float(out.feas_logit.item())
    return {
        "theta":      bus_pred[:, 0].cpu(),
        "V":          bus_pred[:, 1].cpu(),
        "Pg":         gen_pred[:, 0].cpu(),
        "Qg":         gen_pred[:, 1].cpu(),
        "Pij":        flows[:, 0].cpu(),
        "Qij":        flows[:, 1].cpu(),
        "Pji":        flows[:, 2].cpu(),
        "Qji":        flows[:, 3].cpu(),
        "flow_edge_types":  flow_edge_types,
        "flow_edge_counts": flow_edge_counts,
        "feas":       float(torch.sigmoid(out.feas_logit).item()),
        "feas_logit": feas_logit,
    }


__all__ = [
    # Inference
    "GridTransformerBackbone",
    "batch_data_list",
    "drop_offline_rows",
    "load_model",
    "load_from_hf",
    "load_pyg_json",
    "load_opfdata",
    "prepare_for_inference",
    "predict",
    "schema",
    # Edge-type schema keys (used by callers wiring batched flows back to
    # per-edge-type IDs; see `predict()` body for the canonical pattern).
    "AC_LINE_KEY",
    "TRANSFORMER_KEY",
    "__version__",
]
