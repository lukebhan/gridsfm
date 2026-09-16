# `model/` — vendored from Microsoft GridSFM

**Every line of code in this directory comes from
[github.com/microsoft/gridSFM](https://github.com/microsoft/gridSFM)** ("GridSFM: Small
Foundation Models for the Power Grid", Microsoft Research). It is not our work. It is
copied here, unmodified apart from the deletions noted below, so that this repository runs
standalone without a separate `gridsfm` install.

**License: MIT**, as upstream. Copyright belongs to Microsoft Corporation and the GridSFM
authors.

Report bugs, request features and take updates **upstream** — not here. This copy is
pinned for reproducibility of the fine-tuning results in this repository, and will drift
from upstream over time.

| | |
|---|---|
| Source | https://github.com/microsoft/gridSFM |
| Released backbones | https://huggingface.co/microsoft/GridSFM_Open |
| US power-grid dataset | https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid |
| Package version | `gridsfm.__version__ == "1.1.0"` |

## What this copy contains

Upstream's `model/` package, reduced to the inference path that `finetune_model/` actually
imports — `checkpoint.load_model`, `data.*`, `model.GridTransformerBackbone` and
`schema.*`, plus everything those pull in transitively.

```
gridsfm/
├── __init__.py           package API + predict()          [MODIFIED, see below]
├── model.py              GridTransformerBackbone
├── blocks.py             GridBlock, attention, fusion
├── checkpoint.py         load_model / load_from_hf
├── data.py               load_pyg_json, prepare_for_inference, batch_data_list
├── schema.py             column indices and edge-type keys
├── cycle_basis.py        cycle-basis construction
├── hodge_pe.py           Hodge positional encoding
├── pe_features.py        positional-encoding features
├── signed_incidence.py   SignedIncidenceConv
├── dc_prior.py           DC power-flow prior
└── stress_features.py    physics stress features        [comment-only change]
```

### Removed from the upstream package

These are upstream features that nothing in this repository imports. They were deleted to
keep the vendored copy to what is actually exercised here — **use upstream if you want
them**:

| removed | what it was |
|---|---|
| `eval.py` | `eval_pass` evaluation loop |
| `loss.py` | `compute_loss`, upstream's training objective |
| `finetune_opfdata.py` | `finetune_opfdata`, the OPFData fine-tuning entry point |
| `opfdata_train.py` | `OPFDataAdapterDataset` |
| `synthetic.py` | `SyntheticMixedDataset` |
| `hf_util/` | `GridSFM_PG_Loader`, the HuggingFace loader for `microsoft/GridSFM_US_power_grid` |

This repository trains with its own objective and loop in
[`finetune_model/src/`](../finetune_model/src/), which is why upstream's training surface
is not needed.

### Modifications to the files that remain

The trimming touched two files; no other file was edited here. To confirm what this copy
carries relative to a given upstream revision, diff `gridsfm/` against
`model/gridsfm/` in a clone of the upstream repository.

- **`__init__.py`** — the imports and `__all__` entries for the deleted modules were
  removed, and `drop_offline_rows` was added to `__all__`. The consequence is that
  `compute_loss`, `eval_pass`, `finetune_opfdata`, `OPFDataAdapterDataset` and
  `SyntheticMixedDataset` are **not importable from this copy**. Inference,
  `GridTransformerBackbone` and `predict()` are unchanged.
- **`stress_features.py`** — two comments that pointed at the deleted `loss.py` were
  reworded. No code change.

## Citation

If you use this model, cite the upstream work rather than this repository:

> Yang et al. (2026). *GridSFM: A Foundation Model for AC Optimal Power Flow.* Microsoft
> Research.

The canonical BibTeX and the companion power-grid pipeline paper (Britto et al., 2026) are
in the [upstream README](https://github.com/microsoft/gridSFM).
