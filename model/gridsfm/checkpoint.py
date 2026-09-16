"""Checkpoint loading for the minimal public-release format."""
from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download

from .model import GridTransformerBackbone

_logger = logging.getLogger(__name__)


def _adapt_v1_0_w_global(name: str, v_ckpt: torch.Tensor, v_model: torch.Tensor) -> Optional[torch.Tensor]:
    """Adapt a v1.0 `fusion.W_global.weight` to v1.1 shape, or return `None`.

    v1.0 stored 4 type-mean d-wide columns contiguously; v1.1 block-
    interleaves (mean, max) per type. The adapter permutes v1.0 means
    into v1.1's even (mean) slots and zeroes the odd (max) slots.
    Returns `None` for any tensor that does not match the v1.0 W_global
    signature, so the caller can hard-fail on unexpected shape mismatch
    (silent zero-pad / crop would mask architecture corruption).
    """
    if name != "fusion.W_global.weight" or v_ckpt.dim() != 2:
        return None
    d_out, c_in = v_ckpt.shape
    m_out, m_in = v_model.shape
    if not (d_out == m_out and c_in * 2 == m_in and c_in % 4 == 0):
        return None
    target = torch.zeros_like(v_model)
    d = c_in // 4
    for type_idx in range(4):
        src_lo = type_idx * d
        dst_lo = type_idx * 2 * d
        target[:, dst_lo:dst_lo + d] = v_ckpt[:, src_lo:src_lo + d]
    return target


def _hash_state_dict(state_dict: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        t = state_dict[key]
        if torch.is_tensor(t):
            t_cpu = t.detach().cpu().contiguous()
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.to(torch.float32)
            arr = t_cpu.numpy()
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(arr.tobytes())
        else:
            h.update(str(t).encode("utf-8"))
    return h.hexdigest()


def _warn_v1_0_deprecated(stacklevel: int) -> None:
    warnings.warn(
        "Loading a v1.0-shape `gridsfm_open` checkpoint into the v1.1 "
        "backbone. The v1.0 release is deprecated and will be removed in "
        "a future minor release; download v1.1 from HuggingFace: "
        "`hf download microsoft/GridSFM_Open gridsfm_open_v1.1.pt "
        "--local-dir checkpoints`.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def load_model(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    *,
    stacklevel: int = 2,
) -> GridTransformerBackbone:
    """Load a GridSFM release checkpoint and return a `GridTransformerBackbone`.

    Verifies SHA-256 against `metadata.hash`, builds the model from
    `metadata.arch` (defaulting when absent), and cross-version-adapts
    v1.0 → v1.1 `W_global` if the ckpt is the older release. Any
    shape mismatch outside the v1.0 W_global signature raises
    `RuntimeError` rather than silently zero-padding. Model is moved
    to `device` and put in `eval()` mode.
    """
    blob = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    if not (isinstance(blob, dict) and "state_dict" in blob and "metadata" in blob):
        raise ValueError(
            f"{ckpt_path}: not a minimal release checkpoint. "
            f"Expected top-level keys 'state_dict' and 'metadata'. "
            f"Training-format checkpoints (with 'model_state_dict', "
            f"'optimizer_state_dict', etc.) are not loadable here; "
            f"convert them via the export script in the training repo."
        )

    state_dict = blob["state_dict"]
    meta = blob["metadata"]
    expected_hash = meta.get("hash")
    if not expected_hash:
        raise ValueError(
            f"{ckpt_path}: metadata.hash missing. Re-export with the "
            f"release-checkpoint script to embed an integrity hash."
        )

    actual_hash = _hash_state_dict(state_dict)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{ckpt_path}: state_dict hash mismatch.\n"
            f"  expected (metadata.hash): {expected_hash}\n"
            f"  computed:                 {actual_hash}\n"
            f"  Likely causes: partial/corrupted download, in-place "
            f"modification of the file, or the embedded hash being out "
            f"of sync with the weights."
        )

    arch = meta.get("arch") or {}
    model = GridTransformerBackbone(**arch)

    model_sd = model.state_dict()
    adapted, mismatched, skipped = {}, [], []
    v1_0_load = False
    for k, v_ckpt in state_dict.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        v_model = model_sd[k]
        if tuple(v_ckpt.shape) == tuple(v_model.shape):
            adapted[k] = v_ckpt
            continue
        target = _adapt_v1_0_w_global(k, v_ckpt, v_model)
        if target is None:
            raise RuntimeError(
                f"{ckpt_path}: shape mismatch for {k!r}: ckpt "
                f"{tuple(v_ckpt.shape)} vs model {tuple(v_model.shape)}, "
                f"and no cross-version adapter applies. The checkpoint's "
                f"architecture does not match the default backbone; "
                f"re-export with `metadata.arch` embedded so the model "
                f"can be constructed from its true hyperparameters."
            )
        v1_0_load = True
        adapted[k] = target
        mismatched.append((k, tuple(v_ckpt.shape), tuple(v_model.shape)))

    if v1_0_load:
        _warn_v1_0_deprecated(stacklevel=stacklevel + 1)
    for name, csh, msh in mismatched:
        _logger.warning("shape-adapted %s: ckpt %s -> model %s (cross-version load)",
                        name, csh, msh)
    if skipped:
        raise RuntimeError(
            f"{ckpt_path}: {len(skipped)} ckpt key(s) not present in the "
            f"v1.1 backbone: {skipped}. A subset-overlap load would mask "
            f"an architecture mismatch; refusing rather than silently "
            f"discarding ckpt weights."
        )
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    # `adapted` is built only from keys present in `model_sd`, so
    # load_state_dict should never report unexpected keys. Assert to
    # surface any future refactor that breaks this invariant.
    assert not unexpected, f"unexpected keys reached load_state_dict: {unexpected}"
    if missing:
        raise RuntimeError(
            f"{ckpt_path}: {len(missing)} model parameter(s) not present "
            f"in the ckpt: {missing}. These would remain at Kaiming init; "
            f"refusing rather than silently shipping random weights."
        )
    model.to(device).eval()
    return model


def load_from_hf(
    repo_id: str,
    filename: str = "gridsfm_open_v1.1.pt",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> GridTransformerBackbone:
    """Download a release checkpoint from HuggingFace and load it via `load_model`.

    Args:
      repo_id: HF repo (e.g. ``"microsoft/GridSFM_Open"``).
      filename: ckpt filename within the repo.
      revision: optional pinned revision/tag/branch.
      cache_dir: optional HF cache directory.
      **kwargs: forwarded to `load_model` (e.g. `device=`).
    """
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return load_model(ckpt_path, stacklevel=3, **kwargs)
