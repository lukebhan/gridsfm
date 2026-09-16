"""Optimizer / scheduler construction from config, plus parameter freezing.

Kept separate from train.py so each factory is small enough to read and to test
without a GPU or a dataset.
"""
from __future__ import annotations
import torch


def build_optimizer(model, T: dict):
    """adam | adamw | sgd, with the config's betas/eps/weight_decay/momentum."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters -- check model.freeze")
    name = str(T["optimizer"]).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=T["lr"], betas=tuple(T["betas"]),
                                eps=T["eps"], weight_decay=T["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=T["lr"], betas=tuple(T["betas"]),
                                 eps=T["eps"], weight_decay=T["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=T["lr"], momentum=T["momentum"],
                               weight_decay=T["weight_decay"], nesterov=T["nesterov"])
    raise ValueError(f"train.optimizer must be adam|adamw|sgd, got {name!r}")


def build_scheduler(opt, T: dict):
    """Returns a per-EPOCH scheduler, or None.

    Warmup is applied by `lr_at_epoch` rather than by chaining a torch scheduler: the
    cosine schedule must span the POST-warmup epochs, and LambdaLR-style chaining makes
    that relationship hard to read off the config.
    """
    name = str(T["scheduler"]).lower()
    # COSINE HORIZON. By default the cosine spans the whole epoch CAP, which on a run
    # that early-stops far short of it means the LR never actually anneals: with
    # epochs=400 and a stop near epoch 50, lr moved only 1e-4 -> 9.99e-5. `lr_horizon`
    # decouples the two, so the schedule can complete inside the run's real lifetime
    # without also shortening the run.
    n = int(T.get("lr_horizon") or 0) or (int(T["epochs"]) - int(T["warmup_epochs"]))
    n = max(1, n)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n, eta_min=T["lr"] * T["lr_min_frac"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(T["step_size"]), gamma=float(T["step_gamma"]))
    if name == "none":
        return None
    raise ValueError(f"train.scheduler must be cosine|step|none, got {name!r}")


def warmup_lr(T: dict, epoch: int) -> float | None:
    """LR for a warmup epoch (1-indexed), or None once warmup is over.

    Linear ramp from lr*warmup_start_frac to lr across warmup_epochs.
    """
    w = int(T["warmup_epochs"])
    if w <= 0 or epoch > w:
        return None
    f0 = float(T["warmup_start_frac"])
    frac = f0 + (1.0 - f0) * (epoch - 1) / max(w, 1)
    return float(T["lr"]) * frac


def apply_freeze(model, mode: str) -> tuple[int, int]:
    """`trunk` freezes everything except the prediction heads. Returns
    (trainable params, total params) so the run log can state what is being trained."""
    mode = str(mode or "none").lower()
    total = sum(p.numel() for p in model.parameters())
    if mode == "none":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "trunk":
        for p in model.parameters():
            p.requires_grad_(False)
        hit = 0
        for name, mod in model.named_modules():
            # the surgery model's output heads; matched by name so this survives
            # refactors of the trunk
            if name.endswith("head") or ".head" in name or "pred" in name.lower():
                for p in mod.parameters(recurse=True):
                    p.requires_grad_(True)
                    hit += 1
        if not hit:
            raise ValueError("model.freeze=trunk found no head parameters to train")
    else:
        raise ValueError(f"model.freeze must be none|trunk, got {mode!r}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total
