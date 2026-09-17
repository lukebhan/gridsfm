"""The fine-tuning loop.

Surgical fine-tune of the GridSFM on/off model into the control weights that
warm-start the AC-OPF solve. The released base checkpoint is already the on/off
architecture (ON/OFF generator-availability column + DC/ER positional-encoding
removal baked in), loaded via `load_model`; the whole backbone is then trained
supervised on GT AC-OPF dispatch with the max-aligned control loss (see loss.py).

Split handling, deliberately dumb: train keys and test keys come from two separate
manifest lists built from two separate files. This module never shuffles them
together, never re-derives a split, and never touches test until the final
evaluation. The previous harness re-split internally, which would have silently
un-held-out the test set.

Three splits, three distinct jobs: TRAIN fits the weights, VAL chooses which epoch
ships, TEST is read exactly once at the very end. Test is never a selection input --
`train.select_on` accepts val | final | train_loss and deliberately not `test`.
"""
from __future__ import annotations
import json, os, random, time, subprocess
import numpy as np
import torch

from gridsfm.checkpoint import load_model
from store import CaseStore, predict
from loss import control_loss
from optim import build_optimizer, build_scheduler, warmup_lr, apply_freeze
from evaluate import eval_pred, collect_diag
from history import History
from plots import plot_training, plot_test_diagnostics


def set_runtime(cfg: dict):
    """TF32 is a throughput knob only: the loss and every reported metric are computed
    in float64 regardless of what the backbone matmuls use."""
    on = bool(cfg["runtime"].get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def set_seeds(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def preflight(cfg: dict):
    """Refuse to start rather than OOM a co-tenant's job on a shared GPU."""
    need = float(cfg["runtime"]["min_free_vram_gb"])
    if cfg["runtime"]["device"].startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 2**30
        print(f"GPU {torch.cuda.get_device_name(0)}: {free_gb:.1f} GB free of "
              f"{total/2**30:.1f} GB", flush=True)
        if free_gb < need:
            raise RuntimeError(
                f"only {free_gb:.1f} GB free, need {need:.1f} (runtime.min_free_vram_gb). "
                "Another job is probably using this GPU -- refusing to start.")


def run(cfg: dict, man: dict) -> dict:
    mode = str(cfg["loss"].get("mode", "control")).lower()
    if mode == "elastic":
        from train_elastic import run_elastic
        return run_elastic(cfg, man)
    dev = cfg["runtime"]["device"]
    T, L = cfg["train"], cfg["loss"]
    D = cfg["_derived"]
    outdir = cfg["_meta"]["results_dir"]
    ckdir = cfg["_meta"]["ckpt_dir"]
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckdir, exist_ok=True)
    ckpt_out = os.path.join(ckdir, "control_model.pt")

    set_seeds(T["seed"], T["deterministic"])
    set_runtime(cfg)
    preflight(cfg)

    # ---- the three splits, exactly as given ----
    from subset import train_subset, subset_label
    all_train = list(man["splits"]["train"]["keys"])
    train_modes = man["splits"]["train"].get("modes") or [None] * len(all_train)
    n_sub = int(cfg["data"].get("train_subset", 0))
    train_keys = train_subset(all_train, train_modes, n_sub,
                              int(cfg["data"].get("train_subset_seed", 0)))
    val_keys = list(man["splits"]["val"]["keys"])
    test_keys = list(man["splits"]["test"]["keys"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = set(man["splits"][a]["keys"]) & set(man["splits"][b]["keys"])
        assert not ov, f"{len(ov)} cases in both {a} and {b}"
    if n_sub and len(train_keys) < len(all_train):
        import collections as _c
        mix = _c.Counter(m for k, m in zip(all_train, train_modes) if k in set(train_keys))
        print(f"run_label={cfg['run_label']}  TRAIN SUBSET {len(train_keys)}/{len(all_train)} "
              f"(stratified, nested): {dict(sorted(mix.items()))}", flush=True)
    else:
        print(f"run_label={cfg['run_label']}  training on all {len(train_keys)} cases "
              f"from split '{tsplit}'", flush=True)
    print(f"train={len(train_keys)}  val={len(val_keys)} (epoch selection, NOT subsetted)  "
          f"test={len(test_keys)} (held out, read once at the end)", flush=True)
    print(f"derived: topk_pg={D['topk_pg']} ({D['topk_pg_pct']}% of {D['mean_scored_gens']} gens)  "
          f"topk_v={D['topk_v']} ({D['topk_v_pct']}% of {D['mean_scored_vctrl']} V-ctrl)  "
          f"lru={D['lru_cases']} cases", flush=True)

    store = CaseStore(cfg["cache"]["dir"], D["lru_cases"], dev)
    # model.from_scratch: randomly initialised backbone instead of the released
    # checkpoint -- the no-pretraining ablation. Same branch as train_elastic.py, see
    # config/defaults.yaml. With loss.mode=control the anchor is the ONLY term, so this
    # trains the architecture on GT controls alone, no power-flow closure in the loop.
    if bool(cfg["model"].get("from_scratch", False)):
        from gridsfm.model import GridTransformerBackbone
        arch = dict(hidden_dim=int(cfg["model"].get("hidden_dim", 64)),
                    num_blocks=int(cfg["model"].get("num_blocks", 4)))
        model = GridTransformerBackbone(**arch).to(dev)
        _np = sum(q.numel() for q in model.parameters())
        print(f"  model FROM SCRATCH (no pretraining): hidden_dim={arch['hidden_dim']} "
              f"num_blocks={arch['num_blocks']}  {_np/1e6:.2f}M params", flush=True)
    else:
        model = load_model(cfg["base_checkpoint"], dev)
    model.clip_mode = cfg["model"]["clip_mode"]
    n_train_p, n_tot_p = apply_freeze(model, cfg["model"]["freeze"])
    print(f"model: clip_mode={cfg['model']['clip_mode']}  freeze={cfg['model']['freeze']}  "
          f"trainable {n_train_p:,}/{n_tot_p:,} params "
          f"({100*n_train_p/max(n_tot_p,1):.1f}%)", flush=True)
    opt = build_optimizer(model, T)
    sched = build_scheduler(opt, T)
    print(f"optim: {T['optimizer']} lr={T['lr']} betas={tuple(T['betas'])} "
          f"wd={T['weight_decay']} | sched={T['scheduler']} "
          f"warmup={T['warmup_epochs']}ep floor={T['lr_min_frac']}x "
          f"| clip={T['clip_grad']} (L{T['clip_norm_type']:g})", flush=True)

    H = History(outdir)
    def val_score(pg, v):
        """Combined selection measure; weights from eval.val_score_*_weight. The
        default 1x/1000x puts 0.001 p.u. of voltage error on par with 1% of Pg."""
        return cfg["eval"]["val_score_pg_weight"] * pg + cfg["eval"]["val_score_v_weight"] * v

    # A micro-batch larger than the training set makes the step loop
    # `range(0, n - BATCH + 1, BATCH)` EMPTY -- the run would complete every epoch without
    # a single optimizer step and report a perfectly flat loss. Clamp instead.
    BATCH_CFG = int(T["batch"])
    BATCH = max(1, min(BATCH_CFG, len(train_keys)))
    ACC = int(T["accum"])
    if BATCH_CFG > len(train_keys):
        print(f"  train.batch={BATCH_CFG} exceeds the {len(train_keys)}-case training set; "
              f"using batch={BATCH} so the step loop is not empty", flush=True)
    gstep = 0
    best_train = [float("inf")]
    best_val = [float("inf")]
    best_epoch = 0
    best_val_epoch = 0
    t_run = time.time()

    # baseline: the un-fine-tuned surgery model == the `base` warm-start arm
    b_pg, b_v = eval_pred(model, store, val_keys, "val@ep0", bs=cfg["eval"]["batch"])
    print(f"[baseline] untrained-surgery val Pg={b_pg:.2f}% V={b_v:.4f}pu "
          f"(this is the `base` warm-start arm)", flush=True)

    stale = 0                                   # epochs since the last val improvement
    for ep in range(1, T["epochs"] + 1):
        t_ep = time.time()
        wlr = warmup_lr(T, ep)
        if wlr is not None:                     # linear warmup overrides the schedule
            for g in opt.param_groups:
                g["lr"] = wlr
        torch.cuda.reset_peak_memory_stats() if dev.startswith("cuda") else None
        random.shuffle(train_keys)
        model.train(); opt.zero_grad()
        acc_l = acc_p = acc_v = 0.0
        nb = 0; mb = 0
        gns: list[float] = []
        t_data = t_fb = 0.0
        n_clipped = 0

        for i in range(0, len(train_keys) - BATCH + 1, BATCH):
            t0 = time.time()
            S = store.batch(train_keys[i:i + BATCH])
            t_data += time.time() - t0

            t0 = time.time()
            Pg, V = predict(model, S)
            loss, L_pg, L_v = control_loss(
                Pg, S["pg_gt"], V, S["vm_gt"], S["av"], S["vctrl"], L,
                D["topk_pg"], D["topk_v"],
                gen_seg=S["gen_seg"], bus_seg=S["bus_seg"], nseg=S["nseg"])
            scaled = loss / ACC
            if torch.isfinite(scaled):
                scaled.backward()
            with torch.no_grad():                  # detached: logged, not differentiated
                l_, p_, v_ = float(loss.detach()), float(L_pg.detach()), float(L_v.detach())
                acc_l += l_; acc_p += p_; acc_v += v_
            nb += 1; mb += 1

            if mb % ACC == 0:
                if T["clip_grad"]:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), T["clip_grad"],
                        norm_type=float(T["clip_norm_type"]))
                else:                            # clipping off: still measure the norm
                    gn = torch.nn.utils.get_total_norm(
                        [p.grad for p in model.parameters() if p.grad is not None],
                        norm_type=float(T["clip_norm_type"]))
                gnf = float(gn)
                if np.isfinite(gnf):
                    opt.step(); gns.append(gnf)
                    if T["clip_grad"] and gnf > T["clip_grad"]:
                        n_clipped += 1
                opt.zero_grad(); gstep += 1
                if gstep % max(1, int(cfg["logging"]["step_log_every"])) == 0:
                    H.add_step(step=gstep, epoch=ep, loss=round(l_, 6),
                               loss_pg=round(p_, 6), loss_v=round(v_, 6),
                               gnorm=round(gnf, 4), lr=opt.param_groups[0]["lr"],
                               wall_sec=round(time.time() - t_run, 2))
            t_fb += time.time() - t0

        lr_now = opt.param_groups[0]["lr"]
        if sched is not None and warmup_lr(T, ep + 1) is None:
            sched.step()                         # cosine spans the POST-warmup epochs

        t0 = time.time()
        v_pg, v_v = eval_pred(model, store, val_keys, f"val@ep{ep}", bs=cfg["eval"]["batch"])
        vsc = val_score(v_pg, v_v)
        t_eval = time.time() - t0

        ep_loss = acc_l / max(nb, 1)
        if ep_loss < best_train[0]:
            best_train[0] = ep_loss; best_epoch = ep
            torch.save(model.state_dict(), os.path.join(ckdir, "best_train_loss.pt"))
        # `improved` must be read BEFORE best_val is updated, and must carry min_delta:
        # comparing against the already-updated best makes the staleness counter reset on
        # a WORSE epoch, which silently disables early stopping.
        improved = vsc < best_val[0] - float(T["early_stop_min_delta"])
        is_best = vsc < best_val[0]
        if is_best:
            best_val[0] = vsc; best_val_epoch = ep
            torch.save(model.state_dict(), os.path.join(ckdir, "best_val.pt"))
            print(f"  BEST val (score={vsc:.2f}) saved", flush=True)
        torch.save(model.state_dict(), os.path.join(ckdir, "last.pt"))
        se = int(cfg["logging"]["save_every_epochs"])
        if se and ep % se == 0:
            torch.save(model.state_dict(), os.path.join(ckdir, f"epoch_{ep:03d}.pt"))

        H.add_epoch(epoch=ep, loss=ep_loss, loss_pg=acc_p / max(nb, 1),
                    loss_v=acc_v / max(nb, 1), val_pg=v_pg, val_v=v_v,
                    val_score=vsc, lr=lr_now,
                    gnorm_mean=float(np.mean(gns)) if gns else 0.0,
                    gnorm_max=float(np.max(gns)) if gns else 0.0,
                    gnorm_p99=float(np.percentile(gns, 99)) if gns else 0.0,
                    clip_frac=n_clipped / max(len(gns), 1),
                    epoch_sec=time.time() - t_ep, data_sec=t_data,
                    fwd_bwd_sec=t_fb, eval_sec=t_eval,
                    gpu_peak_gb=(torch.cuda.max_memory_allocated() / 2**30
                                 if dev.startswith("cuda") else 0.0),
                    cache_hit_frac=store.hits / max(store.hits + store.misses, 1),
                    is_best=int(is_best))
        if ep % max(1, int(cfg["logging"]["plot_every_epochs"])) == 0 or ep == T["epochs"]:
            plot_training(H.epoch, outdir, os.path.join(outdir, "history_step.csv"))
        print(f"  ep{ep} loss={ep_loss:.2f} gnorm(mean/max)="
              f"{H.epoch['gnorm_mean'][-1]:.1f}/{H.epoch['gnorm_max'][-1]:.0f} "
              f"clip={100*H.epoch['clip_frac'][-1]:.0f}% {H.epoch['epoch_sec'][-1]:.0f}s",
              flush=True)

        # early stopping on the val score (the selection measure), if enabled
        pat = int(T["early_stop_patience"])
        if pat:
            stale = 0 if improved else stale + 1
            if stale >= pat:
                print(f"  early stop: {pat} epochs without val improvement "
                      f"(best ep{best_val_epoch}, score {best_val[0]:.2f})", flush=True)
                break

    epochs_run = len(H.epoch["epoch"])
    train_hours = (time.time() - t_run) / 3600

    # ---- ship the selected weights, then TEST exactly once ----
    pick, ship_epoch = {"val": ("best_val.pt", best_val_epoch),
                        "final": ("last.pt", epochs_run),
                        "train_loss": ("best_train_loss.pt", best_epoch)}[T["select_on"]]
    model.load_state_dict(torch.load(os.path.join(ckdir, pick), map_location=dev))
    # SELF-DESCRIBING for a non-default architecture. A bare state_dict forces the eval
    # harness to infer (hidden_dim, num_blocks) from tensor shapes, that inference was
    # wrong once and loaded a 64x4 checkpoint into the released 128x8 backbone. Recording
    # the arch alongside removes the guess. Written as a sidecar rather than changing the
    # blob's shape, because evaluate_model loads control_model.pt as a plain state_dict.
    torch.save(model.state_dict(), ckpt_out)
    if bool(cfg["model"].get("from_scratch", False)):
        import json as _json
        with open(os.path.join(ckdir, "arch.json"), "w") as _fh:
            _json.dump(dict(hidden_dim=int(cfg["model"].get("hidden_dim", 64)),
                            num_blocks=int(cfg["model"].get("num_blocks", 4)),
                            from_scratch=True), _fh)
    print(f"\nselect_on={T['select_on']} -> {pick} (epoch {ship_epoch}) -> {ckpt_out}", flush=True)

    test_pg, test_v = eval_pred(model, store, test_keys, "TEST", bs=cfg["eval"]["test_batch"])
    print("collecting TEST diagnostics...", flush=True)
    d = collect_diag(model, store, test_keys)
    d.update(test_pg_pct=test_pg, test_v_pu=test_v, n_train=len(train_keys),
             n_val=len(val_keys), n_test=len(test_keys),
             ship_epoch=ship_epoch, select_on=T["select_on"],
             val_pg_pct=H.epoch["val_pg"][ship_epoch - 1],
             val_v_pu=H.epoch["val_v"][ship_epoch - 1],
             baseline_val_pg_pct=b_pg, baseline_val_v_pu=b_v,
             train_hours=train_hours,
             n_gen_scored_per_case=len(d["pg_gt"]) // max(len(test_keys), 1),
             n_vctrl_scored_per_case=len(d["v_abs_err"]) // max(len(test_keys), 1))

    p1 = plot_training(H.epoch, outdir, os.path.join(outdir, "history_step.csv"))
    p2 = plot_test_diagnostics(d, outdir)

    summary = {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
    summary.update(run_label=cfg["run_label"], grid_id=cfg["grid_id"],
                   n_train_available=len(all_train), train_subset=n_sub,
                   topk_pg=D["topk_pg"], topk_v=D["topk_v"],
                   epochs_configured=T["epochs"], epochs_run=epochs_run,
                   batch=BATCH, accum=ACC, lr=T["lr"], seed=T["seed"],
                   optimizer=T["optimizer"], scheduler=T["scheduler"],
                   weight_decay=T["weight_decay"], warmup_epochs=T["warmup_epochs"],
                   clip_grad=T["clip_grad"], freeze=cfg["model"]["freeze"],
                   trainable_params=n_train_p, total_params=n_tot_p,
                   base_checkpoint=cfg["base_checkpoint"],
                   config_sha256=cfg["_meta"]["config_sha256"], git_sha=_git_sha())
    with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(outdir, "test_arrays.npz"),
                        pg_pred=d["pg_pred"].astype(np.float32),
                        pg_gt=d["pg_gt"].astype(np.float32),
                        pg_abs_err=d["pg_abs_err"].astype(np.float32),
                        v_abs_err=d["v_abs_err"].astype(np.float32),
                        per_case_worst_pg=d["per_case_worst_pg"].astype(np.float32),
                        per_case_pg_pct=d["per_case_pg_pct"].astype(np.float32))
    tm = H.write_timing(dict(prep_cache_seconds=man["stats"]["build_seconds"],
                             cache_gb=man["stats"]["total_bytes"] / 2**30,
                             cache_hit_frac=store.hits / max(store.hits + store.misses, 1)))
    with open(ckpt_out.replace(".pt", ".json"), "w") as f:      # checkpoint sidecar
        json.dump(summary, f, indent=2)

    print(f"\n=== DONE in {train_hours:.2f}h | TEST Pg={test_pg:.2f}%  V={test_v:.4f}pu ===")
    print(f"  val @ship epoch {ship_epoch}: Pg={d['val_pg_pct']:.2f}%  V={d['val_v_pu']:.4f}pu")
    print(f"  baseline (no fine-tune, val):     Pg={b_pg:.2f}%  V={b_v:.4f}pu")
    print(f"  Pg err p50/p90/p99/max: {d['pg_err_p50']:.4f}/{d['pg_err_p90']:.4f}/"
          f"{d['pg_err_p99']:.4f}/{d['pg_err_max']:.4f} p.u.")
    print(f"  weights  -> {ckpt_out}")
    print(f"  figures  -> {p1}\n              {p2}")
    print(f"  history  -> {outdir}/history_epoch.csv, history_step.csv")
    print(f"  timing   -> {outdir}/timing.json  ({tm['fwd_bwd_seconds']:.0f}s fwd+bwd, "
          f"{tm['data_seconds']:.0f}s data, {tm['eval_seconds']:.0f}s eval)", flush=True)
    return summary
