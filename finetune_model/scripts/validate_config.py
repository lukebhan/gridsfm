#!/usr/bin/env python3
"""Dry-run a config: resolve it, check every path, report the derived values. No GPU.

    python scripts/validate_config.py --config config/case500_goc_finetune.yaml
"""
import argparse, os, sys
import _bootstrap  # noqa: F401
import config as configmod
from dataset import count_records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", help="overrides data.root (dataset location)")
    ap.add_argument("--data_dir", help="overrides data.dir (this dataset's directory)")
    a = ap.parse_args()

    cfg = configmod.load_config(a.config, data_root=a.data_root, data_dir=a.data_dir)
    warn = configmod.validate(cfg, require_data=True)
    print(f"{cfg['grid_id']}  (config sha {cfg['_meta']['config_sha256']})")
    print(f"  base checkpoint  {cfg['base_checkpoint']}")
    n = {}
    for s in ("train", "val", "test"):
        p = cfg["data"][s]
        n[s] = count_records(p)
        print(f"  {s+'_data.jsonl':<16} {n[s]:>6} cases   {os.path.getsize(p)/1e9:.2f} GB")
    print(f"  {'TOTAL':<16} {sum(n.values()):>6} cases")
    T, L, E = cfg["train"], cfg["loss"], cfg["eval"]
    print(f"\n  MODEL     clip_mode={cfg['model']['clip_mode']}  freeze={cfg['model']['freeze']}")
    print(f"  SCHEDULE  epochs={T['epochs']}  batch={T['batch']}x accum {T['accum']} "
          f"= {T['batch']*T['accum']} effective")
    print(f"  OPTIM     {T['optimizer']}  lr={T['lr']:g}  betas={tuple(T['betas'])}  "
          f"eps={T['eps']:g}  weight_decay={T['weight_decay']:g}"
          + (f"  momentum={T['momentum']} nesterov={T['nesterov']}"
             if str(T['optimizer']).lower() == 'sgd' else ""))
    sched = f"{T['scheduler']}"
    if str(T['scheduler']).lower() == 'cosine':
        sched += f" floor={T['lr_min_frac']:g}x (lr {T['lr']*T['lr_min_frac']:g})"
    elif str(T['scheduler']).lower() == 'step':
        sched += f" every {T['step_size']}ep x{T['step_gamma']:g}"
    if T['warmup_epochs']:
        sched += (f" | warmup {T['warmup_epochs']}ep from "
                  f"{T['warmup_start_frac']:g}x (lr {T['lr']*T['warmup_start_frac']:g})")
    print(f"  LR SCHED  {sched}")
    print(f"  CLIP      max_norm={T['clip_grad']:g} (L{T['clip_norm_type']:g})"
          + ("  [DISABLED]" if not T['clip_grad'] else ""))
    print(f"  SELECT    select_on={T['selectON'] if False else T['select_on']}  "
          f"early_stop_patience={T['early_stop_patience']}"
          + (f" (min_delta {T['early_stop_min_delta']:g})" if T['early_stop_patience'] else "")
          + f"  score = {E['val_score_pg_weight']:g}*Pg% + {E['val_score_v_weight']:g}*V(p.u.)")
    print(f"  SEED      {T['seed']}  deterministic={T['deterministic']}  "
          f"tf32={cfg['runtime']['tf32']}")
    # The top-k / mean / max weights belong to the max-aligned CONTROL loss only. Printing
    # them under an elastic config invites reading a shape that run never uses.
    mode = str(L.get("mode", "control")).lower()
    if mode == "elastic":
        print(f"  LOSS      mode=elastic  L = cost + {L['rho_g']:g}*sum log(1+|F|) + "
              f"{L['rho_h']:g}*sum log(1+relu(h~))  [on the PF-CLOSED state]")
        print(f"            + w_control={L.get('w_control', 0.0):g}*RMSE-to-GT "
              f"+ w_merit={L.get('w_merit', 0.0):g}*0.5*mean(F^2)")
    else:
        print(f"  LOSS      L = {L['mean_weight']:g}*mean + {L['topk_weight']:g}*topk "
              f"+ {L['max_weight']:g}*max   total = {L['w_pg']:g}*Pg + {L['w_v']:g}*V")
    print(f"            tol_pg={L['tol_pg']:g} p.u.  tol_v={L['tol_v']:g} p.u."
          + (f"  segmented_topk={L['segmented_topk']}" if mode == "control" else ""))
    if mode == "control":
        kp = L['topk_pg'] if L['topk_pg'] else f"{L['topk_pg_frac']:.2%} of scored gens"
        kv = L['topk_v'] if L['topk_v'] else f"{L['topk_v_frac']:.2%} of scored V-ctrl buses"
        print(f"            topk_pg = {kp}   topk_v = {kv}   (derived at prep_cache time)")
    print(f"  EVAL      val batch={E['batch']}  test batch={E['test_batch']}")
    print(f"  LOGGING   step_log_every={cfg['logging']['step_log_every']}  "
          f"plot_every={cfg['logging']['plot_every_epochs']}ep  "
          f"save_every={cfg['logging']['save_every_epochs'] or 'off'}")
    print(f"  RUNTIME   device={cfg['runtime']['device']}  "
          f"min_free_vram={cfg['runtime']['min_free_vram_gb']} GB")
    print(f"\n  data    -> {cfg['data']['dir']}")
    print(f"  cache   -> {cfg['cache']['dir']}  ({cfg['cache']['workers']} workers, "
          f"{cfg['cache']['mem_budget_gb']} GB LRU budget)")
    print(f"  results -> {cfg['_meta']['results_dir']}\n  logs    -> {cfg['_meta']['logs_dir']}")
    for w in warn:
        print(f"  WARNING: {w}")
    print("\n  config OK" if not warn else "\n  config OK (with warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
