#!/bin/bash
# Train one grid's fine-tuning series, or its from-scratch ablation, opportunistically:
# it takes any GPU that is idle, starts the next pending run on it, and keeps watching
# until every requested size is done. Several run at once when several GPUs are free.
#
#   bash scripts/train_grid.sh <grid>                      # plan only, no training
#   bash scripts/train_grid.sh <grid> --go                 # n = 10,25,50,100,200,500
#   bash scripts/train_grid.sh <grid> --sizes 50,100 --go  # a subset
#   bash scripts/train_grid.sh <grid> --scratch --go       # n = 1000 from random init
#   CAND="0 6" bash scripts/train_grid.sh <grid> --go      # restrict the device pool
#
# <grid> is one of activsg10k, tx2k, case6470_rte, case500_goc. The config is
# config/<grid>_finetune.yaml, or config/<grid>_scratch1k.yaml with --scratch, both are
# self-contained, so the recipe is whatever that one file says.
#
# RUN LABELS are n<nnnn> (n0010 ... n0500) and n1000_scratch, which is what results/,
# logs/ and checkpoints/ are keyed by and what the evaluation harness and the paper's
# figures look for. Do not rename them.
#
# DEVICE SELECTION. A device is taken only when nvidia-smi reports NO compute process on
# it and at least MIN_FREE_GB free, so an idle GPU belonging to someone else is used and a
# busy one is left alone. That is a courtesy check, not a reservation: if another user
# starts a job on a GPU we are already training on, both slow down. A freshly launched run
# takes 30-60 s to show a CUDA context, so each launch writes a claim file and the device
# counts as ours while that PID lives -- otherwise the loop would fire two runs onto one
# device inside a single poll.
#
# This replaces the per-grid launch_*/queue_*/sweep_* scripts, which are in
# backup/finetune_model/scripts/.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
GRID="${1:-}"; shift || true
[ -n "$GRID" ] || { echo "usage: train_grid.sh <grid> [--sizes a,b,c] [--scratch] [--go]"; exit 2; }
SIZES="10 25 50 100 200 500"; SCRATCH=0; GO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sizes) SIZES=$(echo "$2" | tr ',' ' '); shift 2;;
    --scratch) SCRATCH=1; shift;;
    --go) GO=1; shift;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
CAND="${CAND:-0 1 2 3 4 5 6 7}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
POLL="${POLL:-60}"
if [ "$SCRATCH" = 1 ]; then
  CFG="config/${GRID}_scratch1k.yaml"; QUEUE="1000"; LABELS="n1000_scratch"
else
  CFG="config/${GRID}_finetune.yaml";  QUEUE="$SIZES"; LABELS=""
fi
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 2; }
OUT="logs/_train_grid"; mkdir -p "$OUT"
LOG="$(pwd)/$OUT/${GRID}$([ "$SCRATCH" = 1 ] && echo _scratch).log"
CLAIM="$(pwd)/$OUT/.claim"; mkdir -p "$CLAIM"
say(){ echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
label_for(){ [ "$SCRATCH" = 1 ] && echo n1000_scratch || printf "n%04d" "$1"; }

avail(){
  local d="$1" cp free pid
  cp=$(nvidia-smi -i "$d" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [ -n "$cp" ] && return 1
  free=$(nvidia-smi -i "$d" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$free" ] && return 1
  [ "$free" -lt $((MIN_FREE_GB*1024)) ] && return 1
  if [ -f "$CLAIM/$d" ]; then
    pid=$(cat "$CLAIM/$d"); kill -0 "$pid" 2>/dev/null && return 1; rm -f "$CLAIM/$d"
  fi
  return 0
}

say "=== plan ==="
say "  grid    : $GRID"
say "  config  : $CFG"
say "  runs    : $(for n in $QUEUE; do printf '%s ' "$(label_for "$n")"; done)"
say "  devices : candidates [$CAND], need ${MIN_FREE_GB} GB free and no compute process"
for d in $CAND; do avail "$d" && say "  GPU $d available" || say "  GPU $d busy"; done
[ "$GO" = 1 ] || { say "DRY RUN -- nothing launched. Re-run with --go."; exit 0; }

declare -A RUNPID=() RUNGPU=()
PENDING="$QUEUE"
while :; do
  for lab in "${!RUNPID[@]}"; do
    if ! kill -0 "${RUNPID[$lab]}" 2>/dev/null; then
      say "  DONE $lab (GPU ${RUNGPU[$lab]})"; rm -f "$CLAIM/${RUNGPU[$lab]}"
      unset "RUNPID[$lab]" "RUNGPU[$lab]"
    fi
  done
  if [ -n "${PENDING// /}" ]; then
    for d in $CAND; do
      [ -n "${PENDING// /}" ] || break
      avail "$d" || continue
      n=${PENDING%% *}; [ "$n" = "$PENDING" ] && PENDING="" || PENDING="${PENDING#* }"
      lab=$(label_for "$n")
      say "  LAUNCH $lab (train_subset=$n) on GPU $d"
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$d" \
        setsid $PY scripts/finetune.py --config "$CFG" --train_subset "$n" \
          --device cuda --run_label "$lab" > "$OUT/${GRID}_${lab}.out" 2>&1 < /dev/null &
      p=$!; RUNPID[$lab]=$p; RUNGPU[$lab]=$d; echo "$p" > "$CLAIM/$d"; sleep 5
    done
  fi
  [ -z "${PENDING// /}" ] && [ "${#RUNPID[@]}" -eq 0 ] && break
  sleep "$POLL"
done
say "TRAIN GRID COMPLETE: $GRID $([ "$SCRATCH" = 1 ] && echo scratch || echo "$SIZES")"
