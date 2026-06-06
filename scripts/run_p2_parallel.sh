#!/usr/bin/env bash
#
# Launch the P2 factor-matrix runs (E2-E5, optional E6) on a 4-GPU machine using
# "plan C": each run still uses gpus=2 (so it stays bit-for-bit comparable to the
# E1 baseline), and TWO runs execute in parallel on disjoint GPU pairs. Independent
# jobs have no inter-GPU communication overhead, so this halves wall-clock for the
# same total GPU-hours as running 2-GPU jobs sequentially.
#
# Layout: wave 1 = E2 on GPUs 0,1 + E3 on GPUs 2,3 (in parallel), wait for both,
# then wave 2 = E4 on GPUs 0,1 + E5 on GPUs 2,3. Add E6 with INCLUDE_E6=1 (runs
# alone on GPUs 0,1 as a third wave).
#
# Usage:
#   bash scripts/run_p2_parallel.sh --dry-run     # validate every config, run nothing real
#   bash scripts/run_p2_parallel.sh               # launch the real training waves
#   INCLUDE_E6=1 bash scripts/run_p2_parallel.sh  # also run the optional target=0.8 group
#   GPU_PAIRS="0,1 2,3" bash scripts/run_p2_parallel.sh   # override GPU pairing
#
# Each run's stdout/stderr is teed to results/logs/p2-launch/<name>.log. The script
# does NOT touch git.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="configs/baseline"
LOG_DIR="results/logs/p2-launch"
mkdir -p "$LOG_DIR"

# Ordered list of (label config) pairs. E1 is already trained, so it is excluded.
RUNS=(
  "E2 ${CONFIG_DIR}/p2_lsun_church256_noada_1500.json"
  "E3 ${CONFIG_DIR}/p2_lsun_church256_fixedp02_1500.json"
  "E4 ${CONFIG_DIR}/p2_lsun_church256_subset50k_ada_1500.json"
  "E5 ${CONFIG_DIR}/p2_lsun_church256_target04_1500.json"
)
if [[ "${INCLUDE_E6:-0}" == "1" ]]; then
  RUNS+=("E6 ${CONFIG_DIR}/p2_lsun_church256_target08_1500.json")
fi

# GPU pairs for the two parallel slots. Each run config sets gpus=2.
read -r -a PAIRS <<< "${GPU_PAIRS:-0,1 2,3}"
SLOTS=${#PAIRS[@]}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# --- Dry-run mode: validate every config sequentially, then exit. ---
if [[ "$DRY_RUN" == "1" ]]; then
  echo "== Validating ${#RUNS[@]} configs with --dry-run =="
  for entry in "${RUNS[@]}"; do
    read -r label config <<< "$entry"
    echo "---- $label ($config) ----"
    python scripts/run_experiment.py train --config "$config" --dry-run
  done
  echo "== All configs validated. =="
  exit 0
fi

run_one() {
  # run_one LABEL CONFIG "GPUS"
  local label="$1" config="$2" gpus="$3"
  local log="${LOG_DIR}/${label}.log"
  echo "[launch] $label on GPUs $gpus -> $log"
  CUDA_VISIBLE_DEVICES="$gpus" \
    python scripts/run_experiment.py train --config "$config" \
    >"$log" 2>&1 &
  echo $!  # return the PID
}

echo "== P2 plan C: ${#RUNS[@]} runs, $SLOTS parallel slots, pairs: ${PAIRS[*]} =="
wave=1
i=0
while (( i < ${#RUNS[@]} )); do
  echo "== Wave $wave =="
  pids=()
  labels=()
  for (( slot=0; slot < SLOTS && i < ${#RUNS[@]}; slot++, i++ )); do
    read -r label config <<< "${RUNS[$i]}"
    pid=$(run_one "$label" "$config" "${PAIRS[$slot]}")
    pids+=("$pid")
    labels+=("$label")
  done

  # Wait for every job in this wave; fail loudly if any non-zero exit.
  fail=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[done] ${labels[$idx]} (pid ${pids[$idx]})"
    else
      echo "[FAILED] ${labels[$idx]} (pid ${pids[$idx]}) -- see ${LOG_DIR}/${labels[$idx]}.log" >&2
      fail=1
    fi
  done
  if (( fail )); then
    echo "== Wave $wave had failures; stopping. ==" >&2
    exit 1
  fi
  echo "== Wave $wave complete. =="
  wave=$(( wave + 1 ))
done

echo "== All P2 runs finished. Logs in $LOG_DIR =="
echo "Next: offline learning curves + metrics, e.g."
echo "  python scripts/analyze_results.py \\"
echo "    --run E1=results/runs/p1-lsun-church256-100k-baseline \\"
echo "    --run E2=results/runs/p2-lsun-church256-100k-noada-1500 \\"
echo "    --run E3=results/runs/p2-lsun-church256-100k-fixedp02-1500 \\"
echo "    --run E4=results/runs/p2-lsun-church256-50k-ada-1500 \\"
echo "    --run E5=results/runs/p2-lsun-church256-100k-ada-target04-1500"
