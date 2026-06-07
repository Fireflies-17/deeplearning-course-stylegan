#!/usr/bin/env bash
#
# Launch the SECOND-SEED repeats (seed=1) of the key factor-matrix arms, to put
# error bars on the two main conclusions: ADA on/off (E1 vs E2) and data scale
# (E1 vs E4). Re-runs E1, E2, E4 at the 1500 kimg fair-comparison budget. E3/E5
# are intentionally NOT repeated (lower marginal value).
#
# Same "plan C" as run_p2_parallel.sh: each run uses gpus=2, two runs execute in
# parallel on disjoint GPU pairs. Wave 1 = E1b + E2b, Wave 2 = E4b (alone).
#
# Usage:
#   bash scripts/run_p2_seed1.sh --dry-run     # validate every config, run nothing real
#   bash scripts/run_p2_seed1.sh               # launch the real training waves
#   GPU_PAIRS="0,1 2,3" bash scripts/run_p2_seed1.sh   # override GPU pairing
#
# Each run's stdout/stderr is teed to results/logs/p2-seed1/<name>.log. The script
# does NOT touch git, and does NOT run evaluation (see the runbook for the offline
# FID/KID/PR + analyze step to run afterwards).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="configs/baseline"
LOG_DIR="results/logs/p2-seed1"
mkdir -p "$LOG_DIR"

# Ordered list of (label config) pairs. Second-seed repeats of E1, E2, E4.
RUNS=(
  "E1b ${CONFIG_DIR}/p2_lsun_church256_baseline_seed1_1500.json"
  "E2b ${CONFIG_DIR}/p2_lsun_church256_noada_seed1_1500.json"
  "E4b ${CONFIG_DIR}/p2_lsun_church256_subset50k_ada_seed1_1500.json"
)

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

echo "== P2 seed=1 repeats: ${#RUNS[@]} runs, $SLOTS parallel slots, pairs: ${PAIRS[*]} =="
wave=1
i=0
while (( i < ${#RUNS[@]} )); do
  echo "== Wave $wave =="
  pids=()
  labels=()
  for (( slot=0; slot < SLOTS && i < ${#RUNS[@]}; slot++, i++ )); do
    read -r label config <<< "${RUNS[$i]}"
    log="${LOG_DIR}/${label}.log"
    echo "[launch] $label on GPUs ${PAIRS[$slot]} -> $log"
    CUDA_VISIBLE_DEVICES="${PAIRS[$slot]}" \
      python scripts/run_experiment.py train --config "$config" \
      >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
  done

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

echo "== All seed=1 repeat runs finished. Logs in $LOG_DIR =="
echo "Next: offline evaluation of each final (1500 kimg) snapshot, then re-aggregate."
echo "  for c in baseline_seed1_1500 noada_seed1_1500 subset50k_ada_seed1_1500; do"
echo "    python scripts/run_experiment.py evaluate --config configs/baseline/p2_lsun_church256_\${c}.json"
echo "  done"
