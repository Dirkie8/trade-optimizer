#!/bin/bash

# Simple batch runner for hybrid optimization across all strategy configs.
# - Auto-discovers strategy YAMLs in functions/configs (skips templates/examples)
# - Forwards all CLI options to scripts/hybrid_optimize.py (no parsing here)
# - Uses project venv Python if available
# - Auto-creates logs and tees all output to a timestamped file

set -u

# Prefer project venv Python if present
PY="python"
if [ -x "$(pwd)/python-3.12.4/bin/python3.12" ]; then
  PY="$(pwd)/python-3.12.4/bin/python3.12"
fi

# Basic color support (disable with NO_COLOR env var)
if [ -n "${NO_COLOR:-}" ]; then
  BOLD=""; RESET=""; RED=""; GREEN=""; YELLOW=""; CYAN=""
else
  BOLD="\033[1m"; RESET="\033[0m"; RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"
fi

# Logging: create folder and capture everything (stdout+stderr) via tee
mkdir -p logs
ts=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/optim_${ts}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "Usage: $0 [options forwarded to scripts/hybrid_optimize.py]"
  echo "Examples:"
  echo "  $0 --random_samples 200 --tpe_trials 50 --walk-forward --n-folds 4 --n_jobs 5"
  echo "Notes:"
  echo "  • All args are passed through to scripts/hybrid_optimize.py for each strategy."
  echo "  • Strategies are auto-discovered from functions/configs/*.yaml (templates/examples skipped)."
  echo "  • Log saved to $LOG_FILE"
  exit 0
fi

echo "==> Batch hybrid optimization starting $(date)"
echo "==> Python: $PY"
echo "==> Log: $LOG_FILE"
echo "==> Forwarded args: $*"

# Discover strategy configs (skip templates and examples)
STRATEGY_CONFIGS=()
while IFS= read -r f; do
  STRATEGY_CONFIGS+=("$f")
done < <(find "functions/configs" -type f -name "*.yaml" \
  ! -name "*_template*" ! -name "_template*" ! -name "*template*" \
  ! -name "example*" | sort)

if [ ${#STRATEGY_CONFIGS[@]} -eq 0 ]; then
  echo "No strategy YAMLs found in functions/configs. Exiting."
  exit 1
fi

start_time=$(date +%s)
ok=0
fail=0
FAILED=()

for cfg in "${STRATEGY_CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  # Clear, readable per-strategy header (with color)
  printf "\n${BOLD}${CYAN}=== [%s] %s ===${RESET}\n" "$name" "$(date)"
  echo "$PY -u scripts/hybrid_optimize.py --strategy_config \"$cfg\" --main_config configs/main_config.yaml $*"
  $PY -u scripts/hybrid_optimize.py \
    --strategy_config "$cfg" \
    --main_config configs/main_config.yaml \
    "$@"

  code=$?
  if [ $code -eq 0 ]; then
    printf "${GREEN}[%s] ✅ done${RESET}\n" "$name"
    ok=$((ok+1))
  else
    printf "${RED}[%s] ❌ failed (exit %d)${RESET}\n" "$name" "$code"
    fail=$((fail+1))
    FAILED+=("$name")
  fi
done

end_time=$(date +%s)
dur=$((end_time-start_time))
# Print summary with proper newlines and color for counts
printf "\n==> Completed: ${GREEN}%d ok${RESET}, ${RED}%d failed${RESET} in %ss\n" "$ok" "$fail" "$dur"
if [ $fail -gt 0 ]; then
  printf "   ${RED}Failed:${RESET} %s\n" "${FAILED[@]}"
fi
echo "==> Log saved: $LOG_FILE"
