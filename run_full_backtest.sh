#!/bin/bash

# Minimal runner: evaluate all strategies (or one) on full data using best params
# - Auto-discovers strategy YAMLs (skips templates/examples)
# - Prefers project venv Python if available

set -euo pipefail

PY="python"
if [ -x "$(pwd)/python-3.12.4/bin/python3.12" ]; then
  PY="$(pwd)/python-3.12.4/bin/python3.12"
fi

usage() {
  echo "Usage: $0 [--all | <StrategyNameOrConfigPath>] [--results-root <dir>]"
  echo "  --all                 Evaluate all strategies that have a folder under results-root"
  echo "  --results-root <dir>  Root results directory to pull params from (default: results)"
  exit 0
}

RESULTS_ROOT="results"

# Parse args (support --results-root anywhere)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage ;;
    --results-root)
      RESULTS_ROOT="$2"; shift 2 ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

set -- "${ARGS[@]}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then usage; fi

# Build list of configs
CONFIGS=()
if [[ ${1:-"--all"} == "--all" ]]; then
  # Discover by scanning RESULTS_ROOT for strategy folders that have optimization artifacts
  # Then match them to their config files
  if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "Error: Results root directory '$RESULTS_ROOT' does not exist."
    exit 1
  fi
  
  for strat_dir in "$RESULTS_ROOT"/*/; do
    # Get the folder name (strip trailing /)
    strat_name=$(basename "$strat_dir")
    
    # Skip if no optimizations folder
    if [[ ! -d "$strat_dir/optimizations" ]]; then
      continue
    fi
    
    # Check if there's a bayesian_wf best JSON (the params we want)
    if [[ ! -f "$strat_dir/optimizations/bayesian_wf_optimization_results_best.json" ]] && \
       [[ ! -f "$strat_dir/optimizations/bayesian_optimization_results_best.json" ]] && \
       [[ ! -f "$strat_dir/optimizations/optimization_results_best.json" ]] && \
       [[ ! -f "$strat_dir/optimizations/optimization_results.csv" ]]; then
      continue
    fi
    
    # Find matching config file
    # Try exact match first (snake_case or as-is)
    cfg_exact="functions/configs/${strat_name}.yaml"
    if [[ -f "$cfg_exact" ]]; then
      CONFIGS+=("$cfg_exact")
      continue
    fi
    
    # Try matching by reading YAMLs and comparing results_dir or class
    found=0
    while IFS= read -r cfg; do
      match=$("$PY" -c "
import sys, yaml
try:
    with open('$cfg', 'r') as f:
        c = yaml.safe_load(f)
    s = c.get('strategy', {})
    res_dir = s.get('results_dir') or s.get('class', '')
    print('1' if res_dir == '$strat_name' else '0')
except:
    print('0')
" 2>/dev/null)
      if [[ "$match" == "1" ]]; then
        CONFIGS+=("$cfg")
        found=1
        break
      fi
    done < <(find functions/configs -type f -name "*.yaml" ! -name "*_template*" ! -name "_template*" ! -name "*template*" ! -name "example*" 2>/dev/null)
    
    if [[ $found -eq 0 ]]; then
      echo "Warning: No config found for strategy folder '$strat_name', skipping."
    fi
  done
else
  arg="$1"
  if [[ -f "$arg" ]]; then
    CONFIGS=("$arg")
  else
    # resolve by strategy class or results_dir match
    while IFS= read -r f; do
      if "$PY" - <<'PY'
import sys, yaml, os
cfg_path=sys.argv[1]
name=sys.argv[2]
with open(cfg_path,'r') as f:
    c=yaml.safe_load(f)
s=c.get('strategy',{})
print(1 if (s.get('class')==name or s.get('results_dir')==name) else 0)
PY
 "$f" "$arg" | grep -q '^1$'; then CONFIGS+=("$f"); fi
    done < <(find functions/configs -type f -name "*.yaml")
  fi
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "No strategy configs found to evaluate."
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "================================================================================"
  echo "Evaluating: $cfg"
  echo "Parameters from: $RESULTS_ROOT"
  echo "================================================================================"
  $PY scripts/evaluate_strategy.py \
    --strategy_config "$cfg" \
    --main_config configs/main_config.yaml \
    --results_root "$RESULTS_ROOT"
done

echo "All evaluations complete."
