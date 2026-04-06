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
  echo "Usage: $0 [--all | <StrategyNameOrConfigPath>] [--results-root <dir>] [--param-source <auto|opt|yaml|best_yaml>] [--no-round]"
  echo "  --all                 Evaluate all strategies that have a folder under results-root"
  echo "  --results-root <dir>  Root results directory to pull params from (default: results)"
  echo "  --param-source <src>  Where to source params: auto|opt|yaml|best_yaml (default: auto)"
  echo "  --no-round            Disable rounding floats to 2 decimals in evaluator"
  exit 0
}

RESULTS_ROOT="results"
PARAM_SOURCE="auto"
NO_ROUND_FLAG=""

# Parse args (support --results-root anywhere)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage ;;
    --results-root)
      RESULTS_ROOT="$2"; shift 2 ;;
    --param-source)
      PARAM_SOURCE="$2"; shift 2 ;;
    --no-round)
      NO_ROUND_FLAG="--no_round"; shift 1 ;;
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
  echo "Evaluating config: $cfg"
  echo "Parameters from: $RESULTS_ROOT (all optimization JSONs)"
  echo "================================================================================"

  res_dir=$("$PY" - "$cfg" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    c = yaml.safe_load(f)
s = c.get('strategy', {})
print(s.get('results_dir') or s.get('class', ''))
PY
)

  if [[ -z "$res_dir" ]]; then
    echo "Warning: Could not determine results_dir for $cfg, skipping."
    continue
  fi

  opt_dir="${RESULTS_ROOT}/${res_dir}/optimizations"
  if [[ ! -d "$opt_dir" ]]; then
    res_dir_snake=$("$PY" - "$res_dir" <<'PY'
import re, sys
name = sys.argv[1]
name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
print(name.replace('__', '_').lower())
PY
  )
    opt_dir="${RESULTS_ROOT}/${res_dir_snake}/optimizations"
    if [[ -d "$opt_dir" ]]; then
      res_dir="$res_dir_snake"
    fi
  fi

  if [[ ! -d "$opt_dir" ]]; then
    echo "Warning: Optimization directory not found for $cfg ($opt_dir), skipping."
    continue
  fi

  JSON_FILES=()
  while IFS= read -r jf; do
    JSON_FILES+=("$jf")
  done < <(find "$opt_dir" -maxdepth 1 -type f -name "*.json" -print | sort)

  if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
    echo "Warning: No JSON param files found in $opt_dir, skipping."
    continue
  fi

  for json_file in "${JSON_FILES[@]}"; do
    has_params=$("$PY" - "$json_file" <<'PY'
import json, sys
try:
    with open(sys.argv[1], 'r') as f:
        d = json.load(f)
    p = d.get('params')
    print('1' if isinstance(p, dict) and p else '0')
except Exception:
    print('0')
PY
  )

    if [[ "$has_params" != "1" ]]; then
      echo "Skipping $json_file (no params object)"
      continue
    fi

    tag=$(basename "$json_file")
    tag="${tag%.json}"
    out_dir="${RESULTS_ROOT}/${res_dir}/evaluations"
    mkdir -p "$out_dir"
    out_path="${out_dir}/full_dataset_backtest__${tag}.json"

    echo "--------------------------------------------------------------------------------"
    echo "Evaluating: $cfg"
    echo "Parameters from: $json_file"
    echo "Output: $out_path"
    echo "--------------------------------------------------------------------------------"

    $PY scripts/evaluate_strategy.py \
      --strategy_config "$cfg" \
      --main_config configs/main_config.yaml \
      --results_root "$RESULTS_ROOT" \
      --param_source "$PARAM_SOURCE" \
      --param_json "$json_file" \
      --output "$out_path" \
      $NO_ROUND_FLAG
  done
done

echo "All evaluations complete."
