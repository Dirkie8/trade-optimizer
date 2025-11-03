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
  echo "Usage: $0 [--all | <StrategyNameOrConfigPath>]"
  exit 0
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then usage; fi

# Build list of configs
CONFIGS=()
if [[ ${1:-"--all"} == "--all" ]]; then
  while IFS= read -r f; do CONFIGS+=("$f"); done < <(find functions/configs -type f -name "*.yaml" ! -name "*_template*" ! -name "_template*" ! -name "*template*" ! -name "example*" | sort)
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
  echo "Evaluating: $cfg"
  $PY scripts/evaluate_strategy.py \
    --strategy_config "$cfg" \
    --main_config configs/main_config.yaml
done

echo "All evaluations complete."
