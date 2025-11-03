#!/bin/bash
################################################################################
# Compare All Strategy Performance Metrics
################################################################################
#
# Quick comparison of all optimized strategies' evaluation results.
# Displays a formatted table with key metrics.
#
# Usage:
#   ./compare_all_strategy_performance_metrics.sh
#
# Or with options:
#   ./compare_all_strategy_performance_metrics.sh --sort return
#   ./compare_all_strategy_performance_metrics.sh --min-trades 50
#
################################################################################

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "python-3.12.4/bin" ]; then
    source python-3.12.4/bin/activate
fi

# Run the comparison script with all arguments passed through
python scripts/compare_strategies.py "$@"
