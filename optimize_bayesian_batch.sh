#!/bin/bash
################################################################################
# Batch Bayesian Optimization Script
################################################################################
#
# This script runs Bayesian optimization for all strategies using Optuna TPE.
# Each strategy's results are saved to its own results directory.
#
# Usage:
#   ./optimize_bayesian_batch.sh
#
# Or with custom parameters:
#   ./optimize_bayesian_batch.sh --n_trials 200 --n_jobs 8
#
################################################################################

# Default parameters (can be overridden by command line args)
N_TRIALS=100
N_JOBS=4
MAIN_CONFIG="configs/main_config.yaml"
SORT_BY="reward_metric"

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch Bayesian optimization for all trading strategies.

OPTIONS:
    --n_trials N        Number of Bayesian optimization trials (default: 100)
    --n_jobs N          Number of parallel workers (default: 4, -1 for auto)
    --main_config PATH  Path to main config (default: configs/main_config.yaml)
    --sort-by METRIC    Sort metric: reward_metric, sharpe, sortino, calmar,
                        consistency_score, total_return_pct (default: reward_metric)
    --help              Show this help message

EXAMPLES:
    # Quick test (50 trials, single core)
    $0 --n_trials 50 --n_jobs 1

    # Medium run (200 trials, 4 cores)
    $0 --n_trials 200 --n_jobs 4

    # Full optimization (500 trials, 8 cores)
    $0 --n_trials 500 --n_jobs 8

    # Auto-detect cores
    $0 --n_trials 300 --n_jobs -1

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --main_config)
            MAIN_CONFIG="$2"
            shift 2
            ;;
        --sort-by)
            SORT_BY="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv/bin" ]; then
    echo -e "${BLUE}Activating virtual environment (.venv)...${NC}"
    source .venv/bin/activate
elif [ -d "python-3.12.4/bin" ]; then
    echo -e "${BLUE}Activating virtual environment (python-3.12.4)...${NC}"
    source python-3.12.4/bin/activate
fi

# Discover all strategy configs
echo -e "${CYAN}Discovering strategy configs...${NC}"
STRATEGIES=()
for config in functions/configs/*.yaml; do
    # Skip template files
    if [[ $(basename "$config") == _template* ]]; then
        continue
    fi
    
    # Check if config has parameters_bayesian section
    if grep -q "parameters_bayesian:" "$config"; then
        strategy_name=$(basename "$config" .yaml)
        STRATEGIES+=("$strategy_name:$config")
    else
        echo -e "${YELLOW}  ⚠ Skipping $(basename "$config") - no parameters_bayesian section${NC}"
    fi
done

if [ ${#STRATEGIES[@]} -eq 0 ]; then
    echo -e "${RED}No strategies with parameters_bayesian found!${NC}"
    exit 1
fi

# Print configuration
echo ""
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║         BATCH BAYESIAN OPTIMIZATION (Optuna TPE)              ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Trials:       ${GREEN}${N_TRIALS}${NC}"
echo -e "  Jobs:         ${GREEN}${N_JOBS}${NC}"
echo -e "  Sort by:      ${GREEN}${SORT_BY}${NC}"
echo -e "  Main Config:  ${GREEN}${MAIN_CONFIG}${NC}"
echo -e "  Strategies:   ${GREEN}${#STRATEGIES[@]}${NC}"
echo ""
echo -e "${CYAN}Strategies to optimize:${NC}"
for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME _ <<< "$strategy_info"
    echo -e "  • ${STRATEGY_NAME}"
done
echo ""

# Confirmation prompt
read -p "Continue with optimization? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Optimization cancelled.${NC}"
    exit 0
fi
echo ""

# Track timing and results
START_TIME=$(date +%s)
SUCCESSFUL=0
FAILED=0
FAILED_STRATEGIES=()
STRATEGY_TIMES=()

# Run optimization for each strategy
for i in "${!STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME CONFIG_FILE <<< "${STRATEGIES[$i]}"
    
    STRATEGY_NUM=$((i + 1))
    TOTAL_STRATEGIES=${#STRATEGIES[@]}
    
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}[${STRATEGY_NUM}/${TOTAL_STRATEGIES}] Bayesian Optimization: ${GREEN}${STRATEGY_NAME}${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Config: ${CONFIG_FILE}"
    echo -e "Trials: ${N_TRIALS} | Jobs: ${N_JOBS}"
    echo ""
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
        echo ""
        FAILED=$((FAILED + 1))
        FAILED_STRATEGIES+=("$STRATEGY_NAME (config not found)")
        continue
    fi
    
    # Run Bayesian optimization
    STRATEGY_START=$(date +%s)
    
    python scripts/optimize_bayesian.py \
        --strategy_config "$CONFIG_FILE" \
        --main_config "$MAIN_CONFIG" \
        --n_trials "$N_TRIALS" \
        --n_jobs "$N_JOBS" \
        --sort-by "$SORT_BY"
    
    EXIT_CODE=$?
    STRATEGY_END=$(date +%s)
    STRATEGY_DURATION=$((STRATEGY_END - STRATEGY_START))
    
    # Format duration
    HOURS=$((STRATEGY_DURATION / 3600))
    MINUTES=$(((STRATEGY_DURATION % 3600) / 60))
    SECONDS=$((STRATEGY_DURATION % 60))
    
    # Check if optimization was successful
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ ${STRATEGY_NAME} completed successfully${NC}"
        echo -e "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        SUCCESSFUL=$((SUCCESSFUL + 1))
        STRATEGY_TIMES+=("$STRATEGY_NAME: ${HOURS}h ${MINUTES}m ${SECONDS}s")
        
        # Show best params if available
        BEST_JSON="results/${STRATEGY_NAME}/optimizations/bayesian_optimization_results_best.json"
        if [ -f "$BEST_JSON" ]; then
            echo -e "${CYAN}  Best reward:${NC} $(python -c "import json; print(json.load(open('$BEST_JSON'))['metrics'].get('reward_metric', 'N/A'))" 2>/dev/null || echo "N/A")"
        fi
    else
        echo ""
        echo -e "${RED}✗ ${STRATEGY_NAME} failed with exit code ${EXIT_CODE}${NC}"
        FAILED=$((FAILED + 1))
        FAILED_STRATEGIES+=("$STRATEGY_NAME (exit code $EXIT_CODE)")
    fi
    
    echo ""
    
    # Brief pause between strategies
    sleep 1
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Print summary
echo ""
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║         OPTIMIZATION SUMMARY                                   ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Results:${NC}"
echo -e "  ${GREEN}✓ Successful:${NC} ${SUCCESSFUL}/${TOTAL_STRATEGIES}"
if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}✗ Failed:${NC}     ${FAILED}/${TOTAL_STRATEGIES}"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed strategies:${NC}"
    for failed in "${FAILED_STRATEGIES[@]}"; do
        echo -e "  - $failed"
    done
    echo ""
fi

echo -e "${BOLD}Timing:${NC}"
echo -e "  ${YELLOW}Total time:${NC} ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
if [ ${#STRATEGY_TIMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${CYAN}Per-strategy times:${NC}"
    for time_info in "${STRATEGY_TIMES[@]}"; do
        echo -e "  $time_info"
    done
fi
echo ""

# Show results directories
echo -e "${BOLD}Results:${NC}"
for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME _ <<< "$strategy_info"
    RESULTS_DIR="results/${STRATEGY_NAME}/optimizations"
    CSV_FILE="$RESULTS_DIR/bayesian_optimization_results.csv"
    JSON_FILE="$RESULTS_DIR/bayesian_optimization_results_best.json"
    VAL_FILE="$RESULTS_DIR/bayesian_optimization_results_validation.json"
    
    if [ -f "$CSV_FILE" ] && [ -f "$JSON_FILE" ]; then
        echo -e "  ${GREEN}✓${NC} ${STRATEGY_NAME}"
        echo -e "      CSV:  ${CSV_FILE}"
        echo -e "      Best: ${JSON_FILE}"
        if [ -f "$VAL_FILE" ]; then
            echo -e "      Val:  ${VAL_FILE}"
        fi
    else
        echo -e "  ${RED}✗${NC} ${STRATEGY_NAME} (results missing)"
    fi
done
echo ""

# Performance summary
if [ $SUCCESSFUL -gt 0 ]; then
    AVG_TIME=$((TOTAL_DURATION / SUCCESSFUL))
    AVG_HOURS=$((AVG_TIME / 3600))
    AVG_MINUTES=$(((AVG_TIME % 3600) / 60))
    AVG_SECONDS=$((AVG_TIME % 60))
    echo -e "${CYAN}Average time per strategy:${NC} ${AVG_HOURS}h ${AVG_MINUTES}m ${AVG_SECONDS}s"
    echo ""
fi

# Next steps
echo -e "${BOLD}${MAGENTA}Next steps:${NC}"
echo -e "  1. Review validation results:"
echo -e "     ${CYAN}cat results/<strategy>/optimizations/bayesian_optimization_results_validation.json${NC}"
echo ""
echo -e "  2. Compare strategies:"
echo -e "     ${CYAN}python scripts/compare_strategies.py${NC}"
echo ""
echo -e "  3. Run full backtest on best strategy:"
echo -e "     ${CYAN}python scripts/evaluate_strategy.py --strategy_config <config>${NC}"
echo ""

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Completed with ${FAILED} failure(s).${NC}"
    exit 1
else
    echo -e "${GREEN}${BOLD}All optimizations completed successfully! 🎉${NC}"
    exit 0
fi
