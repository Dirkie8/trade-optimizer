#!/bin/bash
################################################################################
# Batch Bayesian Walk-Forward Optimization Script
################################################################################
#
# This script runs Bayesian optimization with walk-forward validation for all
# strategies using Optuna TPE. Uses optimize_bayesian_walkforward.py.
#
# Usage:
#   ./optimize_bayesian_wf_batch.sh
#
# Or with custom parameters:
#   ./optimize_bayesian_wf_batch.sh --n_trials 200 --n_jobs 8 --n_folds 10
#
################################################################################

# Default parameters (can be overridden by command line args)
N_TRIALS=100
N_JOBS=4
N_FOLDS=8
VALIDATION_RATIO=""  # Will be read from main_config.yaml if not specified
REWARD="balanced"
MAIN_CONFIG="configs/main_config.yaml"
AUTO_YES=0
# Early stop on parameter stagnation (new)
PARAM_STAGNATION_PATIENCE=0   # 0 disables
PARAM_TOLERANCE=0.0           # float tolerance for equality
# Optional filters (comma-separated names). Match against strategy file base (e.g., rsi_strategy)
# Example: --include "RSIStrategy,ADXTrend" or --exclude "BollingerBreakout,MACDMomentum"
INCLUDE_STRATS=""
EXCLUDE_STRATS=""

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch Bayesian optimization with walk-forward validation for all trading strategies.

OPTIONS:
    --n_trials N           Number of Bayesian optimization trials (default: 100)
    --n_jobs N             Number of parallel workers (default: 4)
    --n_folds N            Number of walk-forward folds on training set (default: 8)
    --validation_ratio R   Hold-out validation ratio 0-1 (default: 0.15 = 15%)
    --reward TYPE          Reward metric: balanced, consistency, sharpe, sortino, 
                           calmar (default: balanced)
    --param_stagnation_patience N  Early stop if best params unchanged N consecutive trials (default: 0 disabled)
    --param_tolerance F           Float tolerance for param equality (default: 0.0 exact)
    --main_config PATH     Path to main config (default: configs/main_config.yaml)
    --include LIST         Comma-separated strategies to include (subset run)
    --exclude LIST         Comma-separated strategies to exclude
    --yes, -y              Skip interactive confirmation prompt (assume yes)
    -h, --help             Show this help message

EXAMPLES:
    # Quick test (20 trials, 4 cores, 5 folds)
    $0 --n_trials 20 --n_jobs 4 --n_folds 5

    # Recommended production run (100 trials, 8 folds)
    $0 --n_trials 100 --n_jobs 4 --n_folds 8 --validation_ratio 0.15

    # High-performance overnight run (200 trials, 10 folds, 8 cores)
    $0 --n_trials 200 --n_jobs 8 --n_folds 10 --validation_ratio 0.15

    # Use consistency reward metric
    $0 --n_trials 100 --n_folds 8 --reward consistency

Notes:
    - When splitting a command across multiple lines, add a trailing \\ at the end of each line.
        For example:
            $0 \\
                --n_trials 4 \\
                --n_jobs 4 \\
                --n_folds 5 \\
                --validation_ratio 0.15 \\
                --reward consistency

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
        --n_folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --validation_ratio)
            VALIDATION_RATIO="$2"
            shift 2
            ;;
        --reward)
            REWARD="$2"
            shift 2
            ;;
        --param_stagnation_patience)
            PARAM_STAGNATION_PATIENCE="$2"
            shift 2
            ;;
        --param_tolerance)
            PARAM_TOLERANCE="$2"
            shift 2
            ;;
        --main_config)
            MAIN_CONFIG="$2"
            shift 2
            ;;
        --include)
            INCLUDE_STRATS="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_STRATS="$2"
            shift 2
            ;;
        --yes|-y)
            AUTO_YES=1
            shift 1
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help or -h for usage information"
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

# Setup logging (tee to both terminal and a timestamped log file)
# Logs will be stored under ./logs/bayesian_batch_test_YYYYMMDD_HHMMSS.log
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/bayesian_batch_test_${TS}.log"
# Start redirecting all subsequent stdout/stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Inform user where logs are written
echo "Logging to: $LOG_FILE"


# Activate virtual environment if it exists
if [ -d ".venv/bin" ]; then
    echo -e "${BLUE}Activating virtual environment (.venv)...${NC}"
    source .venv/bin/activate
elif [ -d "python-3.12.4/bin" ]; then
    echo -e "${BLUE}Activating virtual environment (python-3.12.4)...${NC}"
    source python-3.12.4/bin/activate
fi

# Discover all strategy configs (robust, no alias interference)
echo -e "${CYAN}Discovering strategy configs...${NC}"
STRATEGIES=()

# Use find with -print0 to avoid word splitting and ensure we never attempt to execute paths
while IFS= read -r -d '' config; do
    base="$(basename "$config")"
    # Skip template files
    if [[ "$base" == _template* ]]; then
        continue
    fi
    # Ensure it is a regular file
    if [ ! -f "$config" ]; then
        continue
    fi
    # Check if config has parameters_bayesian section using system grep explicitly
    if /usr/bin/grep -q "parameters_bayesian:" "$config"; then
        strategy_name="${base%.yaml}"
        STRATEGIES+=("$strategy_name:$config")
    else
        echo -e "${YELLOW}  ⚠ Skipping ${base} - no parameters_bayesian section${NC}"
    fi
done < <(find functions/configs -maxdepth 1 -type f -name "*.yaml" -print0 | sort -z)

# Apply include/exclude filters if provided
if [ -n "$INCLUDE_STRATS" ] || [ -n "$EXCLUDE_STRATS" ]; then
    IFS=',' read -r -a INCLUDE_ARR <<< "${INCLUDE_STRATS}"
    IFS=',' read -r -a EXCLUDE_ARR <<< "${EXCLUDE_STRATS}"
    
    # Normalize to lowercase for comparison
    # Also strip underscores so that ADXTrend matches adx_trend, RSIStrategy matches rsi_strategy
    for idx in "${!INCLUDE_ARR[@]}"; do INCLUDE_ARR[$idx]="$(echo "${INCLUDE_ARR[$idx]}" | tr '[:upper:]' '[:lower:]' | tr -d '_' | xargs)"; done
    for idx in "${!EXCLUDE_ARR[@]}"; do EXCLUDE_ARR[$idx]="$(echo "${EXCLUDE_ARR[$idx]}" | tr '[:upper:]' '[:lower:]' | tr -d '_' | xargs)"; done

    FILTERED=()
    for entry in "${STRATEGIES[@]}"; do
        IFS=':' read -r SNAME SFILE <<< "$entry"
    LNAME="$(echo "$SNAME" | tr '[:upper:]' '[:lower:]')"
    LNAME_NORM="$(echo "$LNAME" | tr -d '_')"

        # Exclude match?
        EXCL_MATCH=0
        for ex in "${EXCLUDE_ARR[@]}"; do
            [ -z "$ex" ] && continue
            if [[ "$LNAME_NORM" == *"$ex"* ]]; then EXCL_MATCH=1; break; fi
        done
        [ $EXCL_MATCH -eq 1 ] && continue

        # Include logic: if include list provided, require a match
        if [ ${#INCLUDE_ARR[@]} -gt 0 ] && [ -n "${INCLUDE_ARR[0]}" ]; then
            INCL_MATCH=0
            for inc in "${INCLUDE_ARR[@]}"; do
                [ -z "$inc" ] && continue
                if [[ "$LNAME_NORM" == *"$inc"* ]]; then INCL_MATCH=1; break; fi
            done
            [ $INCL_MATCH -eq 0 ] && continue
        fi

        FILTERED+=("$entry")
    done
    STRATEGIES=("${FILTERED[@]}")
fi

if [ ${#STRATEGIES[@]} -eq 0 ]; then
    echo -e "${RED}No strategies with parameters_bayesian found!${NC}"
    exit 1
fi

# Read validation_ratio from main_config.yaml if not specified via CLI
if [ -z "$VALIDATION_RATIO" ]; then
    VALIDATION_RATIO=$(python3 -c "
import yaml
with open('$MAIN_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('general', {}).get('validation_ratio', 0.2))
")
    echo -e "${YELLOW}Using validation_ratio from main_config.yaml: ${VALIDATION_RATIO}${NC}"
fi

# Print configuration
echo ""
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║    BATCH BAYESIAN WALK-FORWARD OPTIMIZATION (Optuna TPE)      ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Trials:           ${GREEN}${N_TRIALS}${NC}"
echo -e "  Parallel Jobs:    ${GREEN}${N_JOBS}${NC}"
echo -e "  WF Folds:         ${GREEN}${N_FOLDS}${NC} (on training set)"
echo -e "  Validation:       ${GREEN}${VALIDATION_RATIO}${NC} (hold-out)"
echo -e "  Reward Metric:    ${GREEN}${REWARD}${NC}"
echo -e "  Main Config:      ${GREEN}${MAIN_CONFIG}${NC}"
echo -e "  Strategies:       ${GREEN}${#STRATEGIES[@]}${NC}"
echo -e "  Param Stag Pat.:  ${GREEN}${PARAM_STAGNATION_PATIENCE}${NC}" 
echo -e "  Param Tolerance:  ${GREEN}${PARAM_TOLERANCE}${NC}" 
echo ""
echo -e "${CYAN}Strategies to optimize:${NC}"
for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME _ <<< "$strategy_info"
    echo -e "  • ${STRATEGY_NAME}"
done
echo ""

# Confirmation prompt
if [ "$AUTO_YES" -eq 0 ]; then
    read -p "Continue with optimization? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Optimization cancelled.${NC}"
        exit 0
    fi
    echo ""
else
    echo -e "${YELLOW}Auto-confirm enabled (--yes). Proceeding without prompt.${NC}"
fi

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
    echo -e "${BOLD}${CYAN}[${STRATEGY_NUM}/${TOTAL_STRATEGIES}] Walk-Forward Optimization: ${GREEN}${STRATEGY_NAME}${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Trials: ${N_TRIALS} | Jobs: ${N_JOBS} | Folds: ${N_FOLDS} | Validation: ${VALIDATION_RATIO}"
    echo ""
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
        echo ""
        FAILED=$((FAILED + 1))
        FAILED_STRATEGIES+=("$STRATEGY_NAME (config not found)")
        continue
    fi
    
    # Run Bayesian walk-forward optimization
    STRATEGY_START=$(date +%s)
    
    python scripts/optimize_bayesian_walkforward.py \
        --strategy "$STRATEGY_NAME" \
        --n_trials "$N_TRIALS" \
        --n_jobs "$N_JOBS" \
        --n_folds "$N_FOLDS" \
        --validation_ratio "$VALIDATION_RATIO" \
        --reward "$REWARD" \
        --param_stagnation_patience "$PARAM_STAGNATION_PATIENCE" \
        --param_tolerance "$PARAM_TOLERANCE"
    
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
        BEST_JSON="results/${STRATEGY_NAME}/optimizations/bayesian_wf_optimization_results_best.json"
        if [ -f "$BEST_JSON" ]; then
            echo -e "${CYAN}  Best train reward:${NC} $(python -c "import json; print(f\"{json.load(open('$BEST_JSON'))['train_reward']:.6f}\")" 2>/dev/null || echo "N/A")"
            echo -e "${CYAN}  Validation reward:${NC} $(python -c "import json; r=json.load(open('$BEST_JSON'))['validation_reward']; print(f'{r:.6f}' if r != float('-inf') and r != float('inf') else 'N/A')" 2>/dev/null || echo "N/A")"
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
    CSV_FILE="$RESULTS_DIR/bayesian_wf_optimization_results.csv"
    JSON_FILE="$RESULTS_DIR/bayesian_wf_optimization_results_best.json"
    VAL_FILE="$RESULTS_DIR/bayesian_wf_optimization_results_validation.json"
    
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
echo -e "  1. Compare walk-forward results across all strategies:"
echo -e "     ${CYAN}python scripts/compare_bayesian_results.py${NC}"
echo ""
echo -e "  2. Review validation details for a specific strategy:"
echo -e "     ${CYAN}cat results/<strategy>/optimizations/bayesian_wf_optimization_results_validation.json${NC}"
echo ""
echo -e "  3. Plot validation equity curve:"
echo -e "     ${CYAN}python scripts/plot_results.py --input results/<strategy>/optimizations/bayesian_wf_optimization_results_validation.json --show${NC}"
echo ""
echo -e "  4. Run full backtest on best strategy:"
echo -e "     ${CYAN}./run_full_backtest.sh <strategy_name>${NC}"
echo ""

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Completed with ${FAILED} failure(s).${NC}"
    exit 1
else
    echo -e "${GREEN}${BOLD}All optimizations completed successfully! 🎉${NC}"
    exit 0
fi
