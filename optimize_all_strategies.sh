#!/bin/bash
################################################################################
# Batch Strategy Optimization Script
################################################################################
#
# This script runs optimization for all strategies in the functions/strategies/
# folder. Each strategy's results are saved to its own results directory.
#
# Usage:
#   ./optimize_all_strategies.sh
#
# Or with custom parameters:
#   ./optimize_all_strategies.sh --method random --n_random 100 --n_jobs 8
#
################################################################################

# Default parameters (can be overridden by command line args)
METHOD="grid"
N_RANDOM=50
N_JOBS=4
MAIN_CONFIG="configs/main_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --n_random)
            N_RANDOM="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--method grid|random] [--n_random N] [--n_jobs N] [--main_config PATH]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "python-3.12.4/bin" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source python-3.12.4/bin/activate
fi

# Define strategies with their config files
# Format: "strategy_name:config_file"
STRATEGIES=(
    "MovingAverageCross:functions/configs/example_strategy.yaml"
    "RSIStrategy:functions/configs/rsi_strategy.yaml"
    "BollingerBreakout:functions/configs/bollinger_breakout.yaml"
    "MACDMomentum:functions/configs/macd_momentum.yaml"
    "StochasticOscillator:functions/configs/stochastic_oscillator.yaml"
    "MultiIndicatorConfluence:functions/configs/multi_indicator_confluence.yaml"
)

# Print configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         BATCH STRATEGY OPTIMIZATION                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Method:       ${GREEN}${METHOD}${NC}"
if [ "$METHOD" = "random" ]; then
    echo -e "  Samples:      ${GREEN}${N_RANDOM}${NC}"
fi
echo -e "  Jobs:         ${GREEN}${N_JOBS}${NC}"
echo -e "  Main Config:  ${GREEN}${MAIN_CONFIG}${NC}"
echo -e "  Strategies:   ${GREEN}${#STRATEGIES[@]}${NC}"
echo ""

# Track timing and results
START_TIME=$(date +%s)
SUCCESSFUL=0
FAILED=0
FAILED_STRATEGIES=()

# Run optimization for each strategy
for i in "${!STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME CONFIG_FILE <<< "${STRATEGIES[$i]}"
    
    STRATEGY_NUM=$((i + 1))
    TOTAL_STRATEGIES=${#STRATEGIES[@]}
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${STRATEGY_NUM}/${TOTAL_STRATEGIES}] Optimizing: ${GREEN}${STRATEGY_NAME}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Config: ${CONFIG_FILE}"
    echo ""
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
        echo ""
        FAILED=$((FAILED + 1))
        FAILED_STRATEGIES+=("$STRATEGY_NAME (config not found)")
        continue
    fi
    
    # Run optimization
    STRATEGY_START=$(date +%s)
    
    if [ "$METHOD" = "random" ]; then
        python scripts/optimize_strategy.py \
            --strategy_config "$CONFIG_FILE" \
            --main_config "$MAIN_CONFIG" \
            --method random \
            --n_random "$N_RANDOM" \
            --n_jobs "$N_JOBS" \
            --sort-by consistency_score
    else
        python scripts/optimize_strategy.py \
            --strategy_config "$CONFIG_FILE" \
            --main_config "$MAIN_CONFIG" \
            --method grid \
            --n_jobs "$N_JOBS" \
            --sort-by consistency_score
    fi
    
    EXIT_CODE=$?
    STRATEGY_END=$(date +%s)
    STRATEGY_DURATION=$((STRATEGY_END - STRATEGY_START))
    
    # Check if optimization was successful
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ ${STRATEGY_NAME} completed successfully${NC} (${STRATEGY_DURATION}s)"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo ""
        echo -e "${RED}✗ ${STRATEGY_NAME} failed with exit code ${EXIT_CODE}${NC}"
        FAILED=$((FAILED + 1))
        FAILED_STRATEGIES+=("$STRATEGY_NAME (exit code $EXIT_CODE)")
    fi
    
    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Print summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         OPTIMIZATION SUMMARY                                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Successful:${NC} ${SUCCESSFUL}/${TOTAL_STRATEGIES}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}✗ Failed:${NC}     ${FAILED}/${TOTAL_STRATEGIES}"
    echo ""
    echo -e "${RED}Failed strategies:${NC}"
    for failed in "${FAILED_STRATEGIES[@]}"; do
        echo -e "  - $failed"
    done
fi
echo ""
echo -e "${YELLOW}Total time:${NC} ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Show results directories
echo -e "${BLUE}Results saved to:${NC}"
for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r STRATEGY_NAME _ <<< "$strategy_info"
    RESULTS_DIR="results/${STRATEGY_NAME}"
    if [ -d "$RESULTS_DIR" ]; then
        echo -e "  ${GREEN}✓${NC} ${RESULTS_DIR}/"
    else
        echo -e "  ${RED}✗${NC} ${RESULTS_DIR}/ (not found)"
    fi
done
echo ""

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All optimizations completed successfully!${NC}"
    exit 0
fi
