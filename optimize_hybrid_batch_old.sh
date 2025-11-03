#!/bin/bash

# Batch Hybrid Optimization: Random + TPE with Walk-Forward Validation
# Runs for all strategies with intelligent two-phase optimization

# Default values
RANDOM_SAMPLES=30
TPE_TRIALS=50
WALK_FORWARD=false
N_FOLDS=3
N_JOBS=1

# Prefer project venv Python if present
VENV_PY="$(pwd)/python-3.12.4/bin/python3.12"
if [ -x "$VENV_PY" ]; then
    PY="$VENV_PY"
else
    PY="python"
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --random_samples)
            RANDOM_SAMPLES="$2"
            shift 2
            ;;
        --tpe_trials)
            TPE_TRIALS="$2"
            shift 2
            ;;
        --walk-forward)
            WALK_FORWARD=true
            shift
            ;;
        --n-folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --random_samples N    Number of random samples in Phase 1 (default: 30)"
            echo "  --tpe_trials N        Number of TPE trials in Phase 2 (default: 50)"
            echo "  --walk-forward        Enable walk-forward validation (default: disabled)"
            echo "  --n-folds N           Number of folds for walk-forward (default: 3)"
            echo "  --n_jobs N            Number of parallel workers (default: 1, use 4 for speed)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                                    # Use defaults"
            echo "  $0 --random_samples 20 --tpe_trials 40               # Custom samples"
            echo "  $0 --random_samples 30 --tpe_trials 50 --walk-forward --n_jobs 4"
            echo "  $0 --walk-forward --n-folds 5 --n_jobs 4            # Fast 5-fold validation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

STRATEGIES=(
    # Moving average families
    "functions/configs/moving_average_cross.yaml"
    "functions/configs/ema_crossover.yaml"
    "functions/configs/triple_moving_average.yaml"

    # RSI / oscillators
    "functions/configs/rsi_strategy.yaml"
    "functions/configs/rsi_ma_filter.yaml"
    "functions/configs/stochastic_oscillator.yaml"

    # Bands / channels
    "functions/configs/bollinger_breakout.yaml"
    "functions/configs/bollinger_mean_reversion.yaml"
    "functions/configs/donchian_breakout.yaml"
    "functions/configs/keltner_channel_breakout.yaml"
    "functions/configs/supertrend_strategy.yaml"
    "functions/configs/atr_channel_breakout.yaml"

    # Momentum / trend
    "functions/configs/macd_momentum.yaml"
    "functions/configs/macd_zero_cross.yaml"
    "functions/configs/roc_momentum.yaml"
    "functions/configs/adx_trend.yaml"
    "functions/configs/ichimoku_kijun_cross.yaml"
    "functions/configs/heikin_ashi_trend.yaml"

    # Mean reversion / others
    "functions/configs/zscore_mean_reversion.yaml"
    "functions/configs/cci_strategy.yaml"

    # Confluence
    "functions/configs/multi_indicator_confluence.yaml"
)

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    BATCH HYBRID OPTIMIZATION                                 ║"
echo "║              Random Exploration + TPE Refinement                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Strategy: Hybrid (Random → TPE)"
echo "Settings:"
echo "  Phase 1 - Random Exploration:"
echo "    • Samples: $RANDOM_SAMPLES"
echo "    • Goal: Broadly explore parameter space"
echo ""
echo "  Phase 2 - TPE Refinement:"
echo "    • Trials: $TPE_TRIALS"
echo "    • Goal: Gradient-based refinement around best regions"
echo ""
echo "  Parallel Processing:"
echo "    • Workers: $N_JOBS"
if [ "$N_JOBS" -gt 1 ]; then
    echo "    • Speed boost: ~${N_JOBS}x faster"
else
    echo "    • Tip: Use --n_jobs 4 for faster execution"
fi
echo ""
echo "  Walk-Forward Validation:"
if [ "$WALK_FORWARD" = true ]; then
    echo "    • Enabled: Yes"
    echo "    • Folds: $N_FOLDS"
    echo "    • Anti-overfitting: Maximum"
else
    echo "    • Enabled: No (faster but less reliable)"
fi
echo ""
echo "Total evaluations per strategy: $(($RANDOM_SAMPLES + $TPE_TRIALS))"
if [ "$WALK_FORWARD" = true ]; then
    echo "Expected time per strategy: ~$(($RANDOM_SAMPLES * 2 + $TPE_TRIALS * 2 / 60))-$(($RANDOM_SAMPLES * 3 + $TPE_TRIALS * 3 / 60)) minutes"
else
    echo "Expected time per strategy: ~$(($RANDOM_SAMPLES / 5 + $TPE_TRIALS / 5))-$(($RANDOM_SAMPLES / 4 + $TPE_TRIALS / 4)) minutes"
fi
echo ""
echo "This is the RECOMMENDED approach for production-quality parameters!"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

# Track start time
start_time=$(date +%s)
success_count=0
failed_count=0
failed_strategies=()

for config in "${STRATEGIES[@]}"; do
    strategy_name=$(basename "$config" .yaml)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Strategy: $strategy_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    strategy_start=$(date +%s)
    
    # Build command with parameters
    cmd="$PY scripts/hybrid_optimize.py \
        --strategy_config \"$config\" \
        --main_config configs/main_config.yaml \
        --random_samples $RANDOM_SAMPLES \
        --tpe_trials $TPE_TRIALS \
        --n_jobs $N_JOBS"
    
    # Add walk-forward if enabled
    if [ "$WALK_FORWARD" = true ]; then
        cmd="$cmd --walk-forward"
    fi
    
    # Execute the command
    eval $cmd
    
    exit_code=$?
    strategy_end=$(date +%s)
    strategy_duration=$((strategy_end - strategy_start))
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ $strategy_name completed successfully in $strategy_duration seconds"
        ((success_count++))
    else
        echo ""
        echo "✗ $strategy_name failed with exit code $exit_code"
        ((failed_count++))
        failed_strategies+=("$strategy_name")
    fi
    echo ""
done

# Calculate total time
end_time=$(date +%s)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))
seconds=$((total_duration % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         BATCH OPTIMIZATION COMPLETE                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  ✓ Successful: $success_count strategies"
echo "  ✗ Failed: $failed_count strategies"
echo ""
if [ $failed_count -gt 0 ]; then
    echo "Failed strategies:"
    for strategy in "${failed_strategies[@]}"; do
        echo "  - $strategy"
    done
    echo ""
fi
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "  1. Compare results:"
echo "     python scripts/compare_strategies.py"
echo ""
echo "  2. Check for overfitting reduction:"
echo "     python scripts/compare_walk_forward.py"
echo ""
echo "  3. Generate full backtest:"
echo "     ./run_full_backtest.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
