#!/bin/bash

# Run reduced optimization with walk-forward validation to reduce overfitting
# This uses 50 samples (instead of 100) and walk-forward cross-validation

STRATEGIES=(
    "functions/configs/moving_average_cross.yaml"
    "functions/configs/rsi_strategy.yaml"
    "functions/configs/bollinger_breakout.yaml"
    "functions/configs/macd_momentum.yaml"
    "functions/configs/stochastic_oscillator.yaml"
    "functions/configs/multi_indicator_confluence.yaml"
)

echo "=================================================="
echo "WALK-FORWARD OPTIMIZATION (Anti-Overfitting Mode)"
echo "=================================================="
echo ""
echo "Settings:"
echo "  - Sample size: 50 (reduced from 100)"
echo "  - Walk-forward folds: 5"
echo "  - Validation weight: 0.3"
echo "  - Method: random sampling"
echo ""
echo "This will take longer per strategy due to cross-validation"
echo "but should produce more robust, generalizable parameters."
echo ""

for config in "${STRATEGIES[@]}"; do
    strategy_name=$(basename "$config" .yaml)
    echo "--------------------------------------------"
    echo "Optimizing: $strategy_name"
    echo "--------------------------------------------"
    
    python scripts/optimize_strategy.py \
        --strategy_config "$config" \
        --main_config configs/main_config.yaml \
        --method random \
        --n_random 50 \
        --walk-forward \
        --n-folds 5 \
        --validation-weight 0.3 \
        --sort-by consistency_score \
        --n_jobs 4
    
    if [ $? -eq 0 ]; then
        echo "✓ $strategy_name completed successfully"
    else
        echo "✗ $strategy_name failed"
    fi
    echo ""
done

echo "=================================================="
echo "Walk-forward optimization complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Compare results with: python scripts/compare_strategies.py"
echo "2. Check for reduced train/test gaps"
echo "3. Verify improved generalization"
