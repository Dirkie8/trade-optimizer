#!/bin/bash
# Quick reference for Bayesian optimization commands

# ============================================================================
# BAYESIAN OPTIMIZATION - QUICK REFERENCE
# ============================================================================

# Basic usage (recommended starting point)
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs 4

# ============================================================================
# EXAMPLES BY STRATEGY COMPLEXITY
# ============================================================================

# Simple strategies (2-4 parameters) - 50-100 trials
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/adx_trend.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 50 \
  --n_jobs 4

# Medium strategies (5-7 parameters) - 100-300 trials
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_ma_filter.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200 \
  --n_jobs 8

# Complex strategies (8+ parameters) - 300-1000 trials
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/multi_indicator_confluence.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 500 \
  --n_jobs 8

# ============================================================================
# DIFFERENT SORT METRICS
# ============================================================================

# Default: reward_metric (consistency + stability - drawdown)
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100

# Sort by Sharpe ratio
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --sort-by sharpe

# Sort by total return
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --sort-by total_return_pct

# ============================================================================
# PARALLEL EXECUTION OPTIONS
# ============================================================================

# Single core (debugging)
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs 1

# Auto-detect cores
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs -1

# Specific number of cores
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs 8

# ============================================================================
# OUTPUT LOCATIONS
# ============================================================================

# CSV results:
# results/<strategy>/optimizations/bayesian_optimization_results.csv

# Best parameters:
# results/<strategy>/optimizations/bayesian_optimization_results_best.json

# Validation results:
# results/<strategy>/optimizations/bayesian_optimization_results_validation.json

# ============================================================================
# CHECKING RESULTS
# ============================================================================

# View best parameters
cat results/RSIStrategy/optimizations/bayesian_optimization_results_best.json

# View validation comparison
cat results/RSIStrategy/optimizations/bayesian_optimization_results_validation.json

# View CSV summary (top 5 results)
head -n 6 results/RSIStrategy/optimizations/bayesian_optimization_results.csv | column -t -s,

# ============================================================================
# FULL WORKFLOW EXAMPLE
# ============================================================================

# 1. Run Bayesian optimization
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200 \
  --n_jobs 4

# 2. Check validation results
cat results/RSIStrategy/optimizations/bayesian_optimization_results_validation.json

# 3. If validation looks good, evaluate on full dataset
python scripts/evaluate_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml

# 4. Plot results
python scripts/plot_results.py \
  --input results/RSIStrategy/full_dataset_backtest.json \
  --show

# ============================================================================
# BATCH OPTIMIZATION (all strategies)
# ============================================================================

# Create batch script
cat > optimize_bayesian_batch.sh << 'EOF'
#!/bin/bash
for config in functions/configs/*.yaml; do
    strategy_name=$(basename "$config" .yaml)
    echo "Optimizing $strategy_name..."
    python scripts/optimize_bayesian.py \
        --strategy_config "$config" \
        --main_config configs/main_config.yaml \
        --n_trials 100 \
        --n_jobs 4
done
EOF

chmod +x optimize_bayesian_batch.sh
./optimize_bayesian_batch.sh
