# Bayesian Optimization Implementation Summary

## What Was Built

A complete Bayesian optimization system for trading strategy parameter tuning using Optuna's TPE (Tree-structured Parzen Estimator) sampler.

## Files Created/Modified

### New Files
1. **`scripts/optimize_bayesian.py`** (442 lines)
   - Main Bayesian optimization script
   - Custom reward metric implementation
   - Parallel execution support
   - Automatic train/validation split
   - CSV and JSON output generation

2. **`BAYESIAN_OPTIMIZATION.md`**
   - Complete documentation
   - Usage examples
   - Workflow comparisons
   - Troubleshooting guide

### Modified Files
1. **`configs/main_config.yaml`**
   - Added `validation_ratio: 0.2` parameter

2. **All strategy configs** (24 files in `functions/configs/`)
   - Added `parameters_bayesian` section to each
   - Defined continuous parameter ranges for Bayesian search

3. **`HOW-TO.txt`**
   - Added Bayesian optimization examples at the top
   - Marked as "Recommended!" method

## Key Features

### 1. Custom Reward Metric
```python
reward = mean_sharpe × stability - 0.1 × drawdown
```
- Emphasizes consistency over raw returns
- Splits trades into 5 chunks and calculates Sharpe for each
- Rewards low variance in Sharpe across chunks
- Penalizes excessive drawdowns

### 2. Intelligent Parameter Search
- Uses Optuna's TPE sampler (Bayesian optimization)
- Learns from past trials to suggest better parameters
- Explores continuous parameter spaces
- All parameters are integers (as required)

### 3. Automatic Validation
- Holds out 20% of data (configurable via `validation_ratio`)
- Trains on 80%, validates on 20%
- Saves both training and validation metrics
- Includes full equity curve for validation

### 4. Parallel Execution
- Supports multi-core optimization via `--n_jobs`
- Progress bar shows best reward and trial count
- Handles failures gracefully

### 5. Comprehensive Outputs
- **CSV**: All trials with full metrics
- **Best JSON**: Parameters and metrics of best trial
- **Validation JSON**: Training vs validation comparison with equity curve

## Usage Examples

### Basic Usage
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs 4
```

### Custom Sort Metric
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200 \
  --sort-by sharpe
```

### Parallel Optimization
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/multi_indicator_confluence.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 500 \
  --n_jobs 8
```

## Output Structure

### CSV File
Location: `results/<strategy>/optimizations/bayesian_optimization_results.csv`

Columns include:
- `param_*`: All parameter values tested
- `reward_metric`: Custom reward score
- `total_return_pct`: Percentage return
- `sharpe`, `sortino`, `calmar`: Risk-adjusted metrics
- `max_drawdown_pct`, `avg_drawdown_pct`: Drawdown metrics
- `trades`: Number of trades executed
- `consistency_score`: Overall consistency metric
- Full backtest results (equity_curve, trades_detail, etc.)

### Best Parameters JSON
Location: `results/<strategy>/optimizations/bayesian_optimization_results_best.json`

```json
{
  "method": "bayesian",
  "n_trials": 100,
  "sort_by": "reward_metric",
  "params": {
    "rsi_period": 14,
    "oversold": 28,
    "overbought": 72,
    "take_profit_pips": 35,
    "stop_loss_pips": 18
  },
  "metrics": {
    "reward_metric": 0.0523,
    "total_return_pct": 45.2,
    "sharpe": 1.85,
    "max_drawdown_pct": 12.3,
    "trades": 156
  }
}
```

### Validation Results JSON
Location: `results/<strategy>/optimizations/bayesian_optimization_results_validation.json`

```json
{
  "params": { ... },
  "train_metrics": {
    "reward_metric": 0.0523,
    "total_return_pct": 45.2,
    "sharpe": 1.85
  },
  "validation_metrics": {
    "reward_metric": 0.0498,
    "total_return_pct": 42.1,
    "sharpe": 1.72
  },
  "equity_curve": [
    {"time": "2020-01-01T00:00:00", "equity": 10000},
    ...
  ]
}
```

## Parameter Configuration

Each strategy config now has two parameter sections:

### `parameters` (Grid/Random Search)
Discrete values for grid/random search:
```yaml
parameters:
  rsi_period: [9, 14, 21, 28]
  oversold: [20, 25, 30, 35]
```

### `parameters_bayesian` (New!)
Continuous ranges for Bayesian search:
```yaml
parameters_bayesian:
  rsi_period: [9, 28]        # Search from 9 to 28
  oversold: [20, 35]         # Search from 20 to 35
```

**Note**: All parameters are integers. For float values, use integers and divide in strategy:
```yaml
parameters_bayesian:
  bb_std_dev: [15, 30]  # Represents 1.5-3.0 when divided by 10
```

## Advantages Over Existing Methods

### vs Grid Search
- **Faster**: Finds good parameters with fewer trials
- **Smarter**: Learns from past trials
- **Scalable**: Handles large parameter spaces efficiently

### vs Random Search
- **More efficient**: Focuses on promising regions
- **Better convergence**: Reliably finds good solutions
- **Adaptive**: Adjusts search based on results

### vs TPE in optimize_strategy.py
- **Custom reward metric**: Emphasizes consistency and stability
- **Automatic validation**: Built-in train/test split
- **Cleaner output**: Dedicated files for Bayesian results
- **Better naming**: Clear distinction from other methods

## Testing

Tested with ADXTrend strategy:
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/my_test_strat.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 5 \
  --n_jobs 1
```

Results:
- ✅ Successfully completed 5 trials
- ✅ Generated CSV with all results
- ✅ Saved best parameters JSON
- ✅ Evaluated on validation set
- ✅ Saved validation results JSON
- ✅ No errors or warnings

## Integration with Existing Workflow

The Bayesian optimizer integrates seamlessly:

1. **Standalone**: Run independently for quick optimization
2. **Compare**: Results saved in same directory structure
3. **Evaluate**: Use existing `evaluate_strategy.py` for full backtest
4. **Plot**: Use existing `plot_results.py` for visualization

## Recommended Workflow

1. **Initial exploration** (Bayesian, 100-200 trials)
   ```bash
   python scripts/optimize_bayesian.py \
     --strategy_config functions/configs/rsi_strategy.yaml \
     --main_config configs/main_config.yaml \
     --n_trials 200 \
     --n_jobs 4
   ```

2. **Check validation results**
   ```bash
   cat results/RSIStrategy/optimizations/bayesian_optimization_results_validation.json
   ```

3. **If validation looks good, run full backtest**
   ```bash
   python scripts/evaluate_strategy.py \
     --strategy_config functions/configs/rsi_strategy.yaml \
     --main_config configs/main_config.yaml
   ```

4. **Compare with other methods** (optional)
   ```bash
   python scripts/compare_strategies.py
   ```

## Future Enhancements (Optional)

1. **Multi-objective optimization**: Optimize for multiple goals (return + Sharpe + drawdown)
2. **Study persistence**: Save Optuna study to SQLite for resume/visualization
3. **Hyperparameter tuning**: Optimize Optuna sampler settings
4. **Warm-start**: Seed with results from grid/random search
5. **Adaptive trial budget**: Stop early if no improvement
6. **Cross-validation**: Multiple train/validation splits

## Dependencies

All required packages are already in `requirements.txt`:
- `optuna`: Bayesian optimization framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `pyyaml`: Config parsing

No additional installations needed!
