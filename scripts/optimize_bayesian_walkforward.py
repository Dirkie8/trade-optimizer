#!/usr/bin/env python3
"""
Bayesian optimization with walk-forward validation for trading strategies.

This combines:
1. Bayesian optimization (Optuna TPE) for efficient parameter search
2. Walk-forward validation on training set (reduces overfitting)
3. Hold-out validation set (20%) for final assessment
4. Custom reward metric emphasizing consistency

Example:
  python scripts/optimize_bayesian_walkforward.py \
    --strategy RSIStrategy \
    --n_trials 100 \
    --n_jobs 4 \
    --n_folds 5
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

# Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy

# ANSI colors
if os.environ.get("NO_COLOR"):
    RESET = BOLD = CYAN = GREEN = YELLOW = MAGENTA = RED = ""
else:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"


def max_drawdown_from_equity(equity_curve: List[Dict[str, Any]]) -> float:
    """Calculate max drawdown from equity curve list."""
    if not equity_curve:
        return 0.0
    eq = np.array([pt["equity"] for pt in equity_curve], dtype=float)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = np.divide(peaks - eq, peaks, out=np.zeros_like(eq, dtype=float), where=peaks != 0)
    return float(dd.max() if dd.size else 0.0)


def calculate_reward_metric(
    trades_df: pd.DataFrame,
    equity_curve: List[Dict[str, Any]],
    metrics: Dict[str, float],
    reward_type: str = "balanced"
) -> float:
    """
    Calculate reward metric based on chosen strategy.
    
    Args:
        trades_df: DataFrame with trade details
        equity_curve: List of equity curve points
        metrics: Dict of calculated metrics (sharpe, sortino, etc.)
        reward_type: Type of reward calculation
            - "balanced": mean_sharpe * stability - drawdown_penalty
            - "consistency": Emphasizes low variance in returns
            - "sharpe": Simple Sharpe ratio
            - "sortino": Sortino ratio
            - "calmar": Calmar ratio
            
    Returns:
        Reward score (higher is better)
    """
    if trades_df.empty or len(trades_df) < 5:
        return -np.inf
    
    if reward_type == "sharpe":
        return metrics.get("sharpe", -np.inf)
    
    elif reward_type == "sortino":
        return metrics.get("sortino", -np.inf)
    
    elif reward_type == "calmar":
        return metrics.get("calmar", -np.inf)
    
    elif reward_type == "consistency":
        # Split into chunks and measure consistency
        pnl = trades_df['pnl'].values
        if np.all(pnl == 0):
            return -np.inf
        
        chunks = np.array_split(pnl, min(5, len(pnl)))
        chunk_returns = [np.sum(chunk) for chunk in chunks if len(chunk) > 0]
        
        if not chunk_returns:
            return -np.inf
        
        mean_ret = np.mean(chunk_returns)
        std_ret = np.std(chunk_returns)
        
        # Reward high mean, low variance
        if std_ret == 0:
            return mean_ret
        
        consistency_score = mean_ret / (std_ret + 1e-6)
        return consistency_score
    
    else:  # "balanced" (default)
        pnl = trades_df['pnl'].values
        if np.all(pnl == 0):
            return -np.inf
        
        # Split into 5 chunks and compute Sharpe for each
        chunks = np.array_split(pnl, min(5, len(pnl)))
        sharpes = []
        for chunk in chunks:
            if len(chunk) > 0 and np.std(chunk) > 0:
                sharpes.append(np.mean(chunk) / np.std(chunk))
        
        if not sharpes:
            return -np.inf
        
        # Stability: lower variance of Sharpe across chunks is better
        sharpes = np.array(sharpes)
        mean_sharpe = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        stability = 1.0 - min(sharpe_std, 1.0)
        
        # Drawdown penalty
        drawdown = max_drawdown_from_equity(equity_curve)
        
        # Combined reward (more weight on drawdown than before)
        reward = mean_sharpe * stability - 0.2 * drawdown
        return float(reward)


def infer_max_lookback(strategy_name: str, params: Dict[str, Any]) -> int:
    """Infer max lookback from strategy parameters."""
    period_keys = [k for k in params.keys() if 'period' in k.lower() or 'window' in k.lower() or 'ema' in k.lower()]
    if period_keys:
        periods = [int(params[k]) for k in period_keys if isinstance(params.get(k), (int, float))]
        if periods:
            return max(periods) + 50
    
    if "Moving Average" in strategy_name or "MA" in strategy_name or "EMA" in strategy_name:
        ma_keys = [k for k in params.keys() if 'ma' in k.lower()]
        if ma_keys:
            return max([int(params[k]) for k in ma_keys if isinstance(params.get(k), (int, float))], default=200) + 50
    
    return 200


def run_walk_forward_backtest(
    params: Dict[str, Any],
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
    n_folds: int = 5,
    reward_type: str = "balanced"
) -> Dict[str, Any]:
    """
    Run walk-forward validation on parameter set.
    
    Splits data into n_folds:
    - Fold 1: Skip (no training data)
    - Fold 2: Train on fold 1, test on fold 2
    - Fold 3: Train on folds 1-2, test on fold 3
    - ... and so on
    
    Returns average metrics across all folds.
    """
    max_lb = infer_max_lookback(strategy_name, params)
    n = len(data)
    fold_size = n // n_folds
    
    if fold_size < 100:
        # Data too small for walk-forward, use simple backtest
        print(f"  Warning: Data too small for {n_folds} folds, using single backtest")
        result = backtest_strategy(
            data=data,
            symbol=symbol,
            strategy_cls=StrategyCls,
            params=params,
            account_cfg=account_cfg,
            max_lookback=max_lb,
            progress=None,
        )
        
        # Add reward metric
        trades_df = pd.DataFrame(result.get('trades', []))
        result['reward_metric'] = calculate_reward_metric(
            trades_df,
            result.get('equity_curve', []),
            result,
            reward_type
        )
        return result
    
    # Collect metrics from each fold
    fold_results = []
    
    for fold in range(1, n_folds + 1):
        # Training: all data up to end of (fold-1)
        # Testing: data in current fold
        train_end = fold_size * (fold - 1)
        test_start = train_end
        test_end = fold_size * fold
        
        if fold == 1:
            # First fold: no training data, skip
            continue
        
        # We test on the fold, but we don't actually use train_data
        # (parameters are already given). This is anchored walk-forward.
        test_data = data.iloc[test_start:test_end]
        
        if len(test_data) < max_lb:
            continue
        
        try:
            result = backtest_strategy(
                data=test_data,
                symbol=symbol,
                strategy_cls=StrategyCls,
                params=params,
                account_cfg=account_cfg,
                max_lookback=max_lb,
                progress=None,
            )
            
            # Calculate reward for this fold
            # Prefer 'trades_detail' list; fallback to 'trades' only if it's a list
            trades_list = result.get('trades_detail', [])
            if not trades_list:
                t_alt = result.get('trades', [])
                trades_list = t_alt if isinstance(t_alt, list) else []
            if trades_list:
                # Convert to DataFrame from dicts or Trade objects
                if hasattr(trades_list[0], '__dict__'):
                    trades_df = pd.DataFrame([{k: v for k, v in t.__dict__.items()} for t in trades_list])
                else:
                    trades_df = pd.DataFrame(trades_list)
            else:
                trades_df = pd.DataFrame()
            
            result['reward_metric'] = calculate_reward_metric(
                trades_df,
                result.get('equity_curve', []),
                result,
                reward_type
            )
            
            fold_results.append(result)
        except Exception as e:
            # Skip failed folds
            print(f"  Fold {fold} failed: {e}")
            continue
    
    if not fold_results:
        # All folds failed
        return {
            'reward_metric': -np.inf,
            'sharpe': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 100,
            'trades': 0,
        }
    
    # Average metrics across folds
    avg_result = {}
    metric_keys = ['sharpe', 'sortino', 'calmar', 
                   'total_return_pct', 'max_drawdown_pct', 'trades', 
                   'win_rate_pct']
    
    for key in metric_keys:
        values = [r.get(key, 0) for r in fold_results]
        avg_result[key] = np.mean(values) if values else 0.0
    
    # Add fold consistency metrics
    avg_result['n_folds'] = len(fold_results)
    avg_result['sharpe_std'] = np.std([r.get('sharpe', 0) for r in fold_results])
    avg_result['return_std'] = np.std([r.get('total_return_pct', 0) for r in fold_results])
    
    # Calculate overall reward_metric from averaged fold metrics
    # Use average reward from folds (since we can't reconstruct trade-level data)
    reward_values = [r.get('reward_metric', -np.inf) for r in fold_results]
    valid_rewards = [r for r in reward_values if np.isfinite(r)]
    
    if valid_rewards:
        avg_result['reward_metric'] = np.mean(valid_rewards)
    else:
        # If no valid rewards, try calculating from averaged metrics
        # This is a fallback using averaged Sharpe and other metrics
        sharpe = avg_result.get('sharpe', 0)
        drawdown = avg_result.get('max_drawdown_pct', 0) / 100.0
        sharpe_std = avg_result.get('sharpe_std', 0)
        stability = 1.0 - min(sharpe_std / (abs(sharpe) + 0.1), 1.0)  # Relative std
        avg_result['reward_metric'] = sharpe * stability - 0.2 * drawdown
    
    # Keep last fold's equity curve for reference
    avg_result['equity_curve'] = fold_results[-1].get('equity_curve', [])
    
    return avg_result


def run_bayesian_optimization(
    strategy_name: str,
    strategy_config_path: Path,
    main_config_path: Path,
    n_trials: int = 100,
    n_jobs: int = 1,
    n_folds: int = 5,
    validation_ratio: float = 0.2,
    reward_type: str = "balanced",
    seed: int = 42,
    param_stagnation_patience: int = 0,
    param_tolerance: float = 0.0,
    top_k_validation: int = 0,
):
    """
    Run Bayesian optimization with walk-forward validation.
    
    Process:
    1. Split data into train (80%) and validation (20%)
    2. On training data, run walk-forward validation for each trial
    3. Optimize parameters based on average walk-forward performance
    4. Test best parameters on hold-out validation set
    """
    print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}Bayesian Optimization with Walk-Forward Validation{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")
    
    # Load configurations
    with open(strategy_config_path, 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Get parameter space for Bayesian optimization
    if 'parameters_bayesian' not in strategy_config:
        print(f"{RED}ERROR: No 'parameters_bayesian' section found in {strategy_config_path}{RESET}")
        print("Add a section like:")
        print("parameters_bayesian:")
        print("  rsi_period: [10, 50]")
        print("  oversold: [20, 40]")
        sys.exit(1)
    
    param_ranges = strategy_config['parameters_bayesian']
    # Log parameter space to confirm parameters_bayesian usage
    try:
        print(f"\n{YELLOW}Parameter space (parameters_bayesian) from {strategy_config_path}:{RESET}")
        for k, v in param_ranges.items():
            print(f"  - {k}: {v}")
    except Exception:
        pass
    
    # Load strategy class
    strategy_module_name = strategy_config.get('strategy', {}).get('module')
    strategy_class_name = strategy_config.get('strategy', {}).get('class')
    
    if not strategy_module_name or not strategy_class_name:
        print(f"{RED}ERROR: Strategy module/class not specified in config{RESET}")
        sys.exit(1)
    
    module = importlib.import_module(strategy_module_name)
    StrategyCls = getattr(module, strategy_class_name)
    
    # Load data
    print(f"{CYAN}Loading data...{RESET}")
    general_conf = main_config.get("general", {})
    symbol = general_conf.get("default_symbol", "EURUSD")
    timeframe = general_conf.get("default_timeframe", "1h")
    data = load_data(symbol, timeframe)
    # Confirm Volume is present and non-null for optimization
    try:
        has_vol = 'Volume' in data.columns
        nonzero = int((data['Volume'] > 0).sum()) if has_vol else 0
        print(f"{GREEN}Data columns:{RESET} {list(data.columns)}")
        print(f"{GREEN}Volume present:{RESET} {has_vol} | nonzero bars: {nonzero}")
    except Exception:
        pass
    
    print(f"Total data points: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Split into train and validation
    split_idx = int(len(data) * (1 - validation_ratio))
    train_data = data.iloc[:split_idx].copy()
    validation_data = data.iloc[split_idx:].copy()
    
    print(f"\n{YELLOW}Data Split:{RESET}")
    print(f"  Training: {len(train_data)} bars ({(1-validation_ratio)*100:.0f}%) - Walk-forward {n_folds} folds")
    print(f"  Validation: {len(validation_data)} bars ({validation_ratio*100:.0f}%) - Hold-out test")
    
    # Account configuration
    account_cfg = main_config.get('account', {})
    
    # Create Optuna study
    print(f"\n{CYAN}Initializing Bayesian optimization...{RESET}")
    print(f"  Trials: {n_trials}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Walk-forward folds: {n_folds}")
    print(f"  Reward type: {reward_type}")
    
    # Configure TPE for stable sampling and interaction modeling
    # Configure TPESampler with stable options compatible with installed Optuna
    # Note: consider_running_trials is not available in current Optuna; constant_liar handles parallelism.
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=min(10, n_trials // 5),
        multivariate=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{strategy_name}_wf_bayesian"
    )
    
    # Store all trial results
    all_results = []
    stopped_due_to_param_stagnation = False
    previous_best_params = None
    stagnation_counter = 0

    def params_equal(p_old: Dict[str, Any], p_new: Dict[str, Any], tol: float) -> bool:
        if p_old is None or p_new is None:
            return False
        if p_old.keys() != p_new.keys():
            return False
        for k in p_old.keys():
            v1 = p_old[k]
            v2 = p_new[k]
            # Numeric tolerance for floats
            if isinstance(v1, float) or isinstance(v2, float):
                if abs(float(v1) - float(v2)) > tol:
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Sample parameters with correct types: ints stay ints; floats explore continuous space
        params = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                low, high = param_range
                # Use integer sampler only when both bounds are integers
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    # Continuous float range
                    try:
                        params[param_name] = trial.suggest_float(param_name, float(low), float(high))
                    except Exception:
                        # Fallback to categorical if sampler not available
                        params[param_name] = trial.suggest_categorical(param_name, [float(low), float(high)])
            else:
                print(f"  Warning: Invalid range for {param_name}: {param_range}")
                params[param_name] = param_range
        
        # Run walk-forward backtest on training data
        try:
            result = run_walk_forward_backtest(
                params=params,
                data=train_data,
                symbol=symbol,
                StrategyCls=StrategyCls,
                account_cfg=account_cfg,
                strategy_name=strategy_name,
                n_folds=n_folds,
                reward_type=reward_type
            )
            
            # Store result
            result_row = {
                **{f'param_{k}': v for k, v in params.items()},
                **result,
                'trial_number': trial.number,
            }
            all_results.append(result_row)
            
            # Return reward metric
            reward = result.get('reward_metric', -np.inf)
            trades_count = result.get('trades', 0)
            
            # Debug: Check why reward might be -inf
            if not np.isfinite(reward) and trades_count >= 5:
                print(f"  Warning: {trades_count:.0f} trades but reward is {reward}")
            
            # Penalize if no trades
            if trades_count < 5:
                return -1e10
            
            if not np.isfinite(reward):
                return -1e10
            
            return float(reward)
            
        except Exception as e:
            print(f"  Trial failed: {e}")
            return -1e10
    
    # Run optimization with progress bar
    print(f"\n{GREEN}Starting optimization...{RESET}\n")
    with tqdm(total=n_trials, desc="Bayesian Optimization", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            best_val = study.best_value if study.best_trial else -np.inf
            pbar.set_postfix({
                'best_reward': f'{best_val:.4f}',
                'trial': trial.number + 1
            })
            # Parameter stagnation early stop (only if patience > 0)
            if param_stagnation_patience > 0 and study.best_trial is not None:
                nonlocal previous_best_params, stagnation_counter, stopped_due_to_param_stagnation
                current_best = study.best_params
                if previous_best_params is None:
                    previous_best_params = dict(current_best)
                    stagnation_counter = 0
                else:
                    if params_equal(previous_best_params, current_best, param_tolerance):
                        stagnation_counter += 1
                    else:
                        previous_best_params = dict(current_best)
                        stagnation_counter = 0
                if stagnation_counter >= param_stagnation_patience:
                    print(f"\n{YELLOW}Early stop: best parameters unchanged for {stagnation_counter} consecutive trials (patience={param_stagnation_patience}).{RESET}")
                    stopped_due_to_param_stagnation = True
                    study.stop()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
            n_jobs=n_jobs,
        )
    
    print(f"\n{GREEN}✓ Optimization complete!{RESET}")
    print(f"\n{BOLD}Best Parameters:{RESET}")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"\n{BOLD}Best Training Reward: {study.best_value:.6f}{RESET}")
    
    # Run validation on hold-out set
    print(f"\n{CYAN}Testing on hold-out validation set...{RESET}")
    max_lb = infer_max_lookback(strategy_name, study.best_params)
    
    validation_result = backtest_strategy(
        data=validation_data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=study.best_params,
        account_cfg=account_cfg,
        max_lookback=max_lb,
        progress=None,
    )
    
    # Calculate validation reward
    # Build trades_df from trades_detail primarily
    trades_list = validation_result.get('trades_detail', [])
    if not trades_list:
        t_alt = validation_result.get('trades', [])
        trades_list = t_alt if isinstance(t_alt, list) else []
    if trades_list:
        if hasattr(trades_list[0], '__dict__'):
            trades_df = pd.DataFrame([{k: v for k, v in t.__dict__.items()} for t in trades_list])
        else:
            trades_df = pd.DataFrame(trades_list)
    else:
        trades_df = pd.DataFrame()
    
    validation_reward = calculate_reward_metric(
        trades_df,
        validation_result.get('equity_curve', []),
        validation_result,
        reward_type
    )
    # Sanitize non-finite rewards and attach reason so downstream reports don't show -inf
    reward_raw = float(validation_reward) if isinstance(validation_reward, (int, float, np.number)) else -np.inf
    reward_reason = None
    reward_floored = reward_raw
    if not np.isfinite(reward_raw):
        reasons = []
        if trades_df.empty or len(trades_df) < 5:
            reasons.append("insufficient_trades")
        else:
            try:
                pnl = trades_df['pnl'].to_numpy()
                if pnl.size == 0:
                    reasons.append("no_pnl_data")
                elif np.all(pnl == 0):
                    reasons.append("all_zero_pnl")
                else:
                    # Try to infer chunk degeneracy
                    chunks = np.array_split(pnl, min(5, len(pnl)))
                    chunk_sums = [np.sum(c) for c in chunks if len(c) > 0]
                    if len(chunk_sums) == 0:
                        reasons.append("no_valid_chunks")
                    else:
                        if np.std(chunk_sums) == 0:
                            reasons.append("chunk_std_zero")
                        # else unknown non-finite source
            except Exception:
                reasons.append("reward_calc_error")
        if not reasons:
            reasons.append("non_finite_reward")
        reward_reason = ",".join(reasons)
        reward_floored = -1e10
    
    print(f"\n{BOLD}Validation Results:{RESET}")
    if reward_reason:
        print(f"  Reward: {reward_floored:.6f} (floored from {reward_raw} due to {reward_reason})")
    else:
        print(f"  Reward: {reward_floored:.6f}")
    print(f"  Return: {validation_result.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe: {validation_result.get('sharpe', 0):.4f}")
    print(f"  Max DD: {validation_result.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Trades: {validation_result.get('trades', 0)}")
    
    # Convert strategy_name to snake_case for consistent folder naming
    import re
    def to_snake_case(name: str) -> str:
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.replace('__', '_').lower()
    
    strategy_folder = to_snake_case(strategy_name)
    
    # Save results
    output_dir = Path('results') / strategy_folder / 'optimizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV of all trials
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'bayesian_wf_optimization_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n{GREEN}✓ Saved all trials to: {csv_path}{RESET}")
    
    # Identify the best trial's training metrics row from all_results
    def _extract_params_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k.replace('param_', ''): row[k] for k in row.keys() if k.startswith('param_')}

    best_row = None
    for row in all_results:
        row_params = _extract_params_from_row(row)
        if params_equal(row_params, study.best_params, param_tolerance):
            best_row = row
            break
    if best_row is None:
        # Fallback: choose row with max reward_metric (handles float rounding / tolerance mismatches)
        try:
            best_row = max(all_results, key=lambda r: r.get('reward_metric', float('-inf')))
        except Exception:
            best_row = all_results[-1] if all_results else {}

    # Prepare params (and a rounded variant for reproducibility with evaluators that round floats)
    def _round_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in p.items():
            if isinstance(v, bool) or isinstance(v, int):
                out[k] = v
            else:
                try:
                    out[k] = round(float(v), decimals)
                except Exception:
                    out[k] = v
        return out

    params_rounded = _round_params(study.best_params)

    # Save best parameters
    best_json = {
        'params': study.best_params,
        'params_rounded': params_rounded,
        'train_reward': float(study.best_value),
        'validation_reward': float(reward_floored),
        'symbol': symbol,
        'timeframe': timeframe,
        'account_cfg': account_cfg,
        'data_split': {
            'train': {
                'start': train_data.index[0].isoformat() if len(train_data) else None,
                'end': train_data.index[-1].isoformat() if len(train_data) else None,
            },
            'validation': {
                'start': validation_data.index[0].isoformat() if len(validation_data) else None,
                'end': validation_data.index[-1].isoformat() if len(validation_data) else None,
            },
        },
        'train_metrics': {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in (best_row or {}).items()
            if not str(k).startswith('param_') and k not in ['equity_curve', 'trades']
        },
        'validation_metrics': {
            'reward_metric': float(reward_floored),
            'reward_raw': float(reward_raw) if np.isfinite(reward_raw) or reward_raw in [np.inf, -np.inf] else float('nan'),
            'reward_reason': reward_reason,
            'reward_floor_applied': bool(reward_reason is not None),
            'total_return_pct': float(validation_result.get('total_return_pct', 0)),
            'sharpe': float(validation_result.get('sharpe', 0)),
            'sortino': float(validation_result.get('sortino', 0)),
            'calmar': float(validation_result.get('calmar', 0)),
            'max_drawdown_pct': float(validation_result.get('max_drawdown_pct', 0)),
            'trades': int(validation_result.get('trades', 0)),
        },
        'early_stop': {
            'param_stagnation_enabled': param_stagnation_patience > 0,
            'param_stagnation_patience': param_stagnation_patience,
            'stopped_due_to_param_stagnation': stopped_due_to_param_stagnation,
            'param_tolerance': param_tolerance,
            'stagnation_counter_final': stagnation_counter,
        }
    }
    
    best_path = output_dir / 'bayesian_wf_optimization_results_best.json'
    with open(best_path, 'w') as f:
        json.dump(best_json, f, indent=2)
    print(f"{GREEN}✓ Saved best parameters to: {best_path}{RESET}")
    
    # Save validation details
    val_path = output_dir / 'bayesian_wf_optimization_results_validation.json'
    validation_output = {
        'params': study.best_params,
        'train_metrics': best_json['train_metrics'],
        'validation_metrics': best_json['validation_metrics'],
        'equity_curve': validation_result.get('equity_curve', []),
    }
    with open(val_path, 'w') as f:
        json.dump(validation_output, f, indent=2)
    print(f"{GREEN}✓ Saved validation details to: {val_path}{RESET}")

    # Optional: run validation on top-K trials (sorted by training reward)
    if isinstance(top_k_validation, int) and top_k_validation and top_k_validation > 1 and len(all_results) > 1:
        # Sort by reward_metric descending, take top-K
        try:
            rows_sorted = sorted(all_results, key=lambda r: r.get('reward_metric', float('-inf')), reverse=True)
        except Exception:
            rows_sorted = list(all_results)
        top_rows = rows_sorted[:min(top_k_validation, len(rows_sorted))]
        topk_validations = []
        for rank, r in enumerate(top_rows, start=1):
            # Reconstruct params from row
            try:
                params_r = {k.replace('param_', ''): r[k] for k in r.keys() if str(k).startswith('param_')}
            except Exception:
                params_r = dict(study.best_params)
            # Run validation
            try:
                max_lb_r = infer_max_lookback(strategy_name, params_r)
                val_r = backtest_strategy(
                    data=validation_data,
                    symbol=symbol,
                    strategy_cls=StrategyCls,
                    params=params_r,
                    account_cfg=account_cfg,
                    max_lookback=max_lb_r,
                    progress=None,
                )
                trades_list_r = val_r.get('trades_detail', [])
                if not trades_list_r:
                    t_alt_r = val_r.get('trades', [])
                    trades_list_r = t_alt_r if isinstance(t_alt_r, list) else []
                if trades_list_r:
                    if hasattr(trades_list_r[0], '__dict__'):
                        trades_df_r = pd.DataFrame([{k: v for k, v in t.__dict__.items()} for t in trades_list_r])
                    else:
                        trades_df_r = pd.DataFrame(trades_list_r)
                else:
                    trades_df_r = pd.DataFrame()
                val_reward_r = calculate_reward_metric(
                    trades_df_r,
                    val_r.get('equity_curve', []),
                    val_r,
                    reward_type,
                )
                # Floor non-finite rewards and capture reason (mirror best validation handling)
                reward_raw_r = float(val_reward_r) if isinstance(val_reward_r, (int, float, np.number)) else -np.inf
                reward_reason_r = None
                reward_floored_r = reward_raw_r
                if not np.isfinite(reward_raw_r):
                    reasons = []
                    if trades_df_r.empty or len(trades_df_r) < 5:
                        reasons.append("insufficient_trades")
                    else:
                        try:
                            pnlr = trades_df_r['pnl'].to_numpy()
                            if pnlr.size == 0:
                                reasons.append("no_pnl_data")
                            elif np.all(pnlr == 0):
                                reasons.append("all_zero_pnl")
                            else:
                                chunksr = np.array_split(pnlr, min(5, len(pnlr)))
                                chunk_sumsr = [np.sum(c) for c in chunksr if len(c) > 0]
                                if len(chunk_sumsr) == 0:
                                    reasons.append("no_valid_chunks")
                                else:
                                    if np.std(chunk_sumsr) == 0:
                                        reasons.append("chunk_std_zero")
                        except Exception:
                            reasons.append("reward_calc_error")
                    if not reasons:
                        reasons.append("non_finite_reward")
                    reward_reason_r = ",".join(reasons)
                    reward_floored_r = -1e10

                topk_validations.append({
                    'trial_number': r.get('trial_number'),
                    'train_reward': r.get('reward_metric'),
                    'params': params_r,
                    'validation_metrics': {
                        'reward_metric': float(reward_floored_r),
                        'reward_raw': float(reward_raw_r) if np.isfinite(reward_raw_r) or reward_raw_r in [np.inf, -np.inf] else float('nan'),
                        'reward_reason': reward_reason_r,
                        'reward_floor_applied': bool(reward_reason_r is not None),
                        'total_return_pct': float(val_r.get('total_return_pct', 0.0)),
                        'sharpe': float(val_r.get('sharpe', 0.0)),
                        'sortino': float(val_r.get('sortino', 0.0)),
                        'calmar': float(val_r.get('calmar', 0.0)),
                        'max_drawdown_pct': float(val_r.get('max_drawdown_pct', 0.0)),
                        'trades': int(val_r.get('trades', 0)),
                    },
                })

                # Write detailed per-item validation JSON with equity curve
                item_details = {
                    'trial_number': r.get('trial_number'),
                    'rank': rank,
                    'train_reward': r.get('reward_metric'),
                    'train_metrics': {k: r.get(k) for k in r.keys() if k not in ['equity_curve'] and not str(k).startswith('param_')},
                    'params': params_r,
                    'validation_metrics': {
                        'reward_metric': float(reward_floored_r),
                        'reward_raw': float(reward_raw_r) if np.isfinite(reward_raw_r) or reward_raw_r in [np.inf, -np.inf] else float('nan'),
                        'reward_reason': reward_reason_r,
                        'reward_floor_applied': bool(reward_reason_r is not None),
                        'total_return_pct': float(val_r.get('total_return_pct', 0.0)),
                        'sharpe': float(val_r.get('sharpe', 0.0)),
                        'sortino': float(val_r.get('sortino', 0.0)),
                        'calmar': float(val_r.get('calmar', 0.0)),
                        'max_drawdown_pct': float(val_r.get('max_drawdown_pct', 0.0)),
                        'trades': int(val_r.get('trades', 0)),
                    },
                    'equity_curve': val_r.get('equity_curve', []),
                }
                item_path = output_dir / f"bayesian_wf_topk_validation_item_{rank}_trial_{r.get('trial_number')}.json"
                try:
                    with open(item_path, 'w') as f:
                        json.dump(item_details, f, indent=2)
                    print(f"{GREEN}✓ Saved detailed top-{rank} validation to: {item_path}{RESET}")
                except Exception as e:
                    print(f"{YELLOW}Warning: failed to save detailed top-{rank} validation: {e}{RESET}")
            except Exception as e:
                topk_validations.append({
                    'trial_number': r.get('trial_number'),
                    'train_reward': r.get('reward_metric'),
                    'params': params_r,
                    'error': f'{e}',
                })
        # Save top-K validations JSON and CSV
        topk_path = output_dir / 'bayesian_wf_topk_validation.json'
        with open(topk_path, 'w') as f:
            json.dump({'top_k': len(topk_validations), 'items': topk_validations}, f, indent=2)
        try:
            pd.DataFrame([
                {
                    'trial_number': it.get('trial_number'),
                    'train_reward': it.get('train_reward'),
                    'validation_reward': it.get('validation_metrics', {}).get('reward_metric'),
                    'validation_return_pct': it.get('validation_metrics', {}).get('total_return_pct'),
                    'validation_sharpe': it.get('validation_metrics', {}).get('sharpe'),
                    'validation_trades': it.get('validation_metrics', {}).get('trades'),
                }
                for it in topk_validations
            ]).to_csv(output_dir / 'bayesian_wf_topk_validation.csv', index=False)
        except Exception:
            pass
        print(f"{GREEN}✓ Saved top-{len(topk_validations)} validation summary to: {topk_path}{RESET}")
    
    print(f"\n{BOLD}{GREEN}{'='*80}{RESET}")
    print(f"{BOLD}{GREEN}Optimization Complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*80}{RESET}\n")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization with walk-forward validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/optimize_bayesian_walkforward.py --strategy RSIStrategy --n_trials 50

  # High-performance overnight run
  python scripts/optimize_bayesian_walkforward.py \\
    --strategy RSIStrategy \\
    --n_trials 200 \\
    --n_jobs 4 \\
    --n_folds 5 \\
    --reward balanced

  # Use 15-minute data with more folds
  python scripts/optimize_bayesian_walkforward.py \\
    --strategy RSIStrategy \\
    --n_trials 100 \\
    --n_folds 8 \\
    --validation_ratio 0.15
        """
    )
    
    parser.add_argument('--strategy', required=True, help='Strategy name (e.g., RSIStrategy)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Bayesian trials (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs (default: 1, use 4+ for speed)')
    parser.add_argument('--n_folds', type=int, default=5, help='Walk-forward folds on training set (default: 5)')
    parser.add_argument('--validation_ratio', type=float, default=0.2, 
                       help='Hold-out validation ratio (default: 0.2)')
    parser.add_argument('--reward', choices=['balanced', 'consistency', 'sharpe', 'sortino', 'calmar'],
                       default='balanced', help='Reward metric type (default: balanced)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--param_stagnation_patience', type=int, default=0, help='Early stop if best params unchanged for this many consecutive trials (0 disables).')
    parser.add_argument('--param_tolerance', type=float, default=0.0, help='Tolerance for float parameter equality when detecting stagnation (default: 0 exact match).')
    parser.add_argument('--top_k_validation', type=int, default=0, help='Validate top-k trials on the hold-out set (0 disables; e.g., 5 or 10).')
    
    args = parser.parse_args()
    
    # Construct file paths
    strategy_config = Path('functions/configs') / f"{args.strategy.lower().replace('strategy', '_strategy')}.yaml"
    if not strategy_config.exists():
        # Try without underscore
        strategy_config = Path('functions/configs') / f"{args.strategy.lower()}.yaml"
    if not strategy_config.exists():
        # Try snake_case conversion from CamelCase (e.g., ADXTrend -> adx_trend)
        import re
        def to_snake(name: str) -> str:
            # Handle acronym + word (e.g., ADXTrend -> ADX_Trend)
            name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
            # Handle camel transitions (e.g., RsiStrategy -> Rsi_Strategy)
            name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
            return name.replace('__', '_').lower()
        snake = to_snake(args.strategy)
        alt1 = Path('functions/configs') / f"{snake}.yaml"
        alt2 = Path('functions/configs') / f"{snake.replace('strategy', '_strategy')}.yaml"
        if alt1.exists():
            strategy_config = alt1
        elif alt2.exists():
            strategy_config = alt2
        else:
            # Final fallback: scan configs and match by normalized stem
            cfg_dir = Path('functions/configs')
            norm_target = args.strategy.lower().replace('strategy', '').replace('_', '')
            for f in cfg_dir.glob('*.yaml'):
                stem_norm = f.stem.replace('_', '').lower()
                if stem_norm == norm_target:
                    strategy_config = f
                    break
    
    if not strategy_config.exists():
        tried = [
            f"functions/configs/{args.strategy.lower().replace('strategy', '_strategy')}.yaml",
            f"functions/configs/{args.strategy.lower()}.yaml",
            f"functions/configs/{to_snake(args.strategy)}.yaml",
            f"functions/configs/{to_snake(args.strategy).replace('strategy','_strategy')}.yaml",
        ]
        print(f"{RED}ERROR: Strategy config not found for '{args.strategy}'. Tried:\n  - " + "\n  - ".join(tried) + f"{RESET}")
        print("Hint: ensure your config filename matches the strategy class in snake_case (e.g., ADXTrend -> adx_trend.yaml) or *_strategy form.")
        sys.exit(1)
    
    main_config = Path('configs/main_config.yaml')
    if not main_config.exists():
        print(f"{RED}ERROR: Main config not found: {main_config}{RESET}")
        sys.exit(1)
    
    # Run optimization
    run_bayesian_optimization(
        strategy_name=args.strategy,
        strategy_config_path=strategy_config,
        main_config_path=main_config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_folds=args.n_folds,
        validation_ratio=args.validation_ratio,
        reward_type=args.reward,
        seed=args.seed,
        param_stagnation_patience=args.param_stagnation_patience,
        param_tolerance=args.param_tolerance,
        top_k_validation=args.top_k_validation,
    )


if __name__ == '__main__':
    main()
