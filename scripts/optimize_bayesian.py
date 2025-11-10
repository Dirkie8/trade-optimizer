#!/usr/bin/env python3
"""
Bayesian optimization for trading strategies using Optuna with custom reward metric.

Example:
  python scripts/optimize_bayesian.py \
    --strategy_config functions/configs/rsi_strategy.yaml \
    --main_config configs/main_config.yaml \
    --n_trials 100 \
    --n_jobs 4
"""
import argparse
import importlib
import json
import os
import sys
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
    OPTUNA_AVAILABLE = False

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


def reward_metric(trades_df: pd.DataFrame, equity_curve: List[Dict[str, Any]]) -> float:
    """
    Custom reward metric emphasizing consistency and stability.
    
    Combines:
    - Mean Sharpe across 5 chunks (stability check)
    - Consistency penalty via Sharpe variance
    - Drawdown penalty
    
    Args:
        trades_df: DataFrame with at least 'pnl' column
        equity_curve: List of dicts with 'time' and 'equity' keys
        
    Returns:
        Higher is better. Negative infinity if degenerate.
    """
    if trades_df.empty or len(trades_df) < 5:
        return -np.inf
    
    pnl = trades_df['pnl'].values
    if np.all(pnl == 0):
        return -np.inf
    
    # Split into 5 chunks and compute Sharpe for each
    chunks = np.array_split(pnl, 5)
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
    stability = 1.0 - min(sharpe_std, 1.0)  # Cap at 1 to prevent negative
    
    # Drawdown penalty
    drawdown = max_drawdown_from_equity(equity_curve)
    
    # Combined reward
    reward = mean_sharpe * stability - 0.1 * drawdown
    return float(reward)


def infer_max_lookback(strategy_name: str, params: Dict[str, Any]) -> int:
    """Infer max lookback from strategy parameters."""
    # Check for common period parameters
    period_keys = [k for k in params.keys() if 'period' in k.lower() or 'window' in k.lower()]
    if period_keys:
        periods = [int(params[k]) for k in period_keys if isinstance(params.get(k), (int, float))]
        if periods:
            return max(periods) + 50  # Add buffer
    
    # Strategy-specific overrides
    if "Moving Average" in strategy_name or "MA" in strategy_name:
        ma_keys = [k for k in params.keys() if 'ma' in k.lower()]
        if ma_keys:
            return max([int(params[k]) for k in ma_keys if isinstance(params.get(k), (int, float))], default=200) + 50
    
    return 200  # Default fallback


def run_single_backtest(
    params: Dict[str, Any],
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
) -> Dict[str, Any]:
    """Execute a single backtest and return results with reward metric."""
    max_lb = infer_max_lookback(strategy_name, params)
    
    result = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=account_cfg,
        max_lookback=max_lb,
        progress=None,
    )
    
    # Build trades DataFrame for reward metric
    trades_detail = result.get("trades_detail", [])
    if trades_detail:
        trades_df = pd.DataFrame(trades_detail)
    else:
        trades_df = pd.DataFrame()
    
    # Calculate custom reward metric
    equity_curve = result.get("equity_curve", [])
    reward = reward_metric(trades_df, equity_curve)
    
    # Add reward to results
    result['reward_metric'] = reward
    
    # Build output row
    row = {**{f"param_{k}": v for k, v in params.items()}, **result}
    return row


def run_bayesian_optimization(
    param_config: Dict[str, Any],
    n_trials: int,
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
    n_jobs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run Bayesian optimization using Optuna TPE sampler.
    
    Args:
        param_config: Dict with parameter names and [min, max] ranges
        n_trials: Number of optimization trials
        data: Training data
        symbol: Trading symbol
        StrategyCls: Strategy class
        account_cfg: Account configuration
        strategy_name: Name of strategy
        n_jobs: Number of parallel workers
        
    Returns:
        List of result dictionaries
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create study with TPE sampler
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(
        direction="maximize",  # Maximize reward metric
        sampler=sampler,
    )
    
    print(f"{BOLD}{CYAN}Bayesian Optimization with Optuna TPE{RESET}")
    print(f"{CYAN}Trials: {n_trials} | Parallel jobs: {n_jobs}{RESET}")
    
    # Store all results
    all_results = []
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        # Sample parameters from continuous ranges
        params = {}
        for param_name, param_range in param_config.items():
            if not isinstance(param_range, list) or len(param_range) != 2:
                raise ValueError(f"Parameter '{param_name}' must be a list [min, max], got: {param_range}")
            
            min_val, max_val = param_range
            
            # All parameters are integers (as per requirement)
            # But we use float for Bayesian search, then round to int
            params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
        
        # Run backtest
        try:
            result = run_single_backtest(
                params=params,
                data=data,
                symbol=symbol,
                StrategyCls=StrategyCls,
                account_cfg=account_cfg,
                strategy_name=strategy_name,
            )
            
            all_results.append(result)
            
            # Get reward metric
            reward = result.get('reward_metric', -np.inf)
            
            # Penalize degenerate candidates (no trades)
            trades_ct = result.get("trades", 0)
            if trades_ct <= 0:
                return -1e10
            
            # Handle NaN/inf
            if not np.isfinite(reward):
                return -1e10
            
            return float(reward)
            
        except Exception as e:
            print(f"{RED}Trial failed: {e}{RESET}")
            return -1e10
    
    # Run optimization with progress bar
    with tqdm(total=n_trials, desc="Bayesian Optimization", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({
                'best_reward': f'{study.best_value:.4f}',
                'trial': trial.number + 1
            })
        
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
            n_jobs=max(1, int(n_jobs)),
        )
    
    print(f"\n{GREEN}Optimization complete!{RESET}")
    print(f"{BOLD}Best reward metric: {study.best_value:.4f}{RESET}")
    print(f"{BOLD}Best parameters: {study.best_params}{RESET}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization for trading strategies")
    parser.add_argument("--strategy_config", required=True, help="Path to strategy YAML config")
    parser.add_argument("--main_config", required=True, help="Path to main YAML config")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Bayesian optimization trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--output", default=None, help="Custom output path (overrides strategy results_dir)")
    parser.add_argument("--sort-by", default="reward_metric", 
                       choices=["reward_metric", "sharpe", "sortino", "calmar", "consistency_score", "total_return_pct"],
                       help="Metric to sort results by (default: reward_metric)")
    args = parser.parse_args()
    
    # Load configurations
    with open(args.strategy_config, "r") as f:
        strat_conf = yaml.safe_load(f)
    with open(args.main_config, "r") as f:
        main_conf = yaml.safe_load(f)
    
    general_conf = main_conf.get("general", {})
    symbol = general_conf.get("default_symbol")
    timeframe = general_conf.get("default_timeframe")
    validation_ratio = float(general_conf.get("validation_ratio", 0.2))
    
    # Load data
    data = load_data(symbol, timeframe)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Loaded data must have a DatetimeIndex")
    data = data.sort_index()
    
    # Split data: train (80%) and validation (20%)
    n = len(data)
    if n < 10:
        raise ValueError("Not enough data for optimization (<10 candles)")
    
    split_idx = int(n * (1.0 - validation_ratio))
    split_idx = min(max(split_idx, 1), n - 1)
    train_df = data.iloc[:split_idx]
    val_df = data.iloc[split_idx:]
    
    print(f"{CYAN}Data split: {len(train_df)} train | {len(val_df)} validation (validation_ratio={validation_ratio:.2f}){RESET}")
    
    # Load strategy class
    strategy_class_name = strat_conf['strategy']['class']
    module_path = strat_conf['strategy'].get('module', 'functions.strategies')
    results_dir = strat_conf['strategy'].get('results_dir', strategy_class_name)
    
    try:
        strategy_module = importlib.import_module(module_path)
        StrategyCls = getattr(strategy_module, strategy_class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import strategy '{strategy_class_name}' from module '{module_path}': {e}")
    
    strategy_name = strat_conf.get("name", strategy_class_name)
    
    # Get Bayesian parameter ranges
    param_config = strat_conf.get("parameters_bayesian", strat_conf.get("parameters_skopt"))
    if not param_config:
        raise ValueError("No 'parameters_bayesian' or 'parameters_skopt' section found in strategy config")
    
    print(f"{BOLD}Strategy: {strategy_name}{RESET}")
    print(f"{BOLD}Parameter ranges:{RESET}")
    for param, range_vals in param_config.items():
        print(f"  {param}: {range_vals}")
    
    # Run Bayesian optimization
    rows = run_bayesian_optimization(
        param_config=param_config,
        n_trials=args.n_trials,
        data=train_df,
        symbol=symbol,
        StrategyCls=StrategyCls,
        account_cfg=main_conf["account"],
        strategy_name=strategy_name,
        n_jobs=args.n_jobs,
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by specified metric
    if not df.empty:
        sort_metric = args.sort_by
        if sort_metric not in df.columns:
            print(f"{YELLOW}Warning: Sort metric '{sort_metric}' not found, falling back to 'reward_metric'{RESET}")
            sort_metric = "reward_metric"
        
        # Sort: primary metric desc, then trades desc, then total_return desc
        sort_cols = [sort_metric]
        if "trades" in df.columns:
            sort_cols.append("trades")
        if "total_return_pct" in df.columns:
            sort_cols.append("total_return_pct")
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    
    # Setup output paths
    if args.output:
        out_path = args.output
    else:
        opt_dir = f"results/{results_dir}/optimizations"
        os.makedirs(opt_dir, exist_ok=True)
        out_path = f"{opt_dir}/bayesian_optimization_results.csv"
    
    # Save CSV
    df.to_csv(out_path, index=False)
    print(f"{GREEN}Results saved to {out_path}{RESET}")
    
    # Save best parameters JSON
    if not df.empty:
        best = df.iloc[0]
        
        # Convert numpy types to native Python types
        def to_native(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            return x
        
        best_params = {k.replace("param_", ""): to_native(best[k]) for k in df.columns if k.startswith("param_")}
        
        # Include key metrics
        metric_fields = [
            "reward_metric",
            "consistency_score",
            "total_return_pct",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown_pct",
            "trades",
        ]
        metrics_payload = {}
        for m in metric_fields:
            if m in df.columns:
                try:
                    metrics_payload[m] = to_native(best[m])
                except Exception:
                    pass
        
        payload = {
            "method": "bayesian",
            "n_trials": args.n_trials,
            "sort_by": args.sort_by,
            "params": best_params,
            "metrics": metrics_payload,
        }
        
        best_json_path = out_path.replace(".csv", "_best.json")
        with open(best_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"{GREEN}Best parameters saved to {best_json_path}{RESET}")
    
    # Evaluate best parameters on validation set
    if not df.empty and len(val_df) > 1:
        print(f"\n{BOLD}{CYAN}Evaluating best parameters on validation set...{RESET}")
        best = df.iloc[0]
        best_params = {k.replace("param_", ""): to_native(best[k]) for k in df.columns if k.startswith("param_")}
        
        # Run backtest on validation data
        val_result = run_single_backtest(
            params=best_params,
            data=val_df,
            symbol=symbol,
            StrategyCls=StrategyCls,
            account_cfg=main_conf["account"],
            strategy_name=strategy_name,
        )
        
        print(f"{GREEN}Validation Results:{RESET}")
        print(f"  Reward Metric: {val_result.get('reward_metric', 0):.4f}")
        print(f"  Total Return: {val_result.get('total_return_pct', 0):.2f}%")
        print(f"  Sharpe: {val_result.get('sharpe', 0):.2f}")
        print(f"  Max Drawdown: {val_result.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Trades: {val_result.get('trades', 0)}")
        
        # Save validation results
        val_json_path = out_path.replace(".csv", "_validation.json")
        val_payload = {
            "params": best_params,
            "train_metrics": metrics_payload,
            "validation_metrics": {
                "reward_metric": val_result.get('reward_metric', 0),
                "total_return_pct": val_result.get('total_return_pct', 0),
                "sharpe": val_result.get('sharpe', 0),
                "sortino": val_result.get('sortino', 0),
                "calmar": val_result.get('calmar', 0),
                "max_drawdown_pct": val_result.get('max_drawdown_pct', 0),
                "trades": val_result.get('trades', 0),
            },
            "equity_curve": val_result.get('equity_curve', []),
        }
        with open(val_json_path, "w") as f:
            json.dump(val_payload, f, indent=2)
        print(f"{GREEN}Validation results saved to {val_json_path}{RESET}")


if __name__ == "__main__":
    main()
