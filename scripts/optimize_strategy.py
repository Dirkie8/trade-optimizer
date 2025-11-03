#!/usr/bin/env python3
"""
Run parameter optimization for a strategy and save results to CSV.

Example:
  python scripts/optimize_strategy.py \
    --strategy_config functions/configs/example_strategy.yaml \
    --main_config configs/main_config.yaml \
    --method grid or random \
    --n_random 20 \
    --n_jobs 4
"""
import argparse
import importlib
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import concurrent.futures as cf

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
import importlib

# Optuna for gradient-based optimization
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Ensure project root is on sys.path so `scripts` and `functions` are importable when executed as a file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy

# ANSI colors (disable with NO_COLOR env var)
if os.environ.get("NO_COLOR"):
    RESET = BOLD = CYAN = GREEN = YELLOW = MAGENTA = ""
else:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"

# =========================
# Spawn-safe worker support
# =========================
_WORKER_CTX: Dict[str, Any] | None = None


def _init_worker(ctx: Dict[str, Any]):
    """Initializer for ProcessPoolExecutor workers (spawn-safe).
    Stores shared context in a module-global so per-task calls don't need closures.
    """
    global _WORKER_CTX
    # Import Strategy class here to avoid importing on every task
    module_path = ctx.get("module_path")
    strategy_class_name = ctx.get("strategy_class_name")
    if module_path and strategy_class_name:
        try:
            m = importlib.import_module(module_path)
            ctx["StrategyCls"] = getattr(m, strategy_class_name)
        except Exception as e:
            # Leave StrategyCls unset; worker_run will raise a clear error
            ctx["StrategyCls"] = None
            ctx["_import_error"] = str(e)
    _WORKER_CTX = ctx


def _worker_run_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker function for evaluating a single param set.
    Uses context established by _init_worker.
    """
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized")
    ctx = _WORKER_CTX
    StrategyCls = ctx.get("StrategyCls")
    if StrategyCls is None:
        raise RuntimeError(f"Strategy import failed: {ctx.get('_import_error', 'unknown')}" )

    data = ctx["data"]
    symbol = ctx["symbol"]
    account_cfg = ctx["account_cfg"]
    strategy_name = ctx["strategy_name"]
    walk_forward = ctx["walk_forward"]
    n_folds = ctx["n_folds"]
    validation_weight = ctx["validation_weight"]

    if walk_forward:
        return run_walk_forward_candidate(
            params,
            data=data,
            symbol=symbol,
            StrategyCls=StrategyCls,
            account_cfg=account_cfg,
            strategy_name=strategy_name,
            n_folds=n_folds,
            validation_weight=validation_weight,
            progress=None,
        )
    else:
        return run_single_candidate(
            params,
            data=data,
            symbol=symbol,
            StrategyCls=StrategyCls,
            account_cfg=account_cfg,
            strategy_name=strategy_name,
            progress=None,
        )


def param_space_from_yaml(param_grid: Dict[str, Any], method: str, n_random: int) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [v if isinstance(v, list) else [v] for v in param_grid.values()]

    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if method == "grid" or n_random >= len(all_combos):
        return all_combos
    rng = np.random.default_rng(123)
    idxs = rng.choice(len(all_combos), size=n_random, replace=False)
    return [all_combos[i] for i in idxs]


def infer_max_lookback(strategy_name: str, params: Dict[str, Any]) -> int:
    if strategy_name == "Moving Average Cross Strategy":
        return int(max(params.get("long_ma_period", 200), params.get("short_ma_period", 50))) + 2
    return 200


def run_single_candidate(
    params: Dict[str, Any],
    *,
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
    progress: Dict[str, Any] | None = None,
):
    """Evaluate a single parameter set and return a result row dict."""
    max_lb = infer_max_lookback(strategy_name, params)
    res = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=account_cfg,
        max_lookback=max_lb,
        progress=progress,
    )
    row = {**{f"param_{k}": v for k, v in params.items()}, **res}
    return row


def run_optuna_optimization(
    *,
    param_config: Dict[str, Any],
    n_trials: int,
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
    sort_metric: str,
    walk_forward: bool = False,
    n_folds: int = 5,
    sampler_type: str = "tpe",
    n_jobs: int = 1,
    warmstart_csv: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Run Optuna-based optimization (TPE/Bayesian or Random).
    
    Args:
        param_config: Dictionary with parameter names and their ranges
        n_trials: Number of optimization trials
        sampler_type: 'tpe' (Bayesian) or 'random'
        
    Returns:
        List of result dictionaries sorted by sort_metric
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Choose sampler (enable multivariate and zero startup if warm-starting)
    if sampler_type == "tpe":
        sampler_kwargs = {"seed": 123}
        # If we plan to warm-start, allow immediate model use
        if warmstart_csv:
            sampler_kwargs.update({"multivariate": True, "n_startup_trials": 0})
        sampler = TPESampler(**sampler_kwargs)
        print(f"{CYAN}Using TPE (Bayesian) sampler - gradient-based optimization{RESET}")
    else:
        sampler = RandomSampler(seed=123)
        print(f"{YELLOW}Using Random sampler{RESET}")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",  # Maximize the sort_metric
        sampler=sampler,
    )

    # Optional warm-start: seed the study with prior random results
    if warmstart_csv and os.path.exists(warmstart_csv):
        try:
            from optuna.trial import TrialState
            from optuna.distributions import CategoricalDistribution
            import pandas as _pd

            df_ws = _pd.read_csv(warmstart_csv)
            # Build distributions from param_config
            distributions = {}
            for p, vals in param_config.items():
                if isinstance(vals, list) and len(vals) > 0:
                    distributions[p] = CategoricalDistribution(choices=vals)
                else:
                    distributions[p] = CategoricalDistribution(choices=[vals])

            seeded = 0
            for _, row in df_ws.iterrows():
                if sort_metric not in row or not np.isfinite(row[sort_metric]):
                    continue
                params = {}
                ok = True
                for p in param_config.keys():
                    col = f"param_{p}"
                    if col in row:
                        v = row[col]
                        # Cast to match choices type when categorical ints
                        dist = distributions.get(p)
                        if isinstance(dist, CategoricalDistribution):
                            choices = dist.choices
                            # Try int-cast if all choices are ints
                            if all(isinstance(c, int) for c in choices):
                                try:
                                    v = int(v)
                                except Exception:
                                    pass
                            elif all(isinstance(c, float) for c in choices):
                                try:
                                    v = float(v)
                                except Exception:
                                    pass
                            # Skip if value not in domain
                            if v not in choices:
                                ok = False
                                break
                        params[p] = v
                if not ok:
                    continue

                try:
                    trial = optuna.create_trial(
                        params=params,
                        distributions=distributions,
                        value=float(row[sort_metric]),
                        state=TrialState.COMPLETE,
                    )
                    study.add_trial(trial)
                    seeded += 1
                except Exception:
                    # skip malformed rows silently
                    pass
            if seeded > 0:
                print(f"Warm-started TPE with {seeded} prior results from {warmstart_csv}")
        except Exception as e:
            print(f"Warning: failed to warm-start from {warmstart_csv}: {e}")
    
    # Store results for DataFrame
    all_results = []
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        # Sample parameters from their defined ranges
        params = {}
        for param_name, param_values in param_config.items():
            if isinstance(param_values, list):
                if len(param_values) == 0:
                    continue
                # Check if values are numeric (continuous) or categorical
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric parameter - infer distribution when possible for better BO
                    try:
                        # Handle integer sequences with constant step via suggest_int
                        if all(isinstance(v, int) for v in param_values):
                            if len(param_values) >= 3:
                                steps = {param_values[i+1] - param_values[i] for i in range(len(param_values)-1)}
                                if len(steps) == 1:
                                    step = abs(next(iter(steps))) or 1
                                    low, high = int(min(param_values)), int(max(param_values))
                                    params[param_name] = trial.suggest_int(param_name, low, high, step=step)
                                else:
                                    # Non-uniform discrete set
                                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                            else:
                                params[param_name] = trial.suggest_categorical(param_name, param_values)
                        else:
                            # Floats: try uniform stepped grid if evenly spaced, else categorical
                            if len(param_values) >= 3:
                                diffs = [float(param_values[i+1]) - float(param_values[i]) for i in range(len(param_values)-1)]
                                tol = 1e-9
                                if max(diffs) - min(diffs) <= tol:
                                    step = max(abs(d) for d in diffs) or None
                                    low, high = float(min(param_values)), float(max(param_values))
                                    # Optuna suggest_float supports step in recent versions; fallback if not
                                    try:
                                        params[param_name] = trial.suggest_float(param_name, low, high, step=step)
                                    except TypeError:
                                        params[param_name] = trial.suggest_float(param_name, low, high)
                                else:
                                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                            else:
                                low, high = float(min(param_values)), float(max(param_values))
                                try:
                                    params[param_name] = trial.suggest_float(param_name, low, high)
                                except TypeError:
                                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                    except Exception:
                        # Fallback to categorical if anything goes wrong
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                # Single value - use as-is
                params[param_name] = param_values
        
        # Run backtest
        try:
            if walk_forward:
                result = run_walk_forward_candidate(
                    params,
                    data=data,
                    symbol=symbol,
                    StrategyCls=StrategyCls,
                    account_cfg=account_cfg,
                    strategy_name=strategy_name,
                    n_folds=n_folds,
                    validation_weight=0.3,
                    progress=None,
                )
            else:
                result = run_single_candidate(
                    params,
                    data=data,
                    symbol=symbol,
                    StrategyCls=StrategyCls,
                    account_cfg=account_cfg,
                    strategy_name=strategy_name,
                    progress=None,
                )
            
            all_results.append(result)

            # Penalize degenerate candidates with no trades
            trades_ct = result.get("trades_count")
            if trades_ct is None:
                trades_ct = result.get("trades", 0)
            if trades_ct is not None and int(trades_ct) <= 0:
                return -1e10

            # Return the metric to optimize
            metric_value = result.get(sort_metric, 0.0)
            
            # Handle NaN/inf
            if not np.isfinite(metric_value):
                return -1e10
            
            return float(metric_value)
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1e10  # Return very bad score on failure
    
    # Run optimization with progress bar
    with tqdm(total=n_trials, desc="Optuna Optimization", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({
                'best': f'{study.best_value:.2f}',
                'trial': trial.number + 1
            })
        
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
            n_jobs=max(1, int(n_jobs) if n_jobs is not None else 1),
        )
    
    print(f"\n{GREEN}Optimization complete!{RESET}")
    print(f"{BOLD}Best {sort_metric}: {study.best_value:.4f}{RESET}")
    print(f"{BOLD}Best parameters: {study.best_params}{RESET}")
    
    return all_results


def run_walk_forward_candidate(
    params: Dict[str, Any],
    *,
    data: pd.DataFrame,
    symbol: str,
    StrategyCls,
    account_cfg: Dict[str, Any],
    strategy_name: str,
    n_folds: int = 5,
    validation_weight: float = 0.3,
    progress: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a single parameter set using walk-forward validation.
    
    Splits data into n_folds, trains on first n-1 folds, validates on last fold.
    Returns weighted average of train and validation metrics.
    
    Args:
        params: Strategy parameters to test
        data: Full dataset
        n_folds: Number of folds for walk-forward
        validation_weight: Weight for validation score (0-1)
        
    Returns:
        Combined metrics dict with train/val scores
    """
    max_lb = infer_max_lookback(strategy_name, params)
    n = len(data)
    fold_size = n // n_folds
    
    if fold_size < 100:
        # Data too small for walk-forward, fall back to single split
        return run_single_candidate(
            params, data=data, symbol=symbol, StrategyCls=StrategyCls,
            account_cfg=account_cfg, strategy_name=strategy_name, progress=progress
        )
    
    # Collect metrics from each fold
    fold_metrics = []
    
    for fold in range(1, n_folds + 1):
        # Train on data up to end of (fold-1), validate on fold
        train_end = fold_size * (fold - 1)
        val_start = train_end
        val_end = fold_size * fold
        
        if fold == 1:
            # First fold: no training data, skip
            continue
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[val_start:val_end]
        
        if len(train_data) < max_lb or len(val_data) < 50:
            continue
        
        # Run backtest on validation fold
        try:
            val_res = backtest_strategy(
                data=val_data,
                symbol=symbol,
                strategy_cls=StrategyCls,
                params=params,
                account_cfg=account_cfg,
                max_lookback=max_lb,
                progress=None,  # Suppress inner progress bars
            )
            fold_metrics.append(val_res)
        except Exception:
            # Skip failed folds
            continue
    
    if not fold_metrics:
        # All folds failed, fall back to single run
        return run_single_candidate(
            params, data=data, symbol=symbol, StrategyCls=StrategyCls,
            account_cfg=account_cfg, strategy_name=strategy_name, progress=progress
        )
    
    # Average metrics across folds
    avg_metrics = {}
    metric_keys = fold_metrics[0].keys()
    
    for key in metric_keys:
        if key == "equity_curve":
            # Don't average equity curves - use last fold's curve
            avg_metrics[key] = fold_metrics[-1][key]
        elif key in ["starting_balance", "ending_balance"]:
            # Use last fold's balance values
            avg_metrics[key] = fold_metrics[-1][key]
        else:
            # Average numeric metrics
            values = [m[key] for m in fold_metrics if key in m]
            if values:
                try:
                    # Try to average - works for numbers
                    avg_metrics[key] = np.mean(values)
                except (TypeError, ValueError):
                    # Non-numeric value (shouldn't happen, but fallback to last)
                    avg_metrics[key] = fold_metrics[-1][key]
            else:
                avg_metrics[key] = 0.0
    
    # Add walk-forward specific metrics
    avg_metrics["n_folds"] = len(fold_metrics)
    avg_metrics["wf_consistency"] = np.std([m.get("consistency_score", 0) for m in fold_metrics])
    
    # Build result row
    row = {**{f"param_{k}": v for k, v in params.items()}, **avg_metrics}
    return row


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--strategy_config", required=True)
    parser.add_argument("--main_config", required=True)
    parser.add_argument("--method", choices=["grid", "random", "tpe", "optuna"], default="grid",
                       help="Optimization method: grid, random, tpe (Bayesian/gradient-based), or optuna (alias for tpe)")
    parser.add_argument("--n_random", type=int, default=20,
                       help="Number of random samples (for random method)")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of trials for Optuna optimization (tpe/optuna methods)")
    parser.add_argument("--output", default=None, help="Custom output path (overrides strategy results_dir)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel workers: -1=auto, 1=no parallel")
    parser.add_argument("--sort-by", default="consistency_score", 
                       choices=["sharpe", "sortino", "calmar", "consistency_score", "total_return_pct"],
                       help="Metric to sort results by (default: consistency_score)")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Enable walk-forward validation to reduce overfitting")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of folds for walk-forward validation (default: 5)")
    parser.add_argument("--validation-weight", type=float, default=0.3,
                       help="Weight for validation score in objective (0.0-1.0, default: 0.3)")
    parser.add_argument("--write_test_eval", action="store_true",
                       help="If set, after optimization evaluate top-N on test split and write under evaluations. Default: disabled.")
    parser.add_argument("--warmstart_csv", default=None,
                       help="Optional CSV to warm-start Optuna (TPE) with prior results (expects param_* columns and target metric)")
    args = parser.parse_args()

    with open(args.strategy_config, "r") as f:
        strat_conf = yaml.safe_load(f)
    with open(args.main_config, "r") as f:
        main_conf = yaml.safe_load(f)

    general_conf = main_conf.get("general", {})
    symbol = general_conf.get("default_symbol")
    timeframe = general_conf.get("default_timeframe")
    train_ratio = float(general_conf.get("train_ratio", 1.0))  # 1.0 = no split
    top_n_eval = int(general_conf.get("evaluation_top_n", 50))

    data = load_data(symbol, timeframe)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Loaded data must have a DatetimeIndex")
    data = data.sort_index()

    # Split train/test per config
    n = len(data)
    if n < 10:
        raise ValueError("Not enough data for optimization (<10 candles)")
    split_idx = int(n * train_ratio)
    split_idx = min(max(split_idx, 1), n - 1)  # ensure 1..n-1
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    if train_ratio < 1.0:
        print(f"{CYAN}Train/Test split: {len(train_df)} train | {len(test_df)} test (ratio={train_ratio:.2f}){RESET}")
    else:
        print(f"{CYAN}No train/test split (train_ratio={train_ratio:.2f}); using full data for optimization only{RESET}")

    # Load strategy class dynamically from YAML config
    strategy_class_name = strat_conf['strategy']['class']
    module_path = strat_conf['strategy'].get('module', 'functions.strategies')
    results_dir = strat_conf['strategy'].get('results_dir', strategy_class_name)  # Default to class name
    
    try:
        strategy_module = importlib.import_module(module_path)
        StrategyCls = getattr(strategy_module, strategy_class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import strategy '{strategy_class_name}' from module '{module_path}': {e}")
    
    strategy_name = strat_conf.get("name", strategy_class_name)

    param_grid = strat_conf["parameters"]
    
    # Check if using Optuna (TPE/Bayesian) optimization
    use_optuna = args.method in ["tpe", "optuna"]
    
    if use_optuna:
        if not OPTUNA_AVAILABLE:
            print("ERROR: Optuna not installed. Run: pip install optuna")
            print("Falling back to random search...")
            use_optuna = False
            args.method = "random"
    
    # Generate candidates or prepare for Optuna
    if use_optuna:
        total = args.n_trials
        print(f"{BOLD}{GREEN}Using Optuna TPE (Bayesian/gradient-based) optimization: {total} trials{RESET}")
    else:
        candidates = param_space_from_yaml(param_grid, args.method, args.n_random)
        total = len(candidates)
    
    # Determine worker count EARLY (used by both Optuna and manual branches)
    if args.n_jobs is None or args.n_jobs == -1:
        cpu = os.cpu_count() or 1
        n_workers = max(1, cpu - 1)  # leave one core free
    else:
        n_workers = max(1, args.n_jobs)

    # Determine optimization method
    if use_optuna:
        # Optuna handles optimization internally
        rows = run_optuna_optimization(
            param_config=param_grid,
            n_trials=total,
            data=train_df if not args.walk_forward else data,
            symbol=symbol,
            StrategyCls=StrategyCls,
            account_cfg=main_conf["account"],
            strategy_name=strategy_name,
            sort_metric=args.sort_by,
            walk_forward=args.walk_forward,
            n_folds=args.n_folds,
            sampler_type="tpe",
            n_jobs=n_workers,
            warmstart_csv=args.warmstart_csv,
        )
    elif args.walk_forward:
        print(f"Starting WALK-FORWARD optimization: {total} parameter sets, {args.n_folds} folds")
        print(f"Validation weight: {args.validation_weight:.2f}")
        optimization_data = data  # Use full data for walk-forward
        # Prepare worker context
        worker_ctx = {
            "data": optimization_data,
            "symbol": symbol,
            "account_cfg": main_conf["account"],
            "strategy_name": strategy_name,
            "walk_forward": True,
            "n_folds": args.n_folds,
            "validation_weight": args.validation_weight,
            "module_path": module_path,
            "strategy_class_name": strategy_class_name,
        }
    else:
        print(f"Starting optimization on TRAIN data: {total} parameter sets")
        optimization_data = train_df
        worker_ctx = {
            "data": optimization_data,
            "symbol": symbol,
            "account_cfg": main_conf["account"],
            "strategy_name": strategy_name,
            "walk_forward": False,
            "n_folds": args.n_folds,
            "validation_weight": args.validation_weight,
            "module_path": module_path,
            "strategy_class_name": strategy_class_name,
        }

    # Setup output paths with strategy-specific subdirectory
    if args.output:
        out_path = args.output
    else:
        opt_dir = f"results/{results_dir}/optimizations"
        os.makedirs(opt_dir, exist_ok=True)
        out_path = f"{opt_dir}/optimization_results.csv"
    
    # Evaluation dir (only used if --write_test_eval)
    eval_dir = f"results/{results_dir}/evaluations"

    # (n_workers already computed above)

    if not use_optuna:
        rows: List[Dict[str, Any]] = []
    
    # Skip manual optimization if using Optuna (already done above)
    if not use_optuna and n_workers == 1:
        # Sequential with progress bar
        # Initialize worker context in main process
        _init_worker(worker_ctx)
        with tqdm(total=total, desc="Optimizing", unit="run") as pbar:
            for i, params in enumerate(candidates, 1):
                pbar.set_postfix(run=f"{i}/{total}")
                pbar.set_description_str(f"Optimizing (run {i}/{total})")
                # Optional: short print per run
                print(f"Run {i}/{total} params={params}")
                row = _worker_run_params(params)
                rows.append(row)
                pbar.update(1)
    elif not use_optuna:
        # Parallel execution with progress bar (skip if Optuna)
        print(f"{CYAN}Using {n_workers} parallel worker(s){RESET}")
        with cf.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(worker_ctx,)) as ex:
            # Submit first wave equal to number of workers, assign bar positions 0..n_workers-1
            pending: dict[int, tuple[int, Dict[str, Any], Any]] = {}
            it = iter(enumerate(candidates, 1))

            def submit_next(position: int):
                try:
                    i, params = next(it)
                except StopIteration:
                    return False
                fut = ex.submit(_worker_run_params, params)
                pending[position] = (i, params, fut)
                return True

            # Prime the pool
            active = 0
            for pos in range(n_workers):
                if submit_next(pos):
                    active += 1

            with tqdm(total=total, desc="Optimizing", unit="run") as pbar:
                while pending:
                    # Wait for any future to complete
                    done, _ = cf.wait([f for (_i, _p, f) in pending.values()], return_when=cf.FIRST_COMPLETED)
                    for position, (i, params, fut) in list(pending.items()):
                        if fut in done:
                            try:
                                row = fut.result()
                                rows.append(row)
                                pbar.set_postfix(run=f"{i}/{total}")
                                pbar.set_description_str(f"Optimizing (run {i}/{total})")
                                print(f"Completed {i}/{total} params={params}")
                            except Exception as e:
                                print(f"Run {i}/{total} failed for params={params}: {e}")
                            finally:
                                pbar.update(1)
                                # Remove completed and submit a new task in the same position
                                del pending[position]
                                submit_next(position)

    df = pd.DataFrame(rows)

    # Sort by user-specified metric (default: consistency_score)
    if not df.empty:
        sort_metric = args.sort_by
        # If primary sort metric doesn't exist, fall back to sharpe
        if sort_metric not in df.columns:
            print(f"Warning: Sort metric '{sort_metric}' not found, falling back to 'sharpe'")
            sort_metric = "sharpe"

        # Prefer more trades in ties: primary metric, then trades_count, then total_return
        sort_cols = [sort_metric]
        if "trades_count" in df.columns:
            sort_cols.append("trades_count")
        sort_cols.append("total_return_pct" if "total_return_pct" in df.columns else sort_metric)
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"{GREEN}Optimization completed and saved to {out_path}{RESET}")

    if not df.empty:
        best = df.iloc[0]
        # Save best params JSON alongside CSV for convenience
        def to_native(x):
            try:
                if isinstance(x, (np.integer,)):
                    return int(x)
                if isinstance(x, (np.floating,)):
                    return float(x)
            except Exception:
                pass
            return x
        best_params = {k.replace("param_", ""): to_native(best[k]) for k in df.columns if k.startswith("param_")}
        best_json_path = out_path.replace(".csv", "_best.json")

        # Include useful metadata to make provenance obvious when comparing files
        metric_fields = [
            "consistency_score",
            "total_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "avg_drawdown_pct",
            "max_drawdown_pct",
            "trades_count",
        ]
        metrics_payload = {}
        for m in metric_fields:
            if m in df.columns:
                try:
                    metrics_payload[m] = to_native(best[m])
                except Exception:
                    pass

        payload = {
            "method": args.method,
            "sort_by": args.sort_by,
            "rows": int(len(df)),
            "params": best_params,
        }
        if metrics_payload:
            payload["metrics"] = metrics_payload

        with open(best_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"{GREEN}Best parameters saved to {best_json_path}{RESET}")
    else:
        best_json_path = out_path.replace(".csv", "_best.json")
        print(f"{YELLOW}No results to save best JSON for {best_json_path}{RESET}")

    # If explicitly requested, and we have a test split and any results, evaluate top-N on TEST data
    if args.write_test_eval and train_ratio < 1.0 and not df.empty and len(test_df) > 1:
        try:
            top_n = max(1, min(top_n_eval, len(df)))
            top_params_list: List[Dict[str, Any]] = []
            for _idx, row in df.head(top_n).iterrows():
                p = {col.replace("param_", ""): row[col] for col in df.columns if col.startswith("param_")}
                # Ensure native types for potential JSON serialization
                for k, v in list(p.items()):
                    if isinstance(v, (np.integer,)):
                        p[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        p[k] = float(v)
                top_params_list.append(p)

            print(f"Evaluating top {len(top_params_list)} parameter sets on TEST data…")

            # Reuse worker pool settings
            eval_rows: List[Dict[str, Any]] = []
            if n_workers == 1:
                with tqdm(total=len(top_params_list), desc="Evaluating", unit="run") as pbar:
                    for i, params in enumerate(top_params_list, 1):
                        pbar.set_postfix(run=f"{i}/{len(top_params_list)}")
                        print(f"Eval {i}/{len(top_params_list)} params={params}")
                        # Reuse worker mechanism sequentially
                        _init_worker({
                            "data": test_df,
                            "symbol": symbol,
                            "account_cfg": main_conf["account"],
                            "strategy_name": strategy_name,
                            "walk_forward": False,
                            "n_folds": args.n_folds,
                            "validation_weight": args.validation_weight,
                            "module_path": module_path,
                            "strategy_class_name": strategy_class_name,
                        })
                        row = _worker_run_params(params)
                        eval_rows.append(row)
                        pbar.update(1)
            else:
                with cf.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=({
                    "data": test_df,
                    "symbol": symbol,
                    "account_cfg": main_conf["account"],
                    "strategy_name": strategy_name,
                    "walk_forward": False,
                    "n_folds": args.n_folds,
                    "validation_weight": args.validation_weight,
                    "module_path": module_path,
                    "strategy_class_name": strategy_class_name,
                },)) as ex:
                    pending_eval: dict[int, tuple[int, Dict[str, Any], Any]] = {}
                    it_eval = iter(enumerate(top_params_list, 1))

                    def submit_next_eval(position: int):
                        try:
                            i, params = next(it_eval)
                        except StopIteration:
                            return False
                        fut = ex.submit(_worker_run_params, params)
                        pending_eval[position] = (i, params, fut)
                        return True

                    active = 0
                    for pos in range(n_workers):
                        if submit_next_eval(pos):
                            active += 1

                    with tqdm(total=len(top_params_list), desc="Evaluating", unit="run") as pbar:
                        while pending_eval:
                            done, _ = cf.wait([f for (_i, _p, f) in pending_eval.values()], return_when=cf.FIRST_COMPLETED)
                            for position, (i, params, fut) in list(pending_eval.items()):
                                if fut in done:
                                    try:
                                        row = fut.result()
                                        eval_rows.append(row)
                                        print(f"Completed Eval {i}/{len(top_params_list)} params={params}")
                                    except Exception as e:
                                        print(f"Eval {i}/{len(top_params_list)} failed for params={params}: {e}")
                                    finally:
                                        pbar.update(1)
                                        del pending_eval[position]
                                        submit_next_eval(position)

            # Compile evaluation DataFrame and save
            eval_df = pd.DataFrame(eval_rows)
            if not eval_df.empty:
                # Use same sort metric as optimization
                sort_metric = args.sort_by
                if sort_metric not in eval_df.columns:
                    sort_metric = "sharpe"
                eval_df = eval_df.sort_values([sort_metric, "total_return_pct"], ascending=[False, False]).reset_index(drop=True)

            os.makedirs(eval_dir, exist_ok=True)
            eval_csv_path = f"{eval_dir}/evaluation_results.csv"
            eval_df.to_csv(eval_csv_path, index=False)
            print(f"Evaluation results saved to {eval_csv_path}")

            # Save best evaluation params JSON and best full eval JSON with equity curve
            if not eval_df.empty:
                best_eval = eval_df.iloc[0]
                best_eval_params = {k.replace("param_", ""): best_eval[k] for k in eval_df.columns if k.startswith("param_")}
                # Coerce numpy types
                for k, v in list(best_eval_params.items()):
                    if isinstance(v, (np.integer,)):
                        best_eval_params[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        best_eval_params[k] = float(v)

                eval_best_path = f"{eval_dir}/evaluation_results_best.json"
                with open(eval_best_path, "w") as f:
                    json.dump({"params": best_eval_params}, f, indent=2)
                print(f"Best evaluation parameters saved to {eval_best_path}")

                # Also produce eval_results.json payload including equity curve for the best
                # Re-run backtest on TEST data for the best to get equity curve (if not already present in row)
                best_payload = backtest_strategy(
                    data=test_df,
                    symbol=symbol,
                    strategy_cls=StrategyCls,
                    params=best_eval_params,
                    account_cfg=main_conf["account"],
                    max_lookback=infer_max_lookback(strategy_name, best_eval_params),
                )
                payload = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "params": best_eval_params,
                    "metrics": {
                        "starting_balance": best_payload["starting_balance"],
                        "ending_balance": best_payload["ending_balance"],
                        "total_return_pct": best_payload["total_return_pct"],
                        "sharpe": best_payload["sharpe"],
                        "max_drawdown_pct": best_payload["max_drawdown_pct"],
                        "trades": best_payload["trades"],
                        "win_rate_pct": best_payload["win_rate_pct"],
                    },
                    "equity_curve": best_payload["equity_curve"],
                }
                eval_json_path = f"{eval_dir}/eval_results.json"
                with open(eval_json_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"Best evaluation payload saved to {eval_json_path}")
        except Exception as e:
            print(f"Evaluation phase skipped due to error: {e}")


if __name__ == "__main__":
    main()
