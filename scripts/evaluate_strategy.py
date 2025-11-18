#!/usr/bin/env python3
"""
Evaluate a strategy with chosen parameters over full dataset and save JSON outputs.

Flow:
- Defaults to reading params from results/<strategy>/optimizations/optimization_results_best.json
- If not available, falls back to optimization_results.csv best row; else YAML defaults
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple
import numbers

import pandas as pd
import yaml

# Ensure project root is on sys.path so `scripts` and `functions` are importable when executed as a file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy
import importlib


def select_params(strategy_conf: Dict[str, Any], default_opt_dir: str, optimization_csv: str | None, selection_metric: str) -> Dict[str, Any]:
    # Priority order:
    # 1. Bayesian optimization best params (newest method)
    # 2. Standard optimization best params
    # 3. Explicit CSV path provided
    # 4. Default optimization CSV
    # 5. YAML defaults
    
    # Check for Bayesian optimization results first (newest/recommended method)
    # Prefer walk-forward (WF) results if available
    wf_best_json = os.path.join(default_opt_dir, "bayesian_wf_optimization_results_best.json")
    if os.path.exists(wf_best_json):
        try:
            with open(wf_best_json, "r") as f:
                print(f"Using WF Bayesian optimization parameters from: {wf_best_json}")
                return json.load(f)["params"]
        except Exception as e:
            print(f"Warning: Failed to load WF Bayesian params: {e}")
            pass
    bayesian_best_json = os.path.join(default_opt_dir, "bayesian_optimization_results_best.json")
    if os.path.exists(bayesian_best_json):
        try:
            with open(bayesian_best_json, "r") as f:
                print(f"Using Bayesian optimization parameters from: {bayesian_best_json}")
                return json.load(f)["params"]
        except Exception as e:
            print(f"Warning: Failed to load Bayesian params: {e}")
            pass
    
    # Then check for standard optimization results
    best_json_default = os.path.join(default_opt_dir, "optimization_results_best.json")
    if os.path.exists(best_json_default):
        try:
            with open(best_json_default, "r") as f:
                print(f"Using optimization parameters from: {best_json_default}")
                return json.load(f)["params"]
        except Exception:
            pass

    # If an optimization CSV was provided, prefer its _best.json then CSV
    if optimization_csv:
        best_json = optimization_csv.replace(".csv", "_best.json")
        if os.path.exists(best_json):
            with open(best_json, "r") as f:
                return json.load(f)["params"]
        if os.path.exists(optimization_csv):
            df = pd.read_csv(optimization_csv)
            if selection_metric in df.columns:
                df = df.sort_values(selection_metric, ascending=False)
            else:
                df = df.sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
            row = df.iloc[0]
            return {col.replace("param_", ""): row[col] for col in df.columns if col.startswith("param_")}

    # Otherwise try default optimization_results.csv
    default_csv = os.path.join(default_opt_dir, "optimization_results.csv")
    if os.path.exists(default_csv):
        df = pd.read_csv(default_csv)
        if selection_metric in df.columns:
            df = df.sort_values(selection_metric, ascending=False)
        else:
            df = df.sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
        row = df.iloc[0]
        return {col.replace("param_", ""): row[col] for col in df.columns if col.startswith("param_")}

    # Fall back to YAML defaults
    params = {}
    for k, v in strategy_conf["parameters"].items():
        params[k] = v[0] if isinstance(v, list) and v else v
    return params


def infer_max_lookback(strategy_name: str, params: Dict[str, Any]) -> int:
    if strategy_name == "Moving Average Cross Strategy":
        return int(max(params.get("long_ma_period", 200), params.get("short_ma_period", 50))) + 2
    return 200


def main():
    parser = argparse.ArgumentParser(description="Evaluate strategy with chosen params")
    parser.add_argument("--strategy_config", required=True)
    parser.add_argument("--main_config", required=True)
    parser.add_argument("--optimization_csv", help="Path to optimization CSV to auto-select best params")
    parser.add_argument("--selection_metric", default="sharpe", help="Metric to select best from CSV")
    parser.add_argument("--output", default=None, help="Optional explicit output JSON path")
    parser.add_argument("--results_root", default="results", help="Root directory to read optimization artifacts from (default: results)")
    parser.add_argument("--param_source", choices=["auto", "opt", "yaml", "best_yaml"], default="auto", help="Where to source parameters from: auto (default priority), opt (optimization artifacts only), yaml (first values in parameters), best_yaml (use best_params in YAML)")
    parser.add_argument("--no_round", action="store_true", help="Disable rounding floats to 2 decimals before backtest")
    args = parser.parse_args()

    with open(args.strategy_config, "r") as f:
        strat_conf = yaml.safe_load(f)
    with open(args.main_config, "r") as f:
        main_conf = yaml.safe_load(f)

    symbol = main_conf["general"]["default_symbol"]
    timeframe = main_conf["general"]["default_timeframe"]

    # Resolve strategy class dynamically
    strategy_class_name = strat_conf['strategy']['class']
    module_path = strat_conf['strategy'].get('module', 'functions.strategies')
    results_dir = strat_conf['strategy'].get('results_dir', strategy_class_name)
    try:
        strategy_module = importlib.import_module(module_path)
        StrategyCls = getattr(strategy_module, strategy_class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import strategy '{strategy_class_name}' from module '{module_path}': {e}")
    strategy_name = strat_conf.get("name", strategy_class_name)

    data = load_data(symbol, timeframe)
    # Allow selecting a specific results root (e.g., results_2025-11-07)
    results_root = args.results_root
    default_opt_dir = os.path.join(results_root, results_dir, "optimizations")
    
    # Fallback to case-insensitive/snake_case variant if exact match not found
    if not os.path.isdir(default_opt_dir):
        # Try snake_case conversion
        import re
        def to_snake(name: str) -> str:
            name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
            name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
            return name.replace('__', '_').lower()
        snake_dir = os.path.join(results_root, to_snake(results_dir), "optimizations")
        if os.path.isdir(snake_dir):
            default_opt_dir = snake_dir
            results_dir = to_snake(results_dir)  # Update results_dir for later eval path
        else:
            # Final fallback to legacy "results" root
            legacy_dir = os.path.join("results", results_dir, "optimizations")
            if os.path.isdir(legacy_dir):
                default_opt_dir = legacy_dir
            else:
                # Try snake_case in legacy too
                legacy_snake = os.path.join("results", to_snake(results_dir), "optimizations")
                if os.path.isdir(legacy_snake):
                    default_opt_dir = legacy_snake
                    results_dir = to_snake(results_dir)
    
    # Helper to resolve params strictly from optimization artifacts
    def select_params_opt_only(opt_dir: str, optimization_csv: str | None, selection_metric: str) -> Tuple[Dict[str, Any], str | None]:
        # Try WF best
        wf_best_json = os.path.join(opt_dir, "bayesian_wf_optimization_results_best.json")
        if os.path.exists(wf_best_json):
            try:
                with open(wf_best_json, "r") as f:
                    return json.load(f)["params"], wf_best_json
            except Exception:
                pass
        bayesian_best_json = os.path.join(opt_dir, "bayesian_optimization_results_best.json")
        if os.path.exists(bayesian_best_json):
            try:
                with open(bayesian_best_json, "r") as f:
                    return json.load(f)["params"], bayesian_best_json
            except Exception:
                pass
        best_json_default = os.path.join(opt_dir, "optimization_results_best.json")
        if os.path.exists(best_json_default):
            try:
                with open(best_json_default, "r") as f:
                    return json.load(f)["params"], best_json_default
            except Exception:
                pass
        # From CSV (explicit or default)
        if optimization_csv and os.path.exists(optimization_csv):
            df = pd.read_csv(optimization_csv)
            if selection_metric in df.columns:
                df = df.sort_values(selection_metric, ascending=False)
            else:
                df = df.sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
            row = df.iloc[0]
            return {col.replace("param_", ""): row[col] for col in df.columns if col.startswith("param_")}, optimization_csv
        default_csv = os.path.join(opt_dir, "optimization_results.csv")
        if os.path.exists(default_csv):
            df = pd.read_csv(default_csv)
            if selection_metric in df.columns:
                df = df.sort_values(selection_metric, ascending=False)
            else:
                df = df.sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
            row = df.iloc[0]
            return {col.replace("param_", ""): row[col] for col in df.columns if col.startswith("param_")}, default_csv
        raise FileNotFoundError("No optimization artifacts found to source params (opt mode)")

    # Helper to select YAML defaults (first values from parameters)
    def select_params_yaml_defaults(strategy_conf: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for k, v in strategy_conf.get("parameters", {}).items():
            params[k] = v[0] if isinstance(v, list) and v else v
        return params

    # Helper to select YAML best_params (optionally timeframe-specific)
    def select_params_yaml_best(strategy_conf: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        # Try exact match best_params_<timeframe>, else best_params
        key_tf = f"best_params_{timeframe}"
        bp = strategy_conf.get(key_tf)
        if isinstance(bp, dict) and bp:
            return dict(bp)
        bp = strategy_conf.get("best_params")
        if isinstance(bp, dict) and bp:
            return dict(bp)
        raise KeyError(f"No best_params or best_params_{timeframe} found in strategy YAML")

    param_source_used = "auto"
    param_source_path: str | None = None
    if args.param_source == "auto":
        params = select_params(strat_conf, default_opt_dir, args.optimization_csv, args.selection_metric)
        # best effort to find the path used (for WF best)
        for cand in [
            os.path.join(default_opt_dir, "bayesian_wf_optimization_results_best.json"),
            os.path.join(default_opt_dir, "bayesian_optimization_results_best.json"),
            os.path.join(default_opt_dir, "optimization_results_best.json"),
            os.path.join(default_opt_dir, "optimization_results.csv"),
        ]:
            if os.path.exists(cand):
                param_source_path = cand
                break
    elif args.param_source == "opt":
        params, param_source_path = select_params_opt_only(default_opt_dir, args.optimization_csv, args.selection_metric)
        param_source_used = "opt"
    elif args.param_source == "yaml":
        params = select_params_yaml_defaults(strat_conf)
        param_source_used = "yaml"
    else:  # best_yaml
        params = select_params_yaml_best(strat_conf, timeframe)
        param_source_used = "best_yaml"

    # Normalize numeric params: round floats to 2 decimals; leave ints (and bools) unchanged
    def _round_numeric_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
        rounded: Dict[str, Any] = {}
        for k, v in p.items():
            # Important: bool is a subclass of int; keep it untouched
            if isinstance(v, bool):
                rounded[k] = v
            elif isinstance(v, int):
                rounded[k] = v
            elif isinstance(v, numbers.Real):
                # Force to float to handle numpy scalars as well
                rounded[k] = round(float(v), decimals)
            else:
                rounded[k] = v
        return rounded

    if not args.no_round:
        rounded_params = _round_numeric_params(params, decimals=2)
        # Log any changes due to rounding to help diagnose odd behaviors
        changes = {k: (params[k], rounded_params[k]) for k in params if params[k] != rounded_params[k]}
        if changes:
            print("\nApplied 2-decimal rounding to float parameters:")
            for k, (before, after) in changes.items():
                print(f"  {k}: {before} -> {after}")
        params = rounded_params

    # Print detailed info
    print(f"\n{'='*80}")
    print(f"BACKTEST CONFIGURATION: {strategy_name}")
    print(f"{'='*80}")
    print(f"\nData:")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Data points: {len(data)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    src_label = param_source_used
    if param_source_used in ("auto", "opt") and param_source_path:
        src_label += f" ({param_source_path})"
    elif param_source_used in ("yaml", "best_yaml"):
        src_label += " (YAML)"
    print(f"\nParameters source: {src_label}")
    print(f"Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nAccount Settings:")
    for k, v in main_conf["account"].items():
        print(f"  {k}: {v}")
    print(f"\nMax lookback/warmup: {infer_max_lookback(strategy_name, params)} bars")
    print(f"{'='*80}\n")

    max_lb = infer_max_lookback(strategy_name, params)
    res = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=main_conf["account"],
        max_lookback=max_lb,
        progress={"desc": f"Backtesting {strategy_name}", "position": 0, "leave": True}
    )

    # If params came from optimization JSON and it included an account snapshot, surface diffs
    account_cfg_used = main_conf["account"]
    account_diff = None
    if param_source_path and os.path.exists(param_source_path) and param_source_path.endswith("_best.json"):
        try:
            with open(param_source_path, "r") as f:
                best_blob = json.load(f)
            snap = best_blob.get("account_cfg")
            if isinstance(snap, dict):
                diffs = {}
                for k in sorted(set(list(snap.keys()) + list(account_cfg_used.keys()))):
                    if snap.get(k) != account_cfg_used.get(k):
                        diffs[k] = {"opt": snap.get(k), "eval": account_cfg_used.get(k)}
                if diffs:
                    account_diff = diffs
                    print("\nWARNING: Account config differs from optimization snapshot:")
                    for k, vv in diffs.items():
                        print(f"  {k}: opt={vv['opt']} eval={vv['eval']}")
        except Exception:
            pass

    payload = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "param_source": param_source_used,
        "param_source_path": param_source_path,
        "account_cfg_used": account_cfg_used,
        **({"account_cfg_diff": account_diff} if account_diff else {}),
        "params": params,
        "metrics": {
            "starting_balance": res["starting_balance"],
            "ending_balance": res["ending_balance"],
            "total_return_pct": res["total_return_pct"],
            "sharpe": res["sharpe"],
            "max_drawdown_pct": res["max_drawdown_pct"],
            "trades": res["trades"],
            "win_rate_pct": res["win_rate_pct"],
        },
        "equity_curve": res["equity_curve"],
        # Include per-trade details: entry/exit time & price, SL/TP, size, and PnL
        "trades_detail": res.get("trades_detail", []),
    }

    # Determine output path: results/<strategy>/evaluations/full_dataset_backtest.json by default
    if args.output:
        out_path = args.output
    else:
        # Save evaluations under the same results root used for reading params (mirror structure)
        out_dir = os.path.join(results_root, results_dir, "evaluations")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "full_dataset_backtest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
