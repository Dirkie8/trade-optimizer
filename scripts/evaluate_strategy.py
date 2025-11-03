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
from typing import Any, Dict

import pandas as pd
import yaml

# Ensure project root is on sys.path so `scripts` and `functions` are importable when executed as a file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy
import importlib


def select_params(strategy_conf: Dict[str, Any], default_opt_dir: str, optimization_csv: str | None, selection_metric: str) -> Dict[str, Any]:
    # Prefer an explicit best.json under optimizations
    best_json_default = os.path.join(default_opt_dir, "optimization_results_best.json")
    if os.path.exists(best_json_default):
        try:
            with open(best_json_default, "r") as f:
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
    default_opt_dir = os.path.join("results", results_dir, "optimizations")
    params = select_params(strat_conf, default_opt_dir, args.optimization_csv, args.selection_metric)

    max_lb = infer_max_lookback(strategy_name, params)
    res = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=main_conf["account"],
        max_lookback=max_lb,
    )

    payload = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
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
    }

    # Determine output path: results/<strategy>/evaluations/full_dataset_backtest.json by default
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join("results", results_dir, "evaluations")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "full_dataset_backtest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
