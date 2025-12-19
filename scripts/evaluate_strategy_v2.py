#!/usr/bin/env python3
"""
Evaluate a strategy end-to-end on full data using selected params (aligned with v2 optimizer).
- Uses same backtest path as optimizer (alignment).
- Reads params from v2 best artifact by default, with fallbacks.
- Saves rich JSON with metrics, equity, trades_detail, and provenance.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy
from scripts.utils.reward_v2 import extract_equity_array, reward_with_constraints
from scripts.utils.strategy_utils_v2 import infer_max_lookback


def to_snake(name: str) -> str:
    import re
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.replace('__', '_').lower()


def load_strategy_class(strategy_yaml: Path):
    with open(strategy_yaml, 'r') as f:
        sc = yaml.safe_load(f)
    mod = sc['strategy'].get('module', 'functions.strategies')
    cls = sc['strategy']['class']
    m = importlib.import_module(mod)
    StrategyCls = getattr(m, cls)
    return sc, StrategyCls


def select_params_v2(results_root: Path, strategy_folder: str) -> Tuple[Dict[str, Any], Path | None]:
    """Prefer v2 best file, then v1 WF best, then other fallbacks."""
    opt_dir = results_root / strategy_folder / 'optimizations'
    cands = [
        opt_dir / 'bayesian_wf_v2_best.json',
        opt_dir / 'bayesian_wf_optimization_results_best.json',
        opt_dir / 'bayesian_optimization_results_best.json',
        opt_dir / 'optimization_results_best.json',
    ]
    for p in cands:
        if p.exists():
            try:
                blob = json.loads(Path(p).read_text())
                params = blob.get('params')
                if isinstance(params, dict) and params:
                    return params, p
            except Exception:
                pass
    # CSV fallback
    csv_path = opt_dir / 'bayesian_wf_v2_results.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if 'objective' in df.columns:
                df = df.sort_values('objective', ascending=False)
            row = df.iloc[0]
            params = {c.replace('param_', ''): row[c] for c in df.columns if c.startswith('param_')}
            return params, csv_path
        except Exception:
            pass
    raise FileNotFoundError("No v2 or legacy optimization artifacts found for params selection.")


def main():
    ap = argparse.ArgumentParser(description="Evaluate strategy v2 (aligned)")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--strategy_config", help="Path to strategy YAML; auto-resolve if omitted")
    ap.add_argument("--main_config", default="configs/main_config.v2.yaml")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--output", help="Explicit output JSON path")
    ap.add_argument("--params_source", choices=["auto","yaml"], default="auto", help="Use YAML best_params or optimization artifacts (auto)")
    args = ap.parse_args()

    strat_cfg = Path(args.strategy_config) if args.strategy_config else None
    if not strat_cfg:
        stem = to_snake(args.strategy)
        for c in [
            Path("functions/configs") / f"{stem}.yaml",
            Path("functions/configs") / f"{stem.replace('strategy','_strategy')}.yaml",
        ]:
            if c.exists():
                strat_cfg = c
                break
    if not strat_cfg or not strat_cfg.exists():
        print(f"ERROR: Strategy config not found for {args.strategy}")
        sys.exit(1)

    with open(args.main_config, 'r') as f:
        main_conf = yaml.safe_load(f)

    symbol = main_conf['general']['default_symbol']
    timeframe = main_conf['general']['default_timeframe']
    account_cfg = main_conf['account']

    strat_yaml, StrategyCls = load_strategy_class(strat_cfg)
    strategy_folder = to_snake(strat_yaml.get('strategy',{}).get('class', args.strategy))

    if args.params_source == 'yaml':
        params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best')
        if not isinstance(params, dict) or not params:
            raise ValueError("YAML best_params not found or empty.")
        params_src = None
    else:
        params, params_src = select_params_v2(Path(args.results_root), strategy_folder)

    # Load data and run backtest
    data = load_data(symbol, timeframe)
    max_lb = infer_max_lookback(params)
    res = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=account_cfg,
        max_lookback=max_lb,
        progress={"desc": f"Backtesting {args.strategy}", "position": 0, "leave": True},
    )

    # Compute v2 reward on full set for reference (not selection)
    k = float(main_conf["reward"].get("k", 5.0))
    eps = float(main_conf["reward"].get("eps", 1e-12))
    max_dd_frac = float(main_conf["reward"].get("max_dd_frac_of_start", 0.2))
    eq = extract_equity_array(res)
    v2_reward = float(reward_with_constraints(eq, max_dd_frac_of_start=max_dd_frac, k=k, eps=eps)) if eq.size else float("-inf")

    payload = {
        "strategy": args.strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "param_source": "yaml" if args.params_source=='yaml' else "opt",
        "param_source_path": str(params_src) if params_src else None,
        "params": params,
        "account_cfg_used": account_cfg,
        "metrics": {
            "starting_balance": res.get("starting_balance"),
            "ending_balance": res.get("ending_balance"),
            "total_return_pct": res.get("total_return_pct"),
            "sharpe": res.get("sharpe"),
            "max_drawdown_pct": res.get("max_drawdown_pct"),
            "trades": res.get("trades"),
            "win_rate_pct": res.get("win_rate_pct"),
            "v2_reward": v2_reward,
        },
        "equity_curve": res.get("equity_curve", []),
        "trades_detail": res.get("trades_detail", []),
        "signal_debug": res.get("signal_debug", []),
    }

    out_path = Path(args.output) if args.output else (Path(args.results_root) / strategy_folder / 'evaluations' / 'full_backtest_v2.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
