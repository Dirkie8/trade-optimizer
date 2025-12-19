#!/usr/bin/env python3
"""
Bayesian optimization v2 with walk-forward validation and robust equity reward.
- Strict validation separation: hold-out set unseen during optimization.
- Unified call path for backtesting (same as evaluator).
- Reward: equity_reward with hard constraints, aggregated across folds with configurable mode.
"""
from __future__ import annotations
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

# Ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy
from scripts.utils.reward_v2 import (
    reward_with_constraints,
    extract_equity_array,
    aggregate_rewards,
)
from scripts.utils.strategy_utils_v2 import infer_max_lookback

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception as e:
    print("ERROR: optuna is required. pip install optuna")
    raise


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


def run_wf_backtest(
    StrategyCls,
    params: Dict[str, Any],
    account_cfg: Dict[str, Any],
    data: pd.DataFrame,
    n_folds: int,
) -> Dict[str, Any]:
    """Walk-forward over training data; return fold rewards and summary metrics."""
    n = len(data)
    max_lb = infer_max_lookback(params)
    if n < max_lb + max(50, n_folds):
        # Too little data; do a single backtest on entire train
        res = backtest_strategy(
            data=data,
            symbol="",
            strategy_cls=StrategyCls,
            params=params,
            account_cfg=account_cfg,
            max_lookback=max_lb,
            progress=None,
        )
        return {"fold_results": [res]}

    fold_sz = n // n_folds
    fold_results: List[Dict[str, Any]] = []
    for fold in range(1, n_folds + 1):
        start = fold_sz * (fold - 1)
        end = fold_sz * fold
        if fold == 1:
            continue  # skip first fold (no train history)
        test_df = data.iloc[start:end]
        if len(test_df) <= max_lb:
            continue
        try:
            res = backtest_strategy(
                data=test_df,
                symbol="",
                strategy_cls=StrategyCls,
                params=params,
                account_cfg=account_cfg,
                max_lookback=max_lb,
                progress=None,
            )
            fold_results.append(res)
        except Exception as e:
            # skip failed folds
            continue
    return {"fold_results": fold_results}


def main():
    ap = argparse.ArgumentParser(description="Bayesian optimization v2 (WF + robust reward)")
    ap.add_argument("--strategy", required=True, help="Strategy class name, e.g., RSIStrategy")
    ap.add_argument("--strategy_config", help="Path to strategy YAML; auto-resolve if omitted")
    ap.add_argument("--main_config", default="configs/main_config.v2.yaml")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--n_folds", type=int, default=None, help="Override folds; default from main_config.v2")
    ap.add_argument("--validation_ratio", type=float, default=None, help="Override validation ratio; default from main_config.v2")
    ap.add_argument("--seed", type=int, default=None, help="Override seed; default from main_config.v2")
    args = ap.parse_args()

    # Resolve strategy YAML path
    strat_cfg = Path(args.strategy_config) if args.strategy_config else None
    if not strat_cfg:
        stem = to_snake(args.strategy)
        # try multiple forms
        cands = [
            Path("functions/configs") / f"{stem}.yaml",
            Path("functions/configs") / f"{stem.replace('strategy','_strategy')}.yaml",
        ]
        for c in cands:
            if c.exists():
                strat_cfg = c
                break
    if not strat_cfg or not strat_cfg.exists():
        print(f"ERROR: Strategy config not found for {args.strategy}")
        sys.exit(1)

    # Load configs
    with open(args.main_config, 'r') as f:
        main_conf = yaml.safe_load(f)

    symbol = main_conf["general"]["default_symbol"]
    timeframe = main_conf["general"]["default_timeframe"]

    # Optimization settings
    n_folds = int(args.n_folds or main_conf["optimization"]["n_folds"])
    validation_ratio = float(args.validation_ratio or main_conf["optimization"]["validation_ratio"])
    seed = int(args.seed or main_conf["optimization"]["seed"])

    fold_mode = main_conf["optimization"].get("fold_aggregation", "mean_var_penalty")
    penalty_lambda = float(main_conf["optimization"].get("variance_penalty_lambda", 0.0))

    # Reward settings
    k = float(main_conf["reward"].get("k", 5.0))
    eps = float(main_conf["reward"].get("eps", 1e-12))
    max_dd_frac = float(main_conf["reward"].get("max_dd_frac_of_start", 0.2))

    # Account config shared with evaluator
    account_cfg = main_conf.get("account", {})

    # Load strategy class
    strat_yaml, StrategyCls = load_strategy_class(strat_cfg)

    # Load data once
    data = load_data(symbol, timeframe)
    split = int(len(data) * (1.0 - validation_ratio))
    train_df = data.iloc[:split].copy()
    val_df = data.iloc[split:].copy()

    # Parameter space
    if 'parameters_bayesian' not in strat_yaml:
        print("ERROR: parameters_bayesian missing in strategy YAML")
        sys.exit(1)
    param_ranges: Dict[str, Any] = strat_yaml['parameters_bayesian']

    # Optuna study
    sampler = TPESampler(seed=seed, n_startup_trials=min(10, max(1, args.n_trials // 5)))
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{args.strategy}_v2_wf")

    all_trials: List[Dict[str, Any]] = []

    def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, rng in param_ranges.items():
            if isinstance(rng, list) and len(rng) == 2:
                lo, hi = rng
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = trial.suggest_int(name, int(lo), int(hi))
                else:
                    params[name] = trial.suggest_float(name, float(lo), float(hi))
            else:
                # categorical or fixed
                if isinstance(rng, (list, tuple)):
                    params[name] = trial.suggest_categorical(name, list(rng))
                else:
                    params[name] = rng
        return params

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        wf = run_wf_backtest(StrategyCls, params, account_cfg, train_df, n_folds)
        fold_results = wf.get('fold_results', [])
        if not fold_results:
            return -1e10
        rewards: List[float] = []
        for res in fold_results:
            eq = extract_equity_array(res)
            r = reward_with_constraints(eq, max_dd_frac_of_start=max_dd_frac, k=k, eps=eps)
            if not np.isfinite(r):
                rewards.append(-1e10)
            else:
                rewards.append(float(r))
        agg = aggregate_rewards(rewards, mode=fold_mode, penalty_lambda=penalty_lambda)
        # Store lightweight row
        all_trials.append({
            **{f"param_{k}": v for k, v in params.items()},
            "fold_rewards": rewards,
            "objective": float(agg) if np.isfinite(agg) else -1e10,
        })
        return float(agg) if np.isfinite(agg) else -1e10

    print("Starting v2 Bayesian optimization...")
    with tqdm(total=args.n_trials, desc="Opt v2", unit="trial") as pbar:
        def cb(st: optuna.Study, tr: optuna.trial.FrozenTrial):
            pbar.update(1)
            pbar.set_postfix(best=f"{st.best_value:.4f}")
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, callbacks=[cb], show_progress_bar=False)

    print("Done. Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best objective (train folds): {study.best_value:.6f}")

    # Single final validation
    max_lb = infer_max_lookback(study.best_params)
    val_res = backtest_strategy(
        data=val_df,
        symbol="",
        strategy_cls=StrategyCls,
        params=study.best_params,
        account_cfg=account_cfg,
        max_lookback=max_lb,
        progress=None,
    )
    val_eq = extract_equity_array(val_res)
    val_reward = reward_with_constraints(val_eq, max_dd_frac_of_start=max_dd_frac, k=k, eps=eps)

    # Save artifacts
    strat_folder = to_snake(strat_yaml.get('strategy',{}).get('class', args.strategy))
    out_dir = Path('results') / strat_folder / 'optimizations'
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_trials).to_csv(out_dir / 'bayesian_wf_v2_results.csv', index=False)

    best_json = {
        'params': study.best_params,
        'train_objective': float(study.best_value),
        'validation_objective': float(val_reward) if np.isfinite(val_reward) else -1e10,
        'symbol': symbol,
        'timeframe': timeframe,
        'account_cfg': account_cfg,
        'data_split': {
            'train': {
                'start': train_df.index[0].isoformat() if len(train_df) else None,
                'end': train_df.index[-1].isoformat() if len(train_df) else None,
            },
            'validation': {
                'start': val_df.index[0].isoformat() if len(val_df) else None,
                'end': val_df.index[-1].isoformat() if len(val_df) else None,
            },
        },
        'reward': {
            'mode': fold_mode,
            'variance_penalty_lambda': penalty_lambda,
            'k': k,
            'eps': eps,
            'max_dd_frac_of_start': max_dd_frac,
        },
    }
    with open(out_dir / 'bayesian_wf_v2_best.json', 'w') as f:
        json.dump(best_json, f, indent=2)

    print(f"Saved v2 artifacts under {out_dir}")


if __name__ == "__main__":
    main()
