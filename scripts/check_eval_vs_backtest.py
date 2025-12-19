#!/usr/bin/env python3
"""
Compare evaluator JSON vs a fresh backtest using the same WF-best params to verify parity.
Usage:
  python scripts/check_eval_vs_backtest.py --eval results/keltner_squeeze_breakout/evaluations/full_dataset_backtest.json \
    --strategy_config functions/configs/keltner_squeeze_breakout.yaml

This will:
- Read param_source_path from the evaluator JSON (should point to bayesian_wf_optimization_results_best.json)
- Load those params and account_cfg_used from the evaluator JSON
- Load the same data and strategy class from YAML
- Run backtest_strategy with identical warmup
- Compare equity_curve, trades count, and key metrics
"""
import argparse
import json
import os
import sys

import yaml
import numpy as np

# Ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils.data_utils import load_data
from scripts.utils.backtest import backtest_strategy
from scripts.utils.strategy_utils_v2 import infer_max_lookback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True, help='Path to evaluator JSON (full_dataset_backtest.json)')
    ap.add_argument('--strategy_config', required=True, help='Path to strategy YAML used by evaluator')
    args = ap.parse_args()

    with open(args.eval, 'r') as f:
        eval_blob = json.load(f)
    ps_path = eval_blob.get('param_source_path')
    if not ps_path or not os.path.exists(ps_path):
        print(f"FAIL: evaluator JSON missing or invalid param_source_path: {ps_path}")
        sys.exit(2)
    with open(ps_path, 'r') as f:
        wf_best = json.load(f)
    params = wf_best.get('params') or {}
    if not isinstance(params, dict) or not params:
        print('FAIL: WF best JSON missing params')
        sys.exit(2)
    # Match evaluator behavior: round numeric float params to 2 decimals
    def _round_numeric_params(p: dict, decimals: int = 2) -> dict:
        out = {}
        for k, v in p.items():
            if isinstance(v, bool) or isinstance(v, int):
                out[k] = v
            else:
                try:
                    out[k] = round(float(v), decimals)
                except Exception:
                    out[k] = v
        return out
    params = _round_numeric_params(params, 2)

    # Strategy class
    with open(args.strategy_config, 'r') as f:
        strat_yaml = yaml.safe_load(f)
    cls_name = strat_yaml['strategy']['class']
    mod_path = strat_yaml['strategy'].get('module', 'functions.strategies')
    mod = __import__(mod_path, fromlist=[cls_name])
    StrategyCls = getattr(mod, cls_name)

    symbol = eval_blob['symbol']
    timeframe = eval_blob['timeframe']
    account_cfg_used = eval_blob.get('account_cfg_used') or {}
    data = load_data(symbol, timeframe)
    warmup = int(infer_max_lookback(params))

    res = backtest_strategy(
        data=data,
        symbol=symbol,
        strategy_cls=StrategyCls,
        params=params,
        account_cfg=account_cfg_used,
        max_lookback=warmup,
        progress=None,
    )

    # Compare equity curves
    eq_eval = eval_blob.get('equity_curve') or []
    eq_bt = res.get('equity_curve') or []
    same_len = len(eq_eval) == len(eq_bt)
    same_times = same_len and all(eq_eval[i]['time'] == eq_bt[i]['time'] for i in range(len(eq_eval)))
    same_vals = same_len and np.allclose([p['equity'] for p in eq_eval], [p['equity'] for p in eq_bt], rtol=1e-9, atol=1e-9)

    # Compare trades and key metrics
    trades_eval = int(eval_blob.get('metrics', {}).get('trades', 0))
    trades_bt = int(res.get('trades', 0))
    end_bal_eval = float(eval_blob.get('metrics', {}).get('ending_balance', 0.0))
    end_bal_bt = float(res.get('ending_balance', 0.0))
    ret_eval = float(eval_blob.get('metrics', {}).get('total_return_pct', 0.0))
    ret_bt = float(res.get('total_return_pct', 0.0))

    status = 'PASS' if (same_len and same_times and same_vals and trades_eval == trades_bt and np.isclose(end_bal_eval, end_bal_bt) and np.isclose(ret_eval, ret_bt)) else 'FAIL'
    print(f"Consistency: {status}")
    print(f"  equity_curve: len={same_len} times={same_times} values={same_vals}")
    print(f"  trades: eval={trades_eval} bt={trades_bt}")
    print(f"  ending_balance: eval={end_bal_eval} bt={end_bal_bt}")
    print(f"  total_return_pct: eval={ret_eval} bt={ret_bt}")

    if status == 'FAIL':
        # Print first mismatch index for visibility
        n = min(len(eq_eval), len(eq_bt))
        for i in range(n):
            if eq_eval[i]['time'] != eq_bt[i]['time'] or not np.isclose(eq_eval[i]['equity'], eq_bt[i]['equity']):
                print(f"  First equity mismatch at index {i}: eval={eq_eval[i]} bt={eq_bt[i]}")
                break
        if len(eq_eval) != len(eq_bt):
            print(f"  Equity length diff: eval={len(eq_eval)} bt={len(eq_bt)}")

if __name__ == '__main__':
    main()
