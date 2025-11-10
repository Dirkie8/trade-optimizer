#!/usr/bin/env python3
"""Live trading bot orchestrator (prototype).

Loads optimized parameters, fetches recent bars from MT5, runs selected strategy,
calculates lot size respecting risk + leverage + broker min, and sends orders.

DISCLAIMER: This is a prototype. Test in demo environment first. Use at your own risk.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import pandas as pd

# Project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.mt5_adapter import MT5Adapter
from scripts.utils.data_utils import pips_to_price, pip_size

# Import base strategy module dynamic later


def to_snake(name: str) -> str:
    import re
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.replace('__', '_').lower()


def infer_max_lookback(params: Dict[str, Any]) -> int:
    candidates = [int(v) for k, v in params.items() if isinstance(v, (int, float)) and any(tok in k.lower() for tok in ("period", "window", "lookback"))]
    return max(candidates) + 50 if candidates else 250


def load_best_params(results_root: str, results_dir: str) -> Dict[str, Any]:
    opt_dir = Path(results_root) / results_dir / 'optimizations'
    wf_best = opt_dir / 'bayesian_wf_optimization_results_best.json'
    legacy_best = opt_dir / 'bayesian_optimization_results_best.json'
    std_best = opt_dir / 'optimization_results_best.json'
    for cand in (wf_best, legacy_best, std_best):
        if cand.exists():
            with open(cand, 'r') as f:
                data = json.load(f)
            return data['params']
    raise FileNotFoundError(f"No best parameter file found in {opt_dir}")


def lot_size_from_risk(balance: float, risk_per_trade: float, entry: float, stop: float, symbol: str, leverage: float, min_lot: float, max_lot_theoretical: float) -> float:
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0
    risk_amount = balance * risk_per_trade
    # For FX: nominal per full lot ≈ 100000 units. Risk per pip = pip_value * lot_size.
    # Approx lot size using price-based risk: risk / (stop_dist * 100000)
    lot = risk_amount / (stop_dist * 100000)
    lot = max(lot, min_lot)
    lot = min(lot, max_lot_theoretical)
    # Round to 0.01 step
    return math.floor(lot * 100) / 100.0


def theoretical_max_lots(balance: float, leverage: float, price: float) -> float:
    # max_lots = balance * leverage / (100000 * price)
    if price <= 0:
        return 0.0
    return balance * leverage / (100000.0 * price)


def run_strategy_once(strategy_cfg_path: str, strategy_section: Dict[str, Any], global_cfg: Dict[str, Any], mt5a: MT5Adapter, risk_cfg: Dict[str, Any]):
    with open(strategy_cfg_path, 'r') as f:
        strat_yaml = yaml.safe_load(f)
    strat_info = strat_yaml['strategy']
    strategy_class_name = strat_info['class']
    module_path = strat_info.get('module', 'functions.strategies')
    results_dir = strat_info.get('results_dir', to_snake(strategy_class_name))

    symbol = strategy_section.get('symbol') or global_cfg['general']['default_symbol']
    timeframe = strategy_section.get('timeframe') or global_cfg['general']['default_timeframe']

    params = load_best_params(global_cfg['bot_results_root'], results_dir)

    # Dynamic import
    mod = __import__(module_path, fromlist=[strategy_class_name])
    StrategyCls = getattr(mod, strategy_class_name)

    # Determine warmup length
    warmup = infer_max_lookback(params)

    # Fetch bars
    df = mt5a.fetch_recent_bars(symbol, timeframe, count=warmup + 5)

    # Instantiate and get signal on full window (bar-close logic)
    strat = StrategyCls(df, params)
    action, sl_pips, tp_pips = strat.generate_signals()

    if action not in ("BUY", "SELL") or not sl_pips or not tp_pips:
        return {"status": "no_signal"}

    tick = mt5a.current_tick(symbol)
    bid = tick['bid']
    ask = tick['ask']
    mid = (bid + ask) / 2.0

    sl_price = None
    tp_price = None
    pip = pip_size(symbol)

    if action == 'BUY':
        entry_price = ask
        sl_price = entry_price - sl_pips * pip
        tp_price = entry_price + tp_pips * pip
    else:
        entry_price = bid
        sl_price = entry_price + sl_pips * pip
        tp_price = entry_price - tp_pips * pip

    balance = global_cfg['account_balance_placeholder']  # replaced at runtime maybe via mt5
    # If account info is available, override
    try:
        acc = mt5a.current_tick(symbol)  # placeholder; real account retrieval below
    except Exception:
        pass

    # For now, get equity from MT5 if available
    import MetaTrader5 as mt5
    ai = mt5.account_info()
    if ai:
        balance = ai.balance

    lev = risk_cfg.get('leverage', 100)
    theoretical_cap = theoretical_max_lots(balance, lev, entry_price)
    lot = lot_size_from_risk(balance, risk_cfg['risk_per_trade'], entry_price, sl_price, symbol, lev, 0.01, theoretical_cap)
    if lot <= 0:
        return {"status": "invalid_lot"}

    # Slippage points: convert pips to points (MT5 points depend on digits). Use symbol_info points for realism later.
    slippage_points = int(risk_cfg.get('slippage_pips', 0.1) * 10)  # crude mapping

    result = mt5a.send_market_order(
        symbol=symbol,
        volume=lot,
        order_type=action,
        price=entry_price,
        sl=sl_price,
        tp=tp_price,
        slippage_points=slippage_points,
        magic=global_cfg['magic_number'],
        comment=f"bot:{results_dir}:{timeframe}"
    )
    return {"status": "sent", "order": result, "lot": lot, "action": action, "entry": entry_price, "sl": sl_price, "tp": tp_price}


def main():
    parser = argparse.ArgumentParser(description="Live trading bot (prototype)")
    parser.add_argument('--config', default='configs/trading_bot_config.yaml')
    parser.add_argument('--once', action='store_true', help='Run one pass per enabled strategy then exit')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        bot_cfg = yaml.safe_load(f)

    # Augment for ease of passing
    general_defaults = {
        'general': {
            'default_symbol': bot_cfg['strategies'][0].get('symbol', 'frxEURUSD'),
            'default_timeframe': bot_cfg['strategies'][0].get('timeframe', '15m')
        },
        'bot_results_root': bot_cfg['runtime']['results_root'],
        'magic_number': bot_cfg['account']['magic_number'],
        'account_balance_placeholder': 100.0,
    }

    mt5a = MT5Adapter(login=bot_cfg['account'].get('login'), password=bot_cfg['account'].get('password'), server=bot_cfg['account'].get('server'))
    if not mt5a.initialize():
        print('Failed to initialize MT5')
        return

    try:
        risk_cfg = bot_cfg['risk']
        strategies = [s for s in bot_cfg['strategies'] if s.get('enabled')]
        if not strategies:
            print('No enabled strategies.')
            return

        while True:
            for strat in strategies:
                try:
                    res = run_strategy_once(strat['strategy_config'], strat, general_defaults, mt5a, risk_cfg)
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {strat['name']} -> {res['status']} {res.get('action','')} lot={res.get('lot','')} entry={res.get('entry','')}")
                except Exception as e:
                    print(f"Error running strategy {strat['name']}: {e}")
            if args.once:
                break
            time.sleep(bot_cfg['runtime']['poll_interval_seconds'])
    finally:
        mt5a.shutdown()


if __name__ == '__main__':
    main()
