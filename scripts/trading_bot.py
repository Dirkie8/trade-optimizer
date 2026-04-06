#!/usr/bin/env python3
"""Live trading bot orchestrator (enhanced).

Features:
 - Loads strategy best_params directly from YAML (no results sync required)
 - Candle-close only evaluation (drops forming bar, per-bar gating)
 - Alignment polling around candle boundaries with dense windows
 - Optional skip-start to wait for next closed candle before first evaluation
 - .env credential loading fallback if not set in config
 - Risk-based lot sizing with caps: free margin + per-trade margin budget (max_allowed_open_trades)
 - Min-lot skip (lot_below_min) instead of rounding up risking oversizing
 - Proactive margin requirement check (insufficient_margin status)
 - Symbol alias mapping (runtime.symbol_aliases)
 - Clean, concise logging with emojis and spread; no retcode clutter
 - CSV trade journal (leveraging, margin data) for later analysis
 - Account / risk / runtime pretty session header

DISCLAIMER: Prototype. Test thoroughly on demo before using live funds.
"""
from __future__ import annotations
import argparse
import json
import csv
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd
from dotenv import load_dotenv

# Project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.mt5_adapter import MT5Adapter
from scripts.utils.data_utils import pip_size, load_data, pips_to_price
from scripts.utils.backtest import backtest_strategy
from scripts.utils.strategy_utils_v2 import infer_max_lookback as infer_max_lookback_shared


def to_snake(name: str) -> str:
    import re
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.replace('__', '_').lower()


def _try_import_mt5():
    """Attempt to import MetaTrader5 dynamically; return module or None (for macOS replay)."""
    try:
        import importlib
        return importlib.import_module('MetaTrader5')
    except Exception:
        return None


def infer_max_lookback(params: Dict[str, Any]) -> int:
    """Delegate to shared inference so evaluator/bot use the same warmup."""
    try:
        return int(infer_max_lookback_shared(params))
    except Exception:
        # Fallback for extreme cases
        candidates = [int(v) for k, v in params.items() if isinstance(v, (int, float)) and any(tok in k.lower() for tok in ("period", "window", "lookback"))]
        return max(candidates) + 50 if candidates else 250


def _pip_size_from_mt5(symbol: str) -> float:
    mt5 = _try_import_mt5()
    if mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                digits = getattr(info, 'digits', None)
                point = getattr(info, 'point', None)
                if point and digits is not None:
                    if digits in (3, 5):
                        return point * 10.0
                    return float(point)
        except Exception:
            pass
    return pip_size(symbol)


def theoretical_max_lots(balance: float, leverage: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return balance * leverage / (100000.0 * price)


def theoretical_max_lots_from_free_margin(free_margin: float, leverage: float, price: float) -> float:
    if price <= 0 or free_margin is None:
        return 0.0
    return float(free_margin) * leverage / (100000.0 * price)


def theoretical_max_lots_from_margin_budget(margin_budget: float, leverage: float, price: float) -> float:
    if price <= 0 or margin_budget is None:
        return 0.0
    return float(margin_budget) * leverage / (100000.0 * price)


def timeframe_to_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith('m'):
        return int(tf[:-1])
    if tf.endswith('h'):
        return int(tf[:-1]) * 60
    if tf.endswith('d'):
        return int(tf[:-1]) * 1440
    raise ValueError(f'Unsupported timeframe {tf}')


def _server_time_utc(symbol: str) -> datetime:
    mt5 = _try_import_mt5()
    if mt5 is not None:
        try:
            t = mt5.symbol_info_tick(symbol)
            if t is not None:
                d = t._asdict()
                if d.get('time_msc'):
                    return datetime.fromtimestamp(d['time_msc']/1000.0, tz=timezone.utc)
                if d.get('time'):
                    return datetime.fromtimestamp(d['time'], tz=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


class ReplayAdapter:
    """Adapter that emulates MT5Adapter over historical data for replay mode."""
    def __init__(self, symbol: str, timeframe: str, data: pd.DataFrame, spread_pips: float = 0.2):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.spread_pips = float(spread_pips or 0.0)
        self.idx = 0
        self.open_positions = []

    def set_index(self, i: int):
        self.idx = max(0, min(int(i), len(self.data)-1))

    def ensure_symbol(self, symbol: str) -> bool:
        return True

    def fetch_recent_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        end = min(self.idx + 1, len(self.data))
        start = max(0, end - int(count))
        return self.data.iloc[start:end].copy()

    def current_tick(self, symbol: str) -> Dict[str, Any]:
        pip = _pip_size_from_mt5(symbol)
        close = float(self.data['Close'].iloc[self.idx])
        spread_px = self.spread_pips * pip
        ask = close + spread_px/2.0
        bid = close - spread_px/2.0
        return {'bid': bid, 'ask': ask}

    def positions_get(self, symbol: str):
        return self.open_positions

    def send_market_order(self, **order_request) -> Dict[str, Any]:
        return {'retcode': 10009, 'order': int(time.time()*1000) % 1000000, 'deal': int(time.time()*1000) % 1000000, **order_request}


def _replay_log(runtime: Dict[str, Any], msg: str, kind: str = 'info') -> None:
    """Log to terminal (minimal by default) and append to runtime.log_path.
    Terminal logging respects runtime['console_log']:
      - 'minimal' (default): prints only start/progress/end/warning/error
      - 'verbose': prints everything
    File logging always writes all messages.
    """
    console_level = str(runtime.get('console_log', 'minimal') or 'minimal').lower()
    print_to_console = (console_level == 'verbose') or (console_level == 'minimal' and kind in ('start','progress','end','warning','error'))
    if print_to_console:
        print(msg)
    try:
        log_path = runtime.get('log_path', 'logs/trading_bot.log')
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('a', encoding='utf-8') as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _append_trade_journal(csv_path: Path, row: Dict[str, Any]) -> None:
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            'ts_utc','strategy','symbol','timeframe','status','action','lot','entry','sl','tp',
            'leverage','balance','free_margin','margin_required','spread','ticket_order','ticket_deal'
        ]
        file_exists = csv_path.exists()
        with csv_path.open('a', encoding='utf-8') as f:
            if not file_exists:
                f.write(','.join(headers)+'\n')
            values = [
                str(row.get('ts_utc','')), str(row.get('strategy','')), str(row.get('symbol','')),
                str(row.get('timeframe','')), str(row.get('status','')), str(row.get('action','')),
                str(row.get('lot','')), str(row.get('entry','')), str(row.get('sl','')), str(row.get('tp','')),
                str(row.get('leverage','')), str(row.get('balance','')), str(row.get('free_margin','')),
                str(row.get('margin_required','')), str(row.get('spread','')),
                str(row.get('ticket_order','')), str(row.get('ticket_deal','')),
            ]
            f.write(','.join(values)+'\n')
    except Exception:
        pass


def _append_balance_journal(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single account balance/equity row to CSV. Creates parent dir if needed."""
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        headers = ['ts_utc','balance','equity','margin','margin_free','leverage','profit']
        file_exists = csv_path.exists()
        with csv_path.open('a', encoding='utf-8') as f:
            if not file_exists:
                f.write(','.join(headers)+'\n')
            values = [
                str(row.get('ts_utc','')),
                str(row.get('balance','')),
                str(row.get('equity','')),
                str(row.get('margin','')),
                str(row.get('margin_free','')),
                str(row.get('leverage','')),
                str(row.get('profit','')),
            ]
            f.write(','.join(values)+'\n')
    except Exception:
        pass


def _simulate_warmup_open_positions(
    *,
    data: pd.DataFrame,
    symbol: str,
    strategy_cls,
    params: Dict[str, Any],
    account_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate trades over historical bars and return open positions at the end.

    Uses parity_bot-style rules (entry on next bar open, SL before TP).
    """
    warmup = infer_max_lookback(params)
    if len(data) < warmup + 2:
        return {'open_positions': [], 'last_closed_bar_ts': None}

    starting_balance = float(account_cfg.get('starting_balance', 10000.0) or 10000.0)
    risk_frac = float(account_cfg.get('risk_per_trade', 0.01) or 0.01)
    spread_pips = float(account_cfg.get('spread_pips', 0.0) or 0.0)
    commission = float(account_cfg.get('commission_per_trade', 0.0) or 0.0)
    leverage = float(account_cfg.get('leverage', 0.0) or 0.0)
    slippage_pips = float(account_cfg.get('slippage_pips', 0.0) or 0.0)
    min_stop_pips = float(account_cfg.get('min_stop_pips', 0.0) or 0.0)
    min_size = float(account_cfg.get('min_size', 0.0) or 0.0)
    lot_step = float(account_cfg.get('lot_step', 0.0) or 0.0)
    max_lot_size = float(account_cfg.get('max_lot_size', 0.0) or 0.0)
    contract_size = float(account_cfg.get('contract_size', 100000.0) or 100000.0)
    equity_rounding = float(account_cfg.get('equity_rounding', 0.0) or 0.0)
    cooldown_bars_after_exit = int(account_cfg.get('cooldown_bars_after_exit', 0) or 0)
    no_same_bar_reentry = bool(account_cfg.get('no_same_bar_reentry', False))

    pip = pip_size(symbol)
    spread_price = pips_to_price(spread_pips, symbol)
    slippage_price = pips_to_price(slippage_pips, symbol)
    pvp_override = account_cfg.get('pip_value_per_lot')
    pip_value_per_lot = float(pvp_override) if pvp_override is not None else contract_size * pip

    equity = starting_balance
    open_positions: list[Dict[str, Any]] = []
    last_exit_bar_index = None

    idx = data.index
    for i in range(warmup, len(data) - 1):
        nxt = idx[i + 1]

        # Exit evaluation first, SL before TP
        if open_positions:
            h = float(data.iloc[i + 1]['High'])
            l = float(data.iloc[i + 1]['Low'])
            for pos in list(open_positions):
                exit_price = None
                if pos['direction'] == 'BUY':
                    if l <= pos['stop_price']:
                        exit_price = pos['stop_price'] - spread_price * 0.5 - slippage_price
                        exit_reason = 'SL'
                    elif h >= pos['tp_price']:
                        exit_price = pos['tp_price'] - spread_price * 0.5 - slippage_price
                        exit_reason = 'TP'
                else:  # SELL
                    if h >= pos['stop_price']:
                        exit_price = pos['stop_price'] + spread_price * 0.5 + slippage_price
                        exit_reason = 'SL'
                    elif l <= pos['tp_price']:
                        exit_price = pos['tp_price'] + spread_price * 0.5 + slippage_price
                        exit_reason = 'TP'
                if exit_price is not None:
                    direction_mult = 1.0 if pos['direction'] == 'BUY' else -1.0
                    price_diff = (exit_price - pos['entry_price'])
                    pips_signed = (price_diff / pip) * direction_mult if pip > 0 else 0.0
                    pnl = pips_signed * pip_value_per_lot * pos['size']
                    pnl -= commission
                    equity += pnl
                    if equity_rounding and equity_rounding > 0:
                        equity = round(equity / equity_rounding) * equity_rounding
                    open_positions.remove(pos)
                    last_exit_bar_index = i + 1

        # Open new trade using backtester rules
        if no_same_bar_reentry and last_exit_bar_index is not None and last_exit_bar_index == i + 1:
            continue
        if cooldown_bars_after_exit and last_exit_bar_index is not None:
            if (i + 1) - last_exit_bar_index < cooldown_bars_after_exit:
                continue

        window = data.iloc[:i + 1]
        strat = strategy_cls(window, params)
        action, sl_pips, tp_pips = strat.generate_signals()
        if action in ('BUY', 'SELL') and sl_pips is not None and tp_pips is not None:
            if min_stop_pips and float(sl_pips) < min_stop_pips:
                continue
            entry_open = float(data.iloc[i + 1]['Open'])
            if action == 'BUY':
                entry_price = entry_open + spread_price * 0.5 + slippage_price
                stop_price = entry_price - pips_to_price(sl_pips, symbol)
                tp_price = entry_price + pips_to_price(tp_pips, symbol)
            else:
                entry_price = entry_open - spread_price * 0.5 - slippage_price
                stop_price = entry_price + pips_to_price(sl_pips, symbol)
                tp_price = entry_price - pips_to_price(tp_pips, symbol)

            stop_dist = abs(entry_price - stop_price)
            if stop_dist <= 0:
                continue
            risk_amount = equity * risk_frac
            stop_pips_eff = float(sl_pips)
            size = max(risk_amount / (stop_pips_eff * pip_value_per_lot), 0.0)
            if leverage and leverage > 0 and entry_price > 0:
                allowable = (equity * leverage) / (contract_size * entry_price)
                if allowable <= 0:
                    continue
                size = min(size, allowable)
            if lot_step and lot_step > 0:
                size = (size // lot_step) * lot_step
            if max_lot_size and max_lot_size > 0 and size > max_lot_size:
                size = max_lot_size
            if min_size and size < min_size:
                continue

            equity -= commission
            if equity_rounding and equity_rounding > 0:
                equity = round(equity / equity_rounding) * equity_rounding
            open_positions.append({
                'direction': action,
                'entry_time': nxt,
                'entry_price': float(entry_price),
                'stop_price': float(stop_price),
                'tp_price': float(tp_price),
                'size': float(size),
            })

    last_closed_bar_ts = data.index[-1] if len(data) > 0 else None
    return {'open_positions': open_positions, 'last_closed_bar_ts': last_closed_bar_ts}


def _update_simulated_positions_for_bar(
    positions: list[Dict[str, Any]],
    *,
    bar_high: float,
    bar_low: float,
    spread_price: float,
    slippage_price: float,
) -> list[Dict[str, Any]]:
    """Update simulated positions for a single closed bar and return remaining positions."""
    remaining: list[Dict[str, Any]] = []
    for pos in positions:
        exit_price = None
        if pos['direction'] == 'BUY':
            if bar_low <= pos['stop_price']:
                exit_price = pos['stop_price'] - spread_price * 0.5 - slippage_price
            elif bar_high >= pos['tp_price']:
                exit_price = pos['tp_price'] - spread_price * 0.5 - slippage_price
        else:
            if bar_high >= pos['stop_price']:
                exit_price = pos['stop_price'] + spread_price * 0.5 + slippage_price
            elif bar_low <= pos['tp_price']:
                exit_price = pos['tp_price'] + spread_price * 0.5 + slippage_price
        if exit_price is None:
            remaining.append(pos)
    return remaining


def run_strategy_once(
    strategy_cfg_path: str,
    strategy_section: Dict[str, Any],
    global_cfg: Dict[str, Any],
    mt5a: MT5Adapter,
    risk_cfg: Dict[str, Any],
    *,
    dry_run: bool = False,
    max_positions: int | None = None,
    last_closed_bar_ts: pd.Timestamp | None = None,
    apply_param_rounding: bool = False,
    round_params_decimals: int | None = None,
):
    mt5 = _try_import_mt5()
    with open(strategy_cfg_path, 'r') as f:
        strat_yaml = yaml.safe_load(f)
    strat_info = strat_yaml['strategy']
    strategy_class_name = strat_info['class']
    module_path = strat_info.get('module', 'functions.strategies')

    symbol = strategy_section.get('symbol') or global_cfg['general']['default_symbol']
    timeframe = strategy_section.get('timeframe') or global_cfg['general']['default_timeframe']
    mt5a.ensure_symbol(symbol)

    params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best')
    if not isinstance(params, dict) or not params:
        return {'status': 'no_params'}

    # Optional: scale pips-based parameters if the strategy YAML declares a scale.
    # ADJUSTED RULE: Only scale values >= 1.0. Fractional pip values (<1) are
    # treated as-is so that a config value like 0.11 produces ~0.11 pips
    # distance (e.g. EURUSD: 0.11 * 0.0001 = 0.000011 ≈ 1 pipette), matching
    # user expectation (entry 1.15623 -> SL 1.15622 when rounded to 5 decimals).
    pips_scale = float(strat_yaml.get('pips_param_scale', 1.0) or 1.0)
    pips_keys = strat_yaml.get('pips_param_keys') or ['stop_loss_pips', 'take_profit_pips']
    if pips_scale != 1.0:
        params = params.copy()
        for k in pips_keys:
            if k in params and isinstance(params[k], (int, float)):
                v = float(params[k])
                if v >= 1.0:
                    params[k] = v * pips_scale

    # Optional rounding to align with evaluator behavior
    if apply_param_rounding and isinstance(round_params_decimals, (int, float)):
        def _round_numeric_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for kk, vv in p.items():
                if isinstance(vv, bool) or isinstance(vv, int):
                    out[kk] = vv
                else:
                    try:
                        out[kk] = round(float(vv), int(decimals))
                    except Exception:
                        out[kk] = vv
            return out
        params = _round_numeric_params(params, int(round_params_decimals))

    mod = __import__(module_path, fromlist=[strategy_class_name])
    StrategyCls = getattr(mod, strategy_class_name)

    warmup = infer_max_lookback(params)
    df = mt5a.fetch_recent_bars(symbol, timeframe, count=warmup + 5)
    if len(df) < 2:
        return {'status': 'no_data'}
    df_closed = df.iloc[:-1]
    closed_ts = df_closed.index[-1]
    if last_closed_bar_ts is not None and pd.to_datetime(last_closed_bar_ts) == closed_ts:
        return {'status': 'no_new_bar', 'closed_bar_ts': closed_ts}

    strat = StrategyCls(df_closed, params)
    action, sl_pips, tp_pips = strat.generate_signals()
    if action not in ('BUY','SELL') or not sl_pips or not tp_pips:
        return {'status': 'no_signal', 'closed_bar_ts': closed_ts}

    # Enforce minimum stop distance in pips (config-level guard similar to backtester)
    try:
        min_stop_pips_cfg = float(risk_cfg.get('min_stop_pips', 0) or 0)
    except Exception:
        min_stop_pips_cfg = 0.0
    if min_stop_pips_cfg and float(sl_pips) < min_stop_pips_cfg:
        return {'status': 'min_stop_violation', 'closed_bar_ts': closed_ts}

    tick = mt5a.current_tick(symbol)
    bid = tick['bid']; ask = tick['ask']
    spread = ask - bid
    entry_price = ask if action=='BUY' else bid
    pip = _pip_size_from_mt5(symbol)
    if action=='BUY':
        sl_price = entry_price - sl_pips * pip
        tp_price = entry_price + tp_pips * pip
    else:
        sl_price = entry_price + sl_pips * pip
        tp_price = entry_price - tp_pips * pip

    ai = None
    try:
        if mt5 is not None:
            ai = mt5.account_info()
    except Exception:
        ai = None
    balance = global_cfg['account_balance_placeholder']
    free_margin = None
    if ai is not None:
        balance = getattr(ai,'equity',None) or getattr(ai,'balance',balance)
        free_margin = getattr(ai,'margin_free',None)

    # Position limit per symbol
    if max_positions is not None and max_positions>0:
        try:
            if mt5 is not None:
                open_pos = mt5.positions_get(symbol=symbol)
                if open_pos is not None and len(open_pos) >= max_positions:
                    return {'status':'position_limit','closed_bar_ts':closed_ts}
        except Exception:
            pass
    
    lev = risk_cfg.get('leverage',100)
    # Cap candidates: free margin and per-trade margin budget
    cap_candidates = []
    if free_margin is not None:
        cap_candidates.append(theoretical_max_lots_from_free_margin(free_margin, lev, entry_price))
    maot = risk_cfg.get('max_allowed_open_trades')
    if maot:
        try:
            per_trade_budget = balance / float(maot)
            effective_budget = min(per_trade_budget, free_margin) if free_margin is not None else per_trade_budget
            cap_candidates.append(theoretical_max_lots_from_margin_budget(effective_budget, lev, entry_price))
        except Exception:
            pass
    theoretical_cap = min(cap_candidates) if cap_candidates else theoretical_max_lots(balance, lev, entry_price)

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {'status':'invalid_stop','closed_bar_ts':closed_ts}
    risk_amount = balance * risk_cfg['risk_per_trade']
    raw_lot = risk_amount / (stop_dist * 100000)
    if raw_lot <= 0:
        return {'status':'invalid_lot','closed_bar_ts':closed_ts}
    lot = min(raw_lot, theoretical_cap)
    if lot < 0.01:
        return {'status':'lot_below_min','raw_lot':raw_lot,'cap':theoretical_cap,'closed_bar_ts':closed_ts,'entry':entry_price,'sl':sl_price,'tp':tp_price,'balance':balance,'leverage':lev,'spread':spread,'action':action}
    lot = math.floor(lot*100)/100.0
    if lot < 0.01:
        return {'status':'lot_below_min','raw_lot':raw_lot,'cap':theoretical_cap,'closed_bar_ts':closed_ts,'entry':entry_price,'sl':sl_price,'tp':tp_price,'balance':balance,'leverage':lev,'spread':spread,'action':action}

    # Margin requirement check
    margin_required = None
    try:
        if mt5 is not None:
            mt5_type = mt5.ORDER_TYPE_BUY if action=='BUY' else mt5.ORDER_TYPE_SELL
            margin_required = mt5.order_calc_margin(mt5_type, symbol, lot, entry_price)
    except Exception:
        pass
    if margin_required is None:
        try:
            margin_required = (lot * 100000.0 * entry_price) / float(lev)
        except Exception:
            margin_required = None
    if free_margin is not None and margin_required is not None and margin_required > free_margin:
        return {'status':'insufficient_margin','entry':entry_price,'sl':sl_price,'tp':tp_price,'lot':lot,'free_margin':free_margin,'margin_required':margin_required,'balance':balance,'leverage':lev,'spread':spread,'closed_bar_ts':closed_ts,'action':action}

    deviation_points = risk_cfg.get('deviation_points')
    if deviation_points is None:
        deviation_points = max(int(risk_cfg.get('slippage_pips',0.1)*10),20)

    order_request = {
        'symbol': symbol,
        'volume': lot,
        'order_type': action,
        'price': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'slippage_points': int(deviation_points),
        'magic': global_cfg['magic_number'],
        'comment': 'bot',
    }

    if dry_run:
        return {'status':'dry_run','order':order_request,'lot':lot,'action':action,'entry':entry_price,'sl':sl_price,'tp':tp_price,'spread':spread,'closed_bar_ts':closed_ts,'balance':balance,'free_margin':free_margin,'margin_required':margin_required}

    result = mt5a.send_market_order(**order_request)
    # Simple success heuristic
    retcode = result.get('retcode') if isinstance(result, dict) else None
    status = 'sent'
    if retcode is None or result.get('error'):
        status = 'order_error'
    return {'status':status,'order':result,'lot':lot,'action':action,'entry':entry_price,'sl':sl_price,'tp':tp_price,'spread':spread,'closed_bar_ts':closed_ts,'balance':balance,'free_margin':free_margin,'margin_required':margin_required}


def _run_replay_mode(bot_cfg, args):
    """Run trading bot in replay mode over historical data without MT5."""
    account = bot_cfg.get('account', {})
    general = bot_cfg.get('general', {})
    runtime = bot_cfg.get('runtime', {})
    strategies = [s for s in bot_cfg['strategies'] if s.get('enabled')]
    if not strategies:
        _replay_log(runtime, 'No enabled strategies for replay.', kind='warning')
        return

    apply_round = bool(runtime.get('apply_param_rounding', False))
    round_dec = int(runtime.get('round_params_decimals', 2))
    starting_balance = float(account.get('starting_balance', 100.0) or 100.0)

    def _load_data_with_aliases(symbol: str, timeframe: str) -> pd.DataFrame:
        # Try alias mapping from runtime if present
        aliases: Dict[str,str] = runtime.get('symbol_aliases', {}) or {}
        tried = set()
        candidates = []
        base = aliases.get(symbol, symbol)
        candidates.append(base)
        # Common variations between broker symbols and local data
        if base.endswith('m'):
            candidates.append(base[:-1])
        if not base.startswith('frx'):
            candidates.append('frx' + base)
        if base.endswith('m') and not base.startswith('frx'):
            candidates.append('frx' + base[:-1])
        # Ensure uniqueness and non-empty
        for sym in [s for s in candidates if s and s not in tried]:
            tried.add(sym)
            try:
                return load_data(sym, timeframe)
            except Exception:
                continue
        # Fallback raise using original
        return load_data(symbol, timeframe)

    for strat in strategies:
        symbol = strat.get('symbol') or general.get('default_symbol') or 'EURUSD'
        timeframe = strat.get('timeframe') or general.get('default_timeframe') or '15m'
        # Load data, trying common alias variations
        _replay_log(runtime, f"Replay: loading data for {symbol} {timeframe}...", kind='start')
        data = _load_data_with_aliases(symbol, timeframe)
        # Apply date filters using tz-aware timestamps
        try:
            start_ts = pd.to_datetime(args.replay_start, utc=True) if args.replay_start else None
            end_ts = pd.to_datetime(args.replay_end, utc=True) if args.replay_end else None
        except Exception:
            start_ts = None; end_ts = None
        filtered = data
        if start_ts is not None:
            filtered = filtered.loc[start_ts:]
        if end_ts is not None:
            filtered = filtered.loc[:end_ts]
        if len(filtered) < 100:
            _replay_log(runtime, f"Replay: insufficient data for requested range ({symbol} {timeframe}). Available: {data.index[0].isoformat()} .. {data.index[-1].isoformat()}", kind='warning')
            continue
        data = filtered

        adapter = ReplayAdapter(symbol, timeframe, data, spread_pips=float(account.get('spread_pips',0.2) or 0.2))
        # Load strategy YAML and params
        try:
            with open(strat['strategy_config'],'r') as f:
                strat_yaml = yaml.safe_load(f)
        except Exception as e:
            _replay_log(runtime, f"Replay: failed to load strategy YAML: {e}")
            continue
        params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best')
        if not isinstance(params, dict) or not params:
            _replay_log(runtime, 'Replay: no best_params found; skipping.')
            continue
        if apply_round:
            def _round_numeric_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for kk, vv in p.items():
                    if isinstance(vv, bool) or isinstance(vv, int):
                        out[kk] = vv
                    else:
                        try:
                            out[kk] = round(float(vv), int(decimals))
                        except Exception:
                            out[kk] = vv
                return out
            params = _round_numeric_params(params, round_dec)

        # Parity mode: use evaluator's backtest function for identical modeling
        if getattr(args, 'replay_model', 'parity') == 'parity':
            try:
                mod_path = strat_yaml['strategy'].get('module', 'functions.strategies')
                cls_name = strat_yaml['strategy']['class']
                mod = __import__(mod_path, fromlist=[cls_name])
                StrategyCls = getattr(mod, cls_name)
            except Exception as e:
                _replay_log(runtime, f"Replay parity: failed to load strategy class: {e}", kind='error')
                continue

            max_lb = infer_max_lookback(params)
            res = backtest_strategy(
                data=data,
                symbol=symbol,
                strategy_cls=StrategyCls,
                params=params,
                account_cfg=account,
                max_lookback=max_lb,
                progress=None,
            )

            payload = {
                'strategy': cls_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'param_source': 'yaml',
                'params': params,
                'metrics': {
                    'starting_balance': res.get('starting_balance'),
                    'ending_balance': res.get('ending_balance'),
                    'total_return_pct': res.get('total_return_pct'),
                    'sharpe': res.get('sharpe'),
                    'max_drawdown_pct': res.get('max_drawdown_pct'),
                    'trades': res.get('trades'),
                    'win_rate_pct': res.get('win_rate_pct'),
                },
                'equity_curve': res.get('equity_curve', []),
                'trades_detail': res.get('trades_detail', []),
                'signal_debug': res.get('signal_debug', []),
            }

            out_path = args.replay_output or os.path.join(runtime.get('results_root','results'), to_snake(strat_yaml['strategy'].get('results_dir') or strat_yaml['strategy']['class']), 'replay', 'full_dataset_backtest_bot.json')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            _replay_log(runtime, f"Replay saved to {out_path}", kind='end')
            # Next strategy
            continue

        # Parity_bot mode: run bot-style loop but apply the exact backtester rules
        if getattr(args, 'replay_model', 'parity') == 'parity_bot':
            try:
                mod_path = strat_yaml['strategy'].get('module', 'functions.strategies')
                cls_name = strat_yaml['strategy']['class']
                mod = __import__(mod_path, fromlist=[cls_name])
                StrategyCls = getattr(mod, cls_name)
            except Exception as e:
                _replay_log(runtime, f"Replay parity_bot: failed to load strategy class: {e}", kind='error')
                continue

            # Account controls
            starting_balance = float(account.get('starting_balance', 10000.0) or 10000.0)
            risk_frac = float(account.get('risk_per_trade', 0.01) or 0.01)
            spread_pips = float(account.get('spread_pips', 0.0) or 0.0)
            commission = float(account.get('commission_per_trade', 0.0) or 0.0)
            leverage = float(account.get('leverage', 0.0) or 0.0)
            slippage_pips = float(account.get('slippage_pips', 0.0) or 0.0)
            min_stop_pips = float(account.get('min_stop_pips', 0.0) or 0.0)
            min_size = float(account.get('min_size', 0.0) or 0.0)
            lot_step = float(account.get('lot_step', 0.0) or 0.0)
            max_lot_size = float(account.get('max_lot_size', 0.0) or 0.0)
            contract_size = float(account.get('contract_size', 100000.0) or 100000.0)
            equity_rounding = float(account.get('equity_rounding', 0.0) or 0.0)
            # Entry cooldown controls (align with backtester)
            cooldown_bars_after_exit = int(account.get('cooldown_bars_after_exit', 0) or 0)
            no_same_bar_reentry = bool(account.get('no_same_bar_reentry', False))

            pip = pip_size(symbol)
            spread_price = pips_to_price(spread_pips, symbol)
            slippage_price = pips_to_price(slippage_pips, symbol)
            pvp_override = account.get('pip_value_per_lot')
            pip_value_per_lot = float(pvp_override) if pvp_override is not None else contract_size * pip

            warmup = infer_max_lookback(params)
            equity = starting_balance
            peak_equity = starting_balance
            equity_curve = []
            open_positions = []  # list of dicts
            trades_detail = []
            last_exit_bar_index = None

            idx = data.index
            for i in range(warmup, len(data)-1):
                now = idx[i]
                nxt = idx[i+1]
                window = data.iloc[:i+1]
                equity_curve.append({'time': now.isoformat(), 'equity': equity})

                # Exit evaluation first, SL before TP
                if open_positions:
                    h = float(data.iloc[i+1]['High'])
                    l = float(data.iloc[i+1]['Low'])
                    for pos in list(open_positions):
                        exit_price = None
                        if pos['direction'] == 'BUY':
                            if l <= pos['stop_price']:
                                exit_price = pos['stop_price'] - spread_price * 0.5 - slippage_price
                                exit_reason = 'SL'
                            elif h >= pos['tp_price']:
                                exit_price = pos['tp_price'] - spread_price * 0.5 - slippage_price
                                exit_reason = 'TP'
                        else:  # SELL
                            if h >= pos['stop_price']:
                                exit_price = pos['stop_price'] + spread_price * 0.5 + slippage_price
                                exit_reason = 'SL'
                            elif l <= pos['tp_price']:
                                exit_price = pos['tp_price'] + spread_price * 0.5 + slippage_price
                                exit_reason = 'TP'
                        if exit_price is not None:
                            direction_mult = 1.0 if pos['direction'] == 'BUY' else -1.0
                            price_diff = (exit_price - pos['entry_price'])
                            pips_signed = (price_diff / pip) * direction_mult if pip > 0 else 0.0
                            pnl = pips_signed * pip_value_per_lot * pos['size']
                            pnl -= commission
                            equity += pnl
                            if equity_rounding and equity_rounding > 0:
                                equity = round(equity / equity_rounding) * equity_rounding
                            trades_detail.append({
                                'direction': pos['direction'],
                                'entry_time': pos['entry_time'].isoformat(),
                                'entry_price': pos['entry_price'],
                                'stop_price': pos['stop_price'],
                                'take_profit_price': pos['tp_price'],
                                'exit_time': nxt.isoformat(),
                                'exit_price': float(exit_price),
                                'size': float(pos['size']),
                                'pnl': pnl,
                                'equity_after': equity,
                                'exit_reason': exit_reason,
                            })
                            open_positions.remove(pos)
                            last_exit_bar_index = i + 1

                # Open new trade using backtester rules
                # Apply optional re-entry cooldown
                if no_same_bar_reentry and last_exit_bar_index is not None and last_exit_bar_index == i + 1:
                    continue
                if cooldown_bars_after_exit and last_exit_bar_index is not None:
                    if (i + 1) - last_exit_bar_index < cooldown_bars_after_exit:
                        continue
                strat = StrategyCls(window, params)
                action, sl_pips, tp_pips = strat.generate_signals()
                if action in ('BUY','SELL') and sl_pips is not None and tp_pips is not None:
                    # Minimum stop distance guard
                    if min_stop_pips and float(sl_pips) < min_stop_pips:
                        continue
                    entry_open = float(data.iloc[i+1]['Open'])
                    if action == 'BUY':
                        entry_price = entry_open + spread_price * 0.5 + slippage_price
                        stop_price = entry_price - pips_to_price(sl_pips, symbol)
                        tp_price = entry_price + pips_to_price(tp_pips, symbol)
                    else:
                        entry_price = entry_open - spread_price * 0.5 - slippage_price
                        stop_price = entry_price + pips_to_price(sl_pips, symbol)
                        tp_price = entry_price - pips_to_price(tp_pips, symbol)

                    stop_dist = abs(entry_price - stop_price)
                    if stop_dist <= 0:
                        continue
                    risk_amount = equity * risk_frac
                    stop_pips_eff = float(sl_pips)
                    size = max(risk_amount / (stop_pips_eff * pip_value_per_lot), 0.0)
                    size_pre_cap = size
                    if leverage and leverage > 0 and entry_price > 0:
                        allowable = (equity * leverage) / (contract_size * entry_price)
                        if allowable <= 0:
                            continue
                        size = min(size, allowable)
                    if lot_step and lot_step > 0:
                        size = (size // lot_step) * lot_step
                    if max_lot_size and max_lot_size > 0 and size > max_lot_size:
                        size = max_lot_size
                    if min_size and size < min_size:
                        # Simple skip (backtester attempts soft rounding tolerance; omitted for brevity)
                        continue

                    equity -= commission
                    if equity_rounding and equity_rounding > 0:
                        equity = round(equity / equity_rounding) * equity_rounding
                    open_positions.append({
                        'direction': action,
                        'entry_time': nxt,
                        'entry_price': float(entry_price),
                        'stop_price': float(stop_price),
                        'tp_price': float(tp_price),
                        'size': float(size),
                    })

            # Final equity point
            if len(data) > 0:
                equity_curve.append({'time': data.index[-1].isoformat(), 'equity': equity})

            # Metrics
            trades = len(trades_detail)
            wins = sum(1 for t in trades_detail if (t.get('pnl') or 0) > 0)
            win_rate_pct = (wins / trades) * 100.0 if trades else 0.0
            total_return_pct = ((equity - starting_balance) / starting_balance) * 100.0 if starting_balance else 0.0
            # Approximate max drawdown
            eq_vals = [pt['equity'] for pt in equity_curve]
            peaks = []
            max_dd = 0.0
            for v in eq_vals:
                peaks.append(max(v, peaks[-1] if peaks else v))
                dd = (peaks[-1] - v) / peaks[-1] if peaks[-1] > 0 else 0.0
                max_dd = max(max_dd, dd * 100.0)

            payload = {
                'strategy': cls_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'param_source': 'yaml',
                'params': params,
                'metrics': {
                    'starting_balance': starting_balance,
                    'ending_balance': equity,
                    'total_return_pct': total_return_pct,
                    'sharpe': 0.0,
                    'max_drawdown_pct': max_dd,
                    'trades': trades,
                    'win_rate_pct': win_rate_pct,
                },
                'equity_curve': equity_curve,
                'trades_detail': trades_detail,
            }

            out_path = args.replay_output or os.path.join(runtime.get('results_root','results'), to_snake(strat_yaml['strategy'].get('results_dir') or strat_yaml['strategy']['class']), 'replay', 'full_dataset_backtest_bot.json')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            _replay_log(runtime, f"Replay saved to {out_path}", kind='end')
            continue

        balance = starting_balance
        equity_curve = []
        trades_detail = []
        open_positions = []
        last_bar_seen = None
        warmup = infer_max_lookback(params)
        _replay_log(runtime, f"Replay: starting loop warmup={warmup}, bars={len(data)} from {data.index[0].isoformat()} to {data.index[-1].isoformat()}", kind='start')
        total_iters = max(warmup, 1)
        total_iters = len(data) - 1 - total_iters
        progress_step = max(total_iters // 20, 1)  # ~5% steps
        for i in range(max(warmup, 1), len(data)-1):
            adapter.set_index(i)
            res = run_strategy_once(
                strat['strategy_config'],
                {'symbol': symbol, **strat},
                {'general': {'default_symbol': symbol, 'default_timeframe': timeframe}, 'magic_number': account.get('magic_number',123456), 'account_balance_placeholder': starting_balance},
                adapter,
                account,
                dry_run=False,
                max_positions=account.get('max_concurrent_positions'),
                last_closed_bar_ts=last_bar_seen,
                apply_param_rounding=apply_round,
                round_params_decimals=round_dec,
            )
            if 'closed_bar_ts' in res and res['status'] not in ('no_data',):
                last_bar_seen = res['closed_bar_ts']

            if res.get('status') == 'sent':
                open_positions.append({
                    'action': res.get('action'),
                    'entry': float(res.get('entry')),
                    'sl': float(res.get('sl')),
                    'tp': float(res.get('tp')),
                    'lot': float(res.get('lot')),
                    'ts': data.index[i],
                })
                _replay_log(runtime, f"Replay: SENT {res.get('action')} lot={res.get('lot')} entry={round(float(res.get('entry')),5)} sl={round(float(res.get('sl')),5)} tp={round(float(res.get('tp')),5)} ts={data.index[i].isoformat()}", kind='debug')

            # Process exits on next bar (set-and-forget)
            bar_next = data.iloc[i+1]
            hi = float(bar_next['High']); lo = float(bar_next['Low'])
            pip = _pip_size_from_mt5(symbol)
            remaining = []
            for pos in open_positions:
                exit_price = None
                exit_reason = None
                if pos['action'] == 'BUY':
                    sl_hit = lo <= pos['sl']
                    tp_hit = hi >= pos['tp']
                    if sl_hit:
                        exit_price = pos['sl']; exit_reason = 'SL'
                    elif tp_hit:
                        exit_price = pos['tp']; exit_reason = 'TP'
                else:
                    sl_hit = hi >= pos['sl']
                    tp_hit = lo <= pos['tp']
                    if sl_hit:
                        exit_price = pos['sl']; exit_reason = 'SL'
                    elif tp_hit:
                        exit_price = pos['tp']; exit_reason = 'TP'
                if exit_price is None:
                    remaining.append(pos)
                else:
                    pips = (exit_price - pos['entry'])/pip
                    if pos['action'] == 'SELL':
                        pips = -pips
                    pnl = pips * float(account.get('pip_value_per_lot',10.0) or 10.0) * pos['lot']
                    balance += pnl
                    trades_detail.append({
                        'entry_time': pos['ts'].isoformat(),
                        'exit_time': bar_next.name.isoformat(),
                        'entry_price': pos['entry'],
                        'exit_price': exit_price,
                        'size_lot': pos['lot'],
                        'action': pos['action'],
                        'pnl_currency': pnl,
                        'exit_reason': exit_reason,
                    })
                    _replay_log(runtime, f"Replay: EXIT {pos['action']} {exit_reason} pnl={round(pnl,2)} bal={round(balance,2)} entry={round(pos['entry'],5)} exit={round(exit_price,5)} ts={bar_next.name.isoformat()}", kind='debug')
            open_positions = remaining
            equity_curve.append({'time': bar_next.name.isoformat(), 'equity': round(balance,2)})

            if (i - max(warmup, 1)) % progress_step == 0:
                done = (i - max(warmup, 1))
                pct = round((done / max(total_iters, 1)) * 100.0, 1)
                _replay_log(runtime, f"Replay: progress {pct}% ({done}/{total_iters}) bal={round(balance,2)} open_positions={len(open_positions)}", kind='progress')

        # Close remaining at final close
        if open_positions:
            last_bar = data.iloc[-1]
            close_last = float(last_bar['Close']); pip = _pip_size_from_mt5(symbol)
            for pos in open_positions:
                pips = (close_last - pos['entry'])/pip
                if pos['action'] == 'SELL':
                    pips = -pips
                pnl = pips * float(account.get('pip_value_per_lot',10.0) or 10.0) * pos['lot']
                balance += pnl
                trades_detail.append({
                    'entry_time': pos['ts'].isoformat(),
                    'exit_time': last_bar.name.isoformat(),
                    'entry_price': pos['entry'],
                    'exit_price': close_last,
                    'size_lot': pos['lot'],
                    'action': pos['action'],
                    'pnl_currency': pnl,
                    'exit_reason': 'close_final',
                })
            equity_curve.append({'time': last_bar.name.isoformat(), 'equity': round(balance,2)})

        total_return_pct = ((balance - starting_balance)/starting_balance)*100.0
        trades = len(trades_detail)
        wins = sum(1 for t in trades_detail if t['pnl_currency'] > 0)
        win_rate_pct = (wins/trades)*100.0 if trades else 0.0
        # rudimentary max DD
        eq_vals = [pt['equity'] for pt in equity_curve]
        peaks = []
        max_dd = 0.0
        for v in eq_vals:
            peaks.append(max(v, peaks[-1] if peaks else v))
            dd = (peaks[-1]-v)/peaks[-1] if peaks[-1]>0 else 0.0
            max_dd = max(max_dd, dd*100.0)

        payload = {
            'strategy': strat_yaml['strategy']['class'],
            'symbol': symbol,
            'timeframe': timeframe,
            'param_source': 'yaml',
            'params': params,
            'metrics': {
                'starting_balance': starting_balance,
                'ending_balance': balance,
                'total_return_pct': total_return_pct,
                'sharpe': 0.0,
                'max_drawdown_pct': max_dd,
                'trades': trades,
                'win_rate_pct': win_rate_pct,
            },
            'equity_curve': equity_curve,
            'trades_detail': trades_detail,
        }

        out_path = args.replay_output or os.path.join(runtime.get('results_root','results'), to_snake(strat_yaml['strategy'].get('results_dir') or strat_yaml['strategy']['class']), 'replay', 'full_dataset_backtest_bot.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        _replay_log(runtime, f"Replay saved to {out_path}", kind='end')

def main():
    parser = argparse.ArgumentParser(description='Live trading bot (enhanced)')
    parser.add_argument('--config', default='configs/trading_bot_config.yaml')
    parser.add_argument('--mode', choices=['live','replay'], default='live', help='Run live against MT5 or replay over historical data without MT5')
    parser.add_argument('--once', action='store_true', help='Run one pass per enabled strategy then exit')
    parser.add_argument('--dry-run', action='store_true', help='Do not send orders, only show intended actions')
    parser.add_argument('--skip-start', action='store_true', help='Skip processing of the current last closed bar; wait for next close')
    # Test / single-trade flags
    parser.add_argument('--test-trade', action='store_true', help='Execute a single test trade using strategy best params then exit')
    parser.add_argument('--test-side', choices=['BUY','SELL'], help='Override strategy signal direction in test mode')
    parser.add_argument('--test-strategy', help='Name of strategy to use for test trade (defaults first enabled)')
    parser.add_argument('--test-sl', type=float, help='Override stop_loss_pips (raw pips, decimals allowed)')
    parser.add_argument('--test-tp', type=float, help='Override take_profit_pips (raw pips, decimals allowed)')
    parser.add_argument('--test-symbol', help='Override symbol for test trade')
    parser.add_argument('--test-dry-run', action='store_true', help='Compute test trade, do not send order')
    parser.add_argument('--test-auto-adjust', action='store_true', help='Automatically widen SL/TP to satisfy broker minimum stop distance')
    # Replay flags
    parser.add_argument('--replay-start', help='Start timestamp/date for replay (ISO, e.g., 2025-01-01)')
    parser.add_argument('--replay-end', help='End timestamp/date for replay (ISO)')
    parser.add_argument('--replay-output', help='Output JSON path for replay results (default under results/<strategy>/replay/)')
    parser.add_argument('--replay-model', choices=['parity','parity_bot','simple'], default='parity', help='Modeling for replay: parity uses evaluator backtest path; parity_bot uses bot loop with backtester rules; simple uses adapter tick simulation')
    args = parser.parse_args()

    load_dotenv()
    with open(args.config,'r') as f:
        bot_cfg = yaml.safe_load(f)

    # Align defaults with main_config-style structure if present; fallback to first strategy
    general_cfg = bot_cfg.get('general', {})
    default_symbol = general_cfg.get('default_symbol') or bot_cfg['strategies'][0].get('symbol','EURUSD')
    default_timeframe = general_cfg.get('default_timeframe') or bot_cfg['strategies'][0].get('timeframe','15m')

    general_defaults = {
        'general': {
            'default_symbol': default_symbol,
            'default_timeframe': default_timeframe,
        },
        'bot_results_root': bot_cfg.get('runtime',{}).get('results_root','results'),
        'magic_number': bot_cfg['account'].get('magic_number', 123456),
        'account_balance_placeholder': 100.0,
    }

    env_login = os.getenv('MT5_LOGIN'); env_password = os.getenv('MT5_PASSWORD'); env_server = os.getenv('MT5_SERVER')
    login = bot_cfg['account'].get('login') or (int(env_login) if env_login else None)
    password = bot_cfg['account'].get('password') or env_password
    server = bot_cfg['account'].get('server') or env_server

    log_path = Path(bot_cfg.get('runtime',{}).get('log_path','logs/trading_bot.log'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(bot_cfg.get('runtime',{}).get('journal_path','scripts/outputs/trade_journal.csv'))
    symbol_aliases: Dict[str,str] = bot_cfg.get('runtime',{}).get('symbol_aliases',{}) or {}

    def resolve_symbol(sym: str | None) -> str:
        base = sym or general_defaults['general']['default_symbol']
        return symbol_aliases.get(base, base)

    def log_line(msg: str) -> None:
        print(msg)
        try:
            with log_path.open('a',encoding='utf-8') as lf:
                lf.write(msg+'\n')
        except Exception:
            pass

    if args.mode == 'replay':
        _run_replay_mode(bot_cfg, args)
        return
    mt5a = MT5Adapter(login=login, password=password, server=server)
    if not mt5a.initialize():
        log_line('Failed to initialize MT5'); return

    try:
        # Canonical: use account keys (aligned with backtester). Risk overrides still supported.
        account_cfg = bot_cfg.get('account', {})
        risk_overrides = bot_cfg.get('risk', {})
        risk_cfg = {**account_cfg, **risk_overrides}
        strategies = [s for s in bot_cfg['strategies'] if s.get('enabled')]
        if not strategies:
            log_line('No enabled strategies.'); return

        # --- Single test trade mode ---
        if args.test_trade:
            # Resolve strategy selection
            chosen = None
            if args.test_strategy:
                for s in strategies:
                    if s['name'] == args.test_strategy:
                        chosen = s
                        break
            if chosen is None:
                chosen = strategies[0]
            strat_cfg_path = chosen['strategy_config']
            # Load strategy YAML
            try:
                with open(strat_cfg_path,'r') as f:
                    strat_yaml = yaml.safe_load(f)
            except Exception as e:
                log_line(f'Test trade: failed to load strategy YAML: {e}')
                return
            params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best') or strat_yaml.get('best_params_15m')
            if not isinstance(params, dict) or not params:
                log_line('Test trade: no best_params found in YAML.')
                return
            # Apply same scaling adjustment as run_strategy_once
            pips_scale = float(strat_yaml.get('pips_param_scale', 1.0) or 1.0)
            pips_keys = strat_yaml.get('pips_param_keys') or ['stop_loss_pips','take_profit_pips']
            if pips_scale != 1.0:
                params = params.copy()
                for k in pips_keys:
                    if k in params and isinstance(params[k], (int,float)):
                        v=float(params[k])
                        if v >= 1.0:
                            params[k]=v*pips_scale
            # Optional: round numeric params to align with evaluator behavior
            rp = bot_cfg.get('runtime',{}).get('round_params_decimals') if bot_cfg.get('runtime') else None
            if bot_cfg.get('runtime',{}).get('apply_param_rounding', False) and isinstance(rp,(int,float)):
                def _round_numeric_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
                    out: Dict[str, Any] = {}
                    for kk, vv in p.items():
                        if isinstance(vv, bool) or isinstance(vv, int):
                            out[kk] = vv
                        else:
                            try:
                                out[kk] = round(float(vv), int(decimals))
                            except Exception:
                                out[kk] = vv
                    return out
                params = _round_numeric_params(params, int(rp))
            # Overrides
            if args.test_sl is not None:
                params['stop_loss_pips']=float(args.test_sl)
            if args.test_tp is not None:
                params['take_profit_pips']=float(args.test_tp)
            if 'stop_loss_pips' not in params or 'take_profit_pips' not in params:
                log_line('Test trade: missing stop_loss_pips/take_profit_pips in params.')
                return
            side = args.test_side
            # If side not provided we attempt to instantiate strategy class and get its signal; fall back BUY
            symbol_tt = args.test_symbol or chosen.get('symbol') or bot_cfg['strategies'][0].get('symbol') or 'EURUSD'
            timeframe_tt = chosen.get('timeframe') or bot_cfg['strategies'][0].get('timeframe') or '15m'
            mt5a.ensure_symbol(symbol_tt)
            mt5 = _try_import_mt5()
            try:
                tick = mt5a.current_tick(symbol_tt)
            except Exception:
                log_line('Test trade: failed to fetch tick.')
                return
            bid=tick['bid']; ask=tick['ask']
            pip=_pip_size_from_mt5(symbol_tt)
            # --- Minimum stop distance / levels ---
            min_stop_distance_points = None
            freeze_level_points = None
            min_stop_price_dist = None
            try:
                info = mt5.symbol_info(symbol_tt)
                if info is not None:
                    # trade_stops_level is in POINTS (not pips). For 5-digit, point=0.00001.
                    min_stop_distance_points = getattr(info,'trade_stops_level',None)
                    freeze_level_points = getattr(info,'freeze_level',None)
                    point_val = getattr(info,'point',None)
                    if point_val and min_stop_distance_points and min_stop_distance_points>0:
                        min_stop_price_dist = min_stop_distance_points * point_val
            except Exception:
                pass
            # Decide action
            if side is None:
                # Try to construct the strategy to get a real signal
                try:
                    strat_info = strat_yaml['strategy']
                    strategy_class_name = strat_info['class']
                    module_path = strat_info.get('module', 'functions.strategies')
                    mod = __import__(module_path, fromlist=[strategy_class_name])
                    StrategyCls = getattr(mod, strategy_class_name)
                    warmup = infer_max_lookback(params)
                    df_sig = mt5a.fetch_recent_bars(symbol_tt, timeframe_tt, count=warmup + 5)
                    if len(df_sig) >= 2:
                        df_closed_sig = df_sig.iloc[:-1]
                        strat_inst = StrategyCls(df_closed_sig, params)
                        action_sig, sl_pips_sig, tp_pips_sig = strat_inst.generate_signals()
                        if action_sig in ('BUY','SELL') and sl_pips_sig and tp_pips_sig:
                            side = action_sig
                            # Use strategy-provided pips (after scaling logic) unless user overrides
                            if args.test_sl is None:
                                params['stop_loss_pips']=sl_pips_sig
                            if args.test_tp is None:
                                params['take_profit_pips']=tp_pips_sig
                except Exception as e:
                    log_line(f'Test trade: strategy signal resolution failed: {e}; defaulting to BUY')
                    side='BUY'
            if side is None:
                side='BUY'
            entry_price = ask if side=='BUY' else bid
            sl_pips = float(params['stop_loss_pips'])
            tp_pips = float(params['take_profit_pips'])
            if side=='BUY':
                sl_price = entry_price - sl_pips * pip
                tp_price = entry_price + tp_pips * pip
            else:
                sl_price = entry_price + sl_pips * pip
                tp_price = entry_price - tp_pips * pip
            # Validate minimum stop distance (only for SL; TP usually also must exceed distance on some brokers)
            def _adjust_level(level_price: float, is_sl: bool) -> float:
                # Round to symbol digits
                try:
                    info = mt5.symbol_info(symbol_tt) if mt5 is not None else None
                    digits = getattr(info,'digits',None)
                    if digits is not None:
                        return round(level_price, digits)
                except Exception:
                    pass
                return round(level_price,5)
            sl_price = _adjust_level(sl_price, True)
            tp_price = _adjust_level(tp_price, False)
            invalid_stop = False
            if min_stop_price_dist and min_stop_price_dist>0:
                if side=='BUY':
                    if entry_price - sl_price < min_stop_price_dist:
                        invalid_stop = True
                else:
                    if sl_price - entry_price < min_stop_price_dist:
                        invalid_stop = True
                # Basic TP distance check (optional)
                tp_invalid = False
                if side=='BUY':
                    if tp_price - entry_price < min_stop_price_dist:
                        tp_invalid = True
                else:
                    if entry_price - tp_price < min_stop_price_dist:
                        tp_invalid = True
                if invalid_stop or tp_invalid:
                    msg = f"[TEST] Broker min stop distance={min_stop_price_dist:.5f} (points={min_stop_distance_points}) violated: SL_delta={(abs(entry_price-sl_price)):.5f} TP_delta={(abs(tp_price-entry_price)):.5f}"
                    log_line(msg)
                    if args.test_auto_adjust:
                        # Widen SL/TP to minimum distance preserving direction
                        if side=='BUY':
                            sl_price = entry_price - min_stop_price_dist
                            if tp_invalid: tp_price = entry_price + max(min_stop_price_dist, abs(tp_price-entry_price))
                        else:
                            sl_price = entry_price + min_stop_price_dist
                            if tp_invalid: tp_price = entry_price - max(min_stop_price_dist, abs(tp_price-entry_price))
                        sl_price = _adjust_level(sl_price, True)
                        tp_price = _adjust_level(tp_price, False)
                        log_line(f"[TEST] Auto-adjusted SL={sl_price:.5f} TP={tp_price:.5f}")
                    else:
                        log_line("[TEST] Use --test-auto-adjust or increase stop_loss_pips / take_profit_pips.")
            # Lot sizing (reuse logic subset from run_strategy_once)
            ai = None
            try:
                if mt5 is not None:
                    ai = mt5.account_info()
            except Exception:
                ai = None
            balance = general_defaults['account_balance_placeholder']
            free_margin = None
            if ai is not None:
                balance = getattr(ai,'equity',None) or getattr(ai,'balance',balance)
                free_margin = getattr(ai,'margin_free',None)
            lev = risk_cfg.get('leverage',100)
            # Cap candidates
            cap_candidates=[]
            if free_margin is not None:
                cap_candidates.append(theoretical_max_lots_from_free_margin(free_margin, lev, entry_price))
            mcp = risk_cfg.get('max_concurrent_positions')
            maot = risk_cfg.get('max_allowed_open_trades') if not mcp else None
            per_trade_divisor=None
            if mcp and isinstance(mcp,(int,float)) and mcp>0:
                per_trade_divisor=float(mcp)
            elif maot and isinstance(maot,(int,float)) and maot>0:
                per_trade_divisor=float(maot)
            if per_trade_divisor:
                try:
                    per_trade_budget = balance / per_trade_divisor
                    effective_budget = min(per_trade_budget, free_margin) if free_margin is not None else per_trade_budget
                    cap_candidates.append(theoretical_max_lots_from_margin_budget(effective_budget, lev, entry_price))
                except Exception:
                    pass
            theoretical_cap = min(cap_candidates) if cap_candidates else theoretical_max_lots(balance, lev, entry_price)
            stop_dist = abs(entry_price - sl_price)
            risk_amount = balance * risk_cfg['risk_per_trade']
            raw_lot = risk_amount / (stop_dist * 100000) if stop_dist>0 else 0
            lot = min(raw_lot, theoretical_cap) if raw_lot>0 else 0
            min_size = float(risk_cfg.get('min_size', 0.01) or 0.01)
            lot_step = float(risk_cfg.get('lot_step', 0.01) or 0.01)
            max_lot_size = float(risk_cfg.get('max_lot_size', 0.0) or 0.0)
            if lot_step and lot_step>0:
                lot = (lot // lot_step)*lot_step
            else:
                lot = math.floor(lot*100)/100.0
            if max_lot_size and max_lot_size>0 and lot>max_lot_size:
                lot=max_lot_size
            deviation_points = risk_cfg.get('deviation_points')
            if deviation_points is None:
                deviation_points = max(int(risk_cfg.get('slippage_pips',0.1)*10),20)
            log_line(f"[TEST] entry={entry_price:.5f} pip_size={pip:.5f} sl_pips={sl_pips} tp_pips={tp_pips} -> SL={sl_price:.5f} TP={tp_price:.5f}")
            log_line(f"[TEST] raw_lot={raw_lot:.4f} capped={lot:.4f} balance={balance:.2f} risk_amt={risk_amount:.2f} stop_dist={stop_dist:.5f}")
            order_request = {
                'symbol': symbol_tt,
                'volume': lot,
                'order_type': side,
                'price': entry_price,
                'sl': sl_price,
                'tp': tp_price,
                'slippage_points': int(deviation_points),
                'magic': general_defaults['magic_number'],
                'comment': 'test_trade',
            }
            if lot < min_size:
                log_line(f"[TEST] Aborting: lot {lot:.4f} below min_size {min_size}")
                return
            if invalid_stop and not args.test_auto_adjust:
                log_line("[TEST] Aborting: invalid stop levels (broker constraints).")
                return
            if args.test_dry_run:
                log_line(f"[TEST] DRY-RUN order: {order_request}")
                return
            # Send order
            result = mt5a.send_market_order(**order_request)
            log_line(f"[TEST] send_market_order retcode={result.get('retcode')} order={result}")
            return

        # --- Warm-up simulation (concurrency gating only) ---
        warmup_days = float(bot_cfg.get('runtime', {}).get('warmup_days', 0) or 0)
        sim_positions_by_symbol: Dict[str, list[Dict[str, Any]]] = {}
        sim_last_bar_ts: Dict[str, pd.Timestamp] = {}
        if warmup_days > 0:
            log_line(f"Warm-up: simulating {warmup_days} days of history for concurrency gating...")
            apply_round = bool(bot_cfg.get('runtime', {}).get('apply_param_rounding', False))
            round_dec = int(bot_cfg.get('runtime', {}).get('round_params_decimals', 2))
            for strat in strategies:
                symbol = resolve_symbol(strat.get('symbol'))
                timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
                mt5a.ensure_symbol(symbol)
                try:
                    with open(strat['strategy_config'], 'r') as f:
                        strat_yaml = yaml.safe_load(f)
                except Exception as e:
                    log_line(f"Warm-up: failed to load strategy YAML for {strat['name']}: {e}")
                    continue

                params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best')
                if not isinstance(params, dict) or not params:
                    log_line(f"Warm-up: no best_params for {strat['name']}; skipping.")
                    continue

                pips_scale = float(strat_yaml.get('pips_param_scale', 1.0) or 1.0)
                pips_keys = strat_yaml.get('pips_param_keys') or ['stop_loss_pips', 'take_profit_pips']
                if pips_scale != 1.0:
                    params = params.copy()
                    for k in pips_keys:
                        if k in params and isinstance(params[k], (int, float)):
                            v = float(params[k])
                            if v >= 1.0:
                                params[k] = v * pips_scale
                if apply_round and isinstance(round_dec, (int, float)):
                    def _round_numeric_params(p: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
                        out: Dict[str, Any] = {}
                        for kk, vv in p.items():
                            if isinstance(vv, bool) or isinstance(vv, int):
                                out[kk] = vv
                            else:
                                try:
                                    out[kk] = round(float(vv), int(decimals))
                                except Exception:
                                    out[kk] = vv
                        return out
                    params = _round_numeric_params(params, int(round_dec))

                try:
                    mod_path = strat_yaml['strategy'].get('module', 'functions.strategies')
                    cls_name = strat_yaml['strategy']['class']
                    mod = __import__(mod_path, fromlist=[cls_name])
                    StrategyCls = getattr(mod, cls_name)
                except Exception as e:
                    log_line(f"Warm-up: failed to load strategy class for {strat['name']}: {e}")
                    continue

                mins = timeframe_to_minutes(timeframe)
                bars_needed = int(math.ceil(warmup_days * 1440 / mins))
                warmup_extra = infer_max_lookback(params) + 5
                count = max(200, bars_needed + warmup_extra)
                try:
                    df = mt5a.fetch_recent_bars(symbol, timeframe, count=count)
                except Exception as e:
                    log_line(f"Warm-up: failed to fetch bars for {symbol} {timeframe}: {e}")
                    continue

                start_ts = _server_time_utc(symbol) - timedelta(days=warmup_days)
                df = df.loc[start_ts:]
                if len(df) < infer_max_lookback(params) + 2:
                    log_line(f"Warm-up: insufficient bars for {strat['name']} after filtering; skipping.")
                    continue

                sim_res = _simulate_warmup_open_positions(
                    data=df,
                    symbol=symbol,
                    strategy_cls=StrategyCls,
                    params=params,
                    account_cfg=risk_cfg,
                )
                if sim_res.get('open_positions'):
                    sim_positions_by_symbol.setdefault(symbol, []).extend(sim_res['open_positions'])
                if sim_res.get('last_closed_bar_ts') is not None:
                    sim_last_bar_ts[f"{symbol}|{timeframe}"] = pd.to_datetime(sim_res['last_closed_bar_ts'])

            for sym, positions in sim_positions_by_symbol.items():
                log_line(f"Warm-up: {sym} simulated_open={len(positions)}")

        # Header
        mt5 = _try_import_mt5()
        ai = None
        try:
            if mt5 is not None:
                ai = mt5.account_info()
        except Exception:
            ai = None
        bal = getattr(ai,'balance',None) if ai else None
        eq = getattr(ai,'equity',None) if ai else None
        fm = getattr(ai,'margin_free',None) if ai else None
        bal_str = f"{bal:.2f}" if bal is not None else "?"
        eq_str = f"{eq:.2f}" if eq is not None else "?"
        fm_str = f"{fm:.2f}" if fm is not None else "?"
        header = [
            '\n==================== Trading Bot Session ====================',
            f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"Account: balance={bal_str} equity={eq_str} free_margin={fm_str}",
            (
                f"Risk: risk_per_trade={risk_cfg.get('risk_per_trade')} leverage={risk_cfg.get('leverage')} "
                f"deviation_points={risk_cfg.get('deviation_points','auto')} max_concurrent_positions={risk_cfg.get('max_concurrent_positions','none')} "
                f"max_allowed_open_trades={risk_cfg.get('max_allowed_open_trades','none')}"
            ),
            f"Runtime: process_on_start={bot_cfg['runtime'].get('process_on_start', True)} poll={bot_cfg['runtime'].get('poll_interval_seconds',5)}s",
            f"Paths: log={log_path} journal={journal_path}",
            '------------------------------------------------------------'
        ]
        for strat in strategies:
            symbol = resolve_symbol(strat.get('symbol'))
            timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
            now_srv = _server_time_utc(symbol)
            # compute next close
            def timeframe_to_minutes(tf:str)->int:
                tf=tf.lower().strip()
                if tf.endswith('m'): return int(tf[:-1])
                if tf.endswith('h'): return int(tf[:-1])*60
                if tf.endswith('d'): return int(tf[:-1])*1440
                raise ValueError(f'Unsupported timeframe {tf}')
            mins = timeframe_to_minutes(timeframe)
            day_start = now_srv.replace(hour=0,minute=0,second=0,microsecond=0)
            mins_since = (now_srv - day_start).total_seconds()//60
            intervals = int(mins_since//mins)+1
            next_close = day_start + timedelta(minutes=intervals*mins)
            try:
                mt5_local = mt5 if 'mt5' in locals() else _try_import_mt5()
                sp_str = '?'
                if mt5_local is not None:
                    t = mt5_local.symbol_info_tick(symbol)
                    sp = (t.ask - t.bid) if t else None
                    sp_str = f"{sp:.5f}" if sp is not None else '?'
            except Exception:
                sp_str='?'
            header.append(f"Strategy: {strat['name']} | Symbol: {symbol} | TF: {timeframe} | Next close: {next_close.strftime('%Y-%m-%d %H:%M:%S')} UTC | Spread: {sp_str}")
        header.append('============================================================\n')
        for line in header:
            log_line(line)

        # State tracking
        last_bar_seen: Dict[str,pd.Timestamp] = {}
        last_bar_boundary_seen: Dict[str,pd.Timestamp] = {}
        # Default: do NOT process current last closed bar on startup; wait for next boundary.
        process_on_start = bool(bot_cfg.get('runtime',{}).get('process_on_start', False))
        if args.skip_start: process_on_start = False

        # Prime last seen bar & boundary unless explicit immediate processing requested.
        if not process_on_start:
            for strat in strategies:
                symbol = resolve_symbol(strat.get('symbol'))
                timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
                key = f"{strat['name']}|{symbol}|{timeframe}"
                try:
                    mt5a.ensure_symbol(symbol)
                    df_seed = mt5a.fetch_recent_bars(symbol, timeframe, count=2)
                    if len(df_seed) >= 2:
                        closed_ts = df_seed.iloc[:-1].index[-1]
                        last_bar_seen[key]=closed_ts
                    now_seed = _server_time_utc(symbol)
                    def tf_to_min(tf:str)->int:
                        tf=tf.lower().strip()
                        if tf.endswith('m'): return int(tf[:-1])
                        if tf.endswith('h'): return int(tf[:-1])*60
                        if tf.endswith('d'): return int(tf[:-1])*1440
                        return 15
                    mins=tf_to_min(timeframe)
                    day_start= now_seed.replace(hour=0,minute=0,second=0,microsecond=0)
                    mins_since=int((now_seed-day_start).total_seconds()//60)
                    current_boundary=(mins_since//mins)*mins
                    boundary_start= day_start + timedelta(minutes=current_boundary)
                    last_bar_boundary_seen[key]=pd.Timestamp(boundary_start)
                except Exception as e:
                    log_line(f"Startup priming failed for {strat['name']}: {e}")

        align_cfg = bot_cfg['runtime'].get('candle_alignment',{}) if isinstance(bot_cfg.get('runtime'),dict) else {}
        align_enabled = bool(align_cfg.get('enabled', False))
        pre_close_seconds = int(align_cfg.get('pre_close_seconds',120))
        post_close_seconds = int(align_cfg.get('post_close_seconds',60))
        dense_poll_seconds = int(align_cfg.get('dense_poll_seconds', bot_cfg['runtime'].get('poll_interval_seconds',5)))
        idle_min_sleep = int(align_cfg.get('idle_min_sleep',30))

        def timeframe_to_minutes(tf:str)->int:
            tf=tf.lower().strip()
            if tf.endswith('m'): return int(tf[:-1])
            if tf.endswith('h'): return int(tf[:-1])*60
            if tf.endswith('d'): return int(tf[:-1])*1440
            raise ValueError(f'Unsupported timeframe {tf}')

        def next_candle_close(ts: datetime, tf: str) -> datetime:
            minutes = timeframe_to_minutes(tf)
            day_start = ts.replace(hour=0,minute=0,second=0,microsecond=0)
            mins_since = (ts - day_start).total_seconds()//60
            intervals = int(mins_since//minutes)+1
            return day_start + timedelta(minutes=intervals*minutes)

        # balance/history CSV
        balance_journal_path = Path(bot_cfg.get('runtime',{}).get('balance_journal_path','scripts/outputs/balance_history.csv'))
        balance_journal_path.parent.mkdir(parents=True, exist_ok=True)

        # Only write balance/equity to CSV once per minute to reduce data volume.
        last_balance_poll: datetime | None = None

        while True:
            try:
                now_bal = datetime.now(timezone.utc)
                if last_balance_poll is None or (now_bal - last_balance_poll).total_seconds() >= 60:
                    mt5_local = mt5 if 'mt5' in locals() else _try_import_mt5()
                    ai_loop = None
                    try:
                        if mt5_local is not None:
                            ai_loop = mt5_local.account_info()
                    except Exception:
                        ai_loop = None
                    if ai_loop is not None:
                        _append_balance_journal(balance_journal_path, {
                            'ts_utc': now_bal.strftime('%Y-%m-%d %H:%M:%S'),
                            'balance': getattr(ai_loop,'balance',None),
                            'equity': getattr(ai_loop,'equity',None),
                            'margin': getattr(ai_loop,'margin',None),
                            'margin_free': getattr(ai_loop,'margin_free',None),
                            'leverage': risk_cfg.get('leverage'),
                            'profit': getattr(ai_loop,'profit',None),
                        })
                        last_balance_poll = now_bal
            except Exception:
                pass

            for strat in strategies:
                symbol_res = resolve_symbol(strat.get('symbol'))
                timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
                key = f"{strat['name']}|{symbol_res}|{timeframe}"
                try:
                    now_loop = _server_time_utc(symbol_res)
                    mins = timeframe_to_minutes(timeframe)
                    day_start = now_loop.replace(hour=0,minute=0,second=0,microsecond=0)
                    mins_since = int((now_loop-day_start).total_seconds()//60)
                    current_boundary = (mins_since//mins)*mins
                    boundary_start = day_start + timedelta(minutes=current_boundary)
                    if (now_loop - boundary_start).total_seconds() < 1:  # wait for close
                        continue
                    if last_bar_boundary_seen.get(key) == pd.Timestamp(boundary_start):
                        continue
                    closed_ts = None
                    simulated_open = 0
                    if warmup_days > 0:
                        try:
                            df_last = mt5a.fetch_recent_bars(symbol_res, timeframe, count=2)
                            if len(df_last) >= 2:
                                closed_bar = df_last.iloc[-2]
                                closed_ts = pd.to_datetime(closed_bar.name)
                                sim_key = f"{symbol_res}|{timeframe}"
                                if sim_last_bar_ts.get(sim_key) != closed_ts:
                                    spread_price = pips_to_price(risk_cfg.get('spread_pips', 0.0) or 0.0, symbol_res)
                                    slippage_price = pips_to_price(risk_cfg.get('slippage_pips', 0.0) or 0.0, symbol_res)
                                    sim_positions_by_symbol[symbol_res] = _update_simulated_positions_for_bar(
                                        sim_positions_by_symbol.get(symbol_res, []),
                                        bar_high=float(closed_bar['High']),
                                        bar_low=float(closed_bar['Low']),
                                        spread_price=spread_price,
                                        slippage_price=slippage_price,
                                    )
                                    sim_last_bar_ts[sim_key] = closed_ts
                        except Exception:
                            pass
                        simulated_open = len(sim_positions_by_symbol.get(symbol_res, []))

                    max_positions_cfg = risk_cfg.get('max_concurrent_positions')
                    max_positions_eff = max_positions_cfg
                    if max_positions_cfg is not None and warmup_days > 0:
                        try:
                            max_positions_eff = int(max_positions_cfg) - int(simulated_open)
                        except Exception:
                            max_positions_eff = max_positions_cfg

                    if warmup_days > 0 and max_positions_eff is not None and isinstance(max_positions_eff, (int, float)) and max_positions_eff <= 0:
                        res = {
                            'status': 'warmup_position_limit',
                            'simulated_open': simulated_open,
                            'max_positions': max_positions_cfg,
                            'closed_bar_ts': closed_ts,
                        }
                        if closed_ts is not None:
                            last_bar_seen[key] = closed_ts
                        last_bar_boundary_seen[key] = pd.Timestamp(boundary_start)
                    else:
                        res = run_strategy_once(
                            strat['strategy_config'],
                            {'symbol':symbol_res, **strat},
                            general_defaults,
                            mt5a,
                            risk_cfg,
                            dry_run=args.dry_run,
                            max_positions=max_positions_eff,
                            last_closed_bar_ts=last_bar_seen.get(key),
                            apply_param_rounding=bool(bot_cfg.get('runtime',{}).get('apply_param_rounding', False)),
                            round_params_decimals=int(bot_cfg.get('runtime',{}).get('round_params_decimals', 2)),
                        )
                    if 'closed_bar_ts' in res and res['status'] not in ('no_data',):
                        last_bar_seen[key] = res['closed_bar_ts']
                        last_bar_boundary_seen[key] = pd.Timestamp(boundary_start)
                    if res['status'] == 'no_new_bar':
                        continue
                    parts = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {res['status']}"]
                    if res.get('action') in ('BUY','SELL') or res['status'] in ('sent','dry_run','order_error','lot_below_min','insufficient_margin','warmup_position_limit'):
                        if res.get('action'): parts.append(str(res.get('action')))
                        if res.get('lot') is not None: parts.append(f"lot={res.get('lot')}")
                        if res.get('entry') is not None:
                            try: parts.append(f"entry={float(res.get('entry')):.5f}")
                            except Exception: parts.append(f"entry={res.get('entry')}")
                        if res.get('sl') is not None:
                            try: parts.append(f"sl={float(res.get('sl')):.5f}")
                            except Exception: parts.append(f"sl={res.get('sl')}")
                        if res.get('tp') is not None:
                            try: parts.append(f"tp={float(res.get('tp')):.5f}")
                            except Exception: parts.append(f"tp={res.get('tp')}")
                        if res.get('spread') is not None:
                            try: parts.append(f"spread={float(res.get('spread')):.5f}")
                            except Exception: parts.append(f"spread={res.get('spread')}")
                        if res.get('balance') is not None:
                            try: parts.append(f"bal={float(res.get('balance')):.2f}")
                            except Exception: parts.append(f"bal={res.get('balance')}")
                        if res['status']=='insufficient_margin':
                            if res.get('free_margin') is not None:
                                try: parts.append(f"fm={float(res.get('free_margin')):.2f}")
                                except Exception: parts.append(f"fm={res.get('free_margin')}")
                            if res.get('margin_required') is not None:
                                try: parts.append(f"mreq={float(res.get('margin_required')):.2f}")
                                except Exception: parts.append(f"mreq={res.get('margin_required')}")
                        if res['status']=='lot_below_min':
                            if res.get('leverage') is not None: parts.append(f"lev={res.get('leverage')}")
                        if res['status']=='warmup_position_limit':
                            if res.get('simulated_open') is not None: parts.append(f"sim_open={res.get('simulated_open')}")
                            if res.get('max_positions') is not None: parts.append(f"max_pos={res.get('max_positions')}")
                    if 'closed_bar_ts' in res and res['closed_bar_ts'] is not None:
                        try:
                            bar_ts = pd.to_datetime(res['closed_bar_ts'])
                            parts.append(f"bar={bar_ts.strftime('%Y-%m-%d %H:%M:%S')}Z")
                        except Exception:
                            pass
                    prefix = ''
                    if res.get('status')=='sent': prefix='✅ '
                    elif res.get('status')=='order_error': prefix='❌ '
                    log_line(prefix + ' '.join(parts))

                    # Journal
                    try:
                        order = res.get('order') if isinstance(res.get('order'),dict) else {}
                        _append_trade_journal(journal_path, {
                            'ts_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'strategy': strat['name'],
                            'symbol': symbol_res,
                            'timeframe': timeframe,
                            'status': res.get('status'),
                            'action': res.get('action'),
                            'lot': res.get('lot'),
                            'entry': res.get('entry'),
                            'sl': res.get('sl'),
                            'tp': res.get('tp'),
                            'leverage': res.get('leverage'),
                            'balance': res.get('balance'),
                            'free_margin': res.get('free_margin'),
                            'margin_required': res.get('margin_required'),
                            'spread': res.get('spread'),
                            'ticket_order': order.get('order'),
                            'ticket_deal': order.get('deal'),
                        })
                    except Exception:
                        pass
                except Exception as e:
                    log_line(f"Error running strategy {strat['name']}: {e}")
            if args.once:
                break
            if align_enabled:
                tf = strategies[0].get('timeframe') or general_defaults['general']['default_timeframe']
                sym0 = resolve_symbol(strategies[0].get('symbol'))
                now = _server_time_utc(sym0)
                nxt = next_candle_close(now, tf)
                sec_to_close = (nxt - now).total_seconds()
                if sec_to_close <= pre_close_seconds and sec_to_close >= -post_close_seconds:
                    sleep_for = max(1, dense_poll_seconds)
                else:
                    if sec_to_close > pre_close_seconds:
                        sleep_for = max(idle_min_sleep, int(sec_to_close - pre_close_seconds))
                    else:
                        following = next_candle_close(nxt, tf)
                        sec_following = (following - now).total_seconds()
                        sleep_for = max(idle_min_sleep, int(sec_following - pre_close_seconds))
                time.sleep(sleep_for)
            else:
                time.sleep(bot_cfg['runtime'].get('poll_interval_seconds',5))
    finally:
        mt5a.shutdown()


if __name__ == '__main__':
    main()

        
