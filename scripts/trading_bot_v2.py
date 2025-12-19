#!/usr/bin/env python3
"""
Trading bot v2 (aligned):
- Uses same params and backtest-aligned strategy interface.
- Candle-close evaluation; no implicit scaling of SL/TP (strategy defines them).
- Risk-based lot sizing, broker margin checks, concise logs and CSV journals.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from dotenv import load_dotenv

# Project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.mt5_adapter import MT5Adapter
from scripts.utils.data_utils import pip_size
from scripts.utils.strategy_utils_v2 import infer_max_lookback


def to_snake(name: str) -> str:
    import re
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.replace('__', '_').lower()


def _pip_size_from_mt5(symbol: str) -> float:
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is not None:
            digits = getattr(info, 'digits', None)
            point = getattr(info, 'point', None)
            if point and digits is not None:
                if digits in (3, 5):
                    return float(point) * 10.0
                return float(point)
    except Exception:
        pass
    return pip_size(symbol)


def _server_time_utc(symbol: str) -> datetime:
    try:
        import MetaTrader5 as mt5
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
):
    import MetaTrader5 as mt5
    with open(strategy_cfg_path, 'r') as f:
        strat_yaml = yaml.safe_load(f)
    strat_info = strat_yaml['strategy']
    strategy_class_name = strat_info['class']
    module_path = strat_info.get('module', 'functions.strategies')

    symbol = strategy_section.get('symbol') or global_cfg['general']['default_symbol']
    timeframe = strategy_section.get('timeframe') or global_cfg['general']['default_timeframe']
    mt5a.ensure_symbol(symbol)

    # Resolve parameters source: YAML or results artifact
    params_source = strategy_section.get('params_source', 'yaml')
    params = None
    if params_source == 'results':
        results_root = Path(global_cfg.get('bot_results_root','results'))
        # Prefer explicit path if provided
        explicit = strategy_section.get('params_results_path')
        if explicit:
            p = Path(explicit)
            if p.exists():
                try:
                    blob = yaml.safe_load(p.read_text()) if p.suffix in ('.yaml','.yml') else __import__('json').loads(p.read_text())
                    params = blob.get('params') if isinstance(blob, dict) else None
                except Exception:
                    params = None
        if params is None:
            # Auto-discover under results_root/<strategy_folder>/optimizations
            strat_folder = to_snake(strat_info.get('class', 'strategy'))
            opt_dir = results_root / strat_folder / 'optimizations'
            for cand in [
                opt_dir / 'bayesian_wf_v2_best.json',
                opt_dir / 'bayesian_wf_optimization_results_best.json',
                opt_dir / 'bayesian_optimization_results_best.json',
                opt_dir / 'optimization_results_best.json',
            ]:
                if cand.exists():
                    try:
                        blob = __import__('json').loads(cand.read_text())
                        pp = blob.get('params')
                        if isinstance(pp, dict) and pp:
                            params = pp
                            break
                    except Exception:
                        continue
    if params is None:
        params = strat_yaml.get('best_params') or strat_yaml.get('parameters_best')
    if not isinstance(params, dict) or not params:
        return {'status': 'no_params'}

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
    if action not in ('BUY','SELL') or sl_pips is None or tp_pips is None:
        return {'status': 'no_signal', 'closed_bar_ts': closed_ts}

    tick = mt5a.current_tick(symbol)
    bid = tick['bid']; ask = tick['ask']
    spread = ask - bid
    entry_price = ask if action=='BUY' else bid
    pip = _pip_size_from_mt5(symbol)
    if action=='BUY':
        sl_price = entry_price - float(sl_pips) * pip
        tp_price = entry_price + float(tp_pips) * pip
    else:
        sl_price = entry_price + float(sl_pips) * pip
        tp_price = entry_price - float(tp_pips) * pip

    ai = mt5.account_info()
    balance = global_cfg['account_balance_placeholder']
    free_margin = None
    if ai is not None:
        balance = getattr(ai,'equity',None) or getattr(ai,'balance',balance)
        free_margin = getattr(ai,'margin_free',None)

    # Position limit per symbol
    if max_positions is not None and max_positions>0:
        try:
            open_pos = mt5.positions_get(symbol=symbol)
            if open_pos is not None and len(open_pos) >= max_positions:
                return {'status':'position_limit','closed_bar_ts':closed_ts}
        except Exception:
            pass

    lev = risk_cfg.get('leverage',100)
    # Cap by theoretical from balance/leverage and free margin
    def theoretical_max_lots(balance: float, leverage: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return balance * leverage / (100000.0 * price)

    def theoretical_max_lots_from_free_margin(free_margin: float, leverage: float, price: float) -> float:
        if price <= 0 or free_margin is None:
            return 0.0
        return float(free_margin) * leverage / (100000.0 * price)

    cap_candidates = []
    if free_margin is not None:
        cap_candidates.append(theoretical_max_lots_from_free_margin(free_margin, lev, entry_price))
    mcp = risk_cfg.get('max_concurrent_positions')
    per_trade_divisor = float(mcp) if mcp else None
    if per_trade_divisor:
        try:
            per_trade_budget = balance / per_trade_divisor
            eff = min(per_trade_budget, free_margin) if free_margin is not None else per_trade_budget
            cap_candidates.append(theoretical_max_lots_from_free_margin(eff, lev, entry_price))
        except Exception:
            pass
    theo_cap = min(cap_candidates) if cap_candidates else theoretical_max_lots(balance, lev, entry_price)

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {'status':'invalid_stop','closed_bar_ts':closed_ts}
    risk_amount = balance * float(risk_cfg.get('risk_per_trade', 0.01))
    raw_lot = risk_amount / (stop_dist * 100000)
    lot = min(raw_lot, theo_cap) if raw_lot > 0 else 0.0

    lot_step = float(risk_cfg.get('lot_step', 0.01) or 0.01)
    min_size = float(risk_cfg.get('min_size', 0.01) or 0.01)
    max_lot_size = float(risk_cfg.get('max_lot_size', 0.0) or 0.0)

    if lot_step and lot_step > 0:
        lot = (lot // lot_step) * lot_step
    else:
        lot = math.floor(lot * 100) / 100.0
    if max_lot_size and max_lot_size > 0 and lot > max_lot_size:
        lot = max_lot_size
    if lot < min_size:
        return {'status':'lot_below_min','closed_bar_ts':closed_ts,'lot':lot,'entry':entry_price,'sl':sl_price,'tp':tp_price,'balance':balance,'leverage':lev,'spread':spread,'action':action}

    # Margin requirement check
    margin_required = None
    try:
        import MetaTrader5 as mt5
        mt5_type = mt5.ORDER_TYPE_BUY if action=='BUY' else mt5.ORDER_TYPE_SELL
        margin_required = mt5.order_calc_margin(mt5_type, symbol, lot, entry_price)
    except Exception:
        pass
    if free_margin is not None and margin_required is not None and margin_required > free_margin:
        return {'status':'insufficient_margin','entry':entry_price,'sl':sl_price,'tp':tp_price,'lot':lot,'free_margin':free_margin,'margin_required':margin_required,'balance':balance,'leverage':lev,'spread':spread,'closed_bar_ts':closed_ts,'action':action}

    deviation_points = risk_cfg.get('deviation_points')
    if deviation_points is None:
        deviation_points = max(int(float(risk_cfg.get('slippage_pips',0.1))*10), 20)

    order_request = {
        'symbol': symbol,
        'volume': lot,
        'order_type': action,
        'price': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'slippage_points': int(deviation_points),
        'magic': global_cfg['magic_number'],
        'comment': 'bot_v2',
    }

    if dry_run:
        return {'status':'dry_run','order':order_request,'lot':lot,'action':action,'entry':entry_price,'sl':sl_price,'tp':tp_price,'spread':spread,'closed_bar_ts':closed_ts,'balance':balance,'free_margin':free_margin,'margin_required':margin_required}

    result = mt5a.send_market_order(**order_request)
    retcode = result.get('retcode') if isinstance(result, dict) else None
    status = 'sent' if retcode else 'order_error'
    return {'status':status,'order':result,'lot':lot,'action':action,'entry':entry_price,'sl':sl_price,'tp':tp_price,'spread':spread,'closed_bar_ts':closed_ts,'balance':balance,'free_margin':free_margin,'margin_required':margin_required}


def main():
    ap = argparse.ArgumentParser(description='Trading bot v2 (aligned)')
    ap.add_argument('--config', default='configs/trading_bot_config.v2.yaml')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-start', action='store_true')
    args = ap.parse_args()

    load_dotenv()
    with open(args.config,'r') as f:
        bot_cfg = yaml.safe_load(f)

    # Read main config v2
    main_cfg_path = bot_cfg['general']['main_config_path']
    with open(main_cfg_path,'r') as f:
        main_conf = yaml.safe_load(f)

    general_defaults = {
        'general': {
            'default_symbol': main_conf['general']['default_symbol'],
            'default_timeframe': main_conf['general']['default_timeframe'],
        },
        'bot_results_root': bot_cfg['runtime']['results_root'],
        'magic_number': bot_cfg['account']['magic_number'],
        'account_balance_placeholder': 100.0,
    }

    env_login = os.getenv('MT5_LOGIN'); env_password = os.getenv('MT5_PASSWORD'); env_server = os.getenv('MT5_SERVER')
    login = bot_cfg['account'].get('login') or (int(env_login) if env_login else None)
    password = bot_cfg['account'].get('password') or env_password
    server = bot_cfg['account'].get('server') or env_server

    log_path = Path(bot_cfg.get('runtime',{}).get('log_path','logs/trading_bot_v2.log'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(bot_cfg.get('runtime',{}).get('journal_path','scripts/outputs/trade_journal_v2.csv'))

    def log_line(msg: str) -> None:
        print(msg)
        try:
            with log_path.open('a',encoding='utf-8') as lf:
                lf.write(msg+'\n')
        except Exception:
            pass

    # Init MT5
    mt5a = MT5Adapter(login=login, password=password, server=server)
    if not mt5a.initialize():
        log_line('Failed to initialize MT5'); return

    try:
        risk_cfg = bot_cfg['risk']
        strategies = [s for s in bot_cfg['strategies'] if s.get('enabled')]
        if not strategies:
            log_line('No enabled strategies.'); return

        # Session header
        import MetaTrader5 as mt5
        ai = mt5.account_info()
        bal = getattr(ai,'balance',None) if ai else None
        eq = getattr(ai,'equity',None) if ai else None
        fm = getattr(ai,'margin_free',None) if ai else None
        log_line(f"\n===== Trading Bot v2 Session =====")
        log_line(f"Account: balance={bal if bal is not None else '?'} equity={eq if eq is not None else '?'} free_margin={fm if fm is not None else '?'}")
        for strat in strategies:
            symbol = strat.get('symbol') or general_defaults['general']['default_symbol']
            timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
            now_srv = _server_time_utc(symbol)
            def tf_to_min(tf:str)->int:
                t=tf.lower().strip()
                if t.endswith('m'): return int(t[:-1])
                if t.endswith('h'): return int(t[:-1])*60
                if t.endswith('d'): return int(t[:-1])*1440
                return 15
            mins = tf_to_min(timeframe)
            day_start = now_srv.replace(hour=0,minute=0,second=0,microsecond=0)
            mins_since = (now_srv - day_start).total_seconds()//60
            intervals = int(mins_since//mins)+1
            next_close = day_start + timedelta(minutes=intervals*mins)
            try:
                t = mt5.symbol_info_tick(symbol)
                sp = (t.ask - t.bid) if t else None
                sp_str = f"{sp:.5f}" if sp is not None else '?'
            except Exception:
                sp_str='?'
            log_line(f"Strategy: {strat['name']} | Symbol: {symbol} | TF: {timeframe} | Next close: {next_close.strftime('%Y-%m-%d %H:%M:%S')} UTC | Spread: {sp_str}")

        # State
        last_bar_seen: Dict[str,pd.Timestamp] = {}
        last_bar_boundary_seen: Dict[str,pd.Timestamp] = {}
        process_on_start = bool(bot_cfg.get('runtime',{}).get('process_on_start', False))
        if args.skip_start: process_on_start = False
        if not process_on_start:
            for strat in strategies:
                symbol = strat.get('symbol') or general_defaults['general']['default_symbol']
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
                        t=tf.lower().strip()
                        if t.endswith('m'): return int(t[:-1])
                        if t.endswith('h'): return int(t[:-1])*60
                        if t.endswith('d'): return int(t[:-1])*1440
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
            t=tf.lower().strip()
            if t.endswith('m'): return int(t[:-1])
            if t.endswith('h'): return int(t[:-1])*60
            if t.endswith('d'): return int(t[:-1])*1440
            raise ValueError(f'Unsupported timeframe {tf}')

        def next_candle_close(ts: datetime, tf: str) -> datetime:
            minutes = timeframe_to_minutes(tf)
            day_start = ts.replace(hour=0,minute=0,second=0,microsecond=0)
            mins_since = (ts - day_start).total_seconds()//60
            intervals = int(mins_since//minutes)+1
            return day_start + timedelta(minutes=intervals*minutes)

        balance_journal_path = Path(bot_cfg.get('runtime',{}).get('balance_journal_path','scripts/outputs/balance_history_v2.csv'))
        balance_journal_path.parent.mkdir(parents=True, exist_ok=True)
        last_balance_poll: datetime | None = None

        while True:
            try:
                now_bal = datetime.now(timezone.utc)
                if last_balance_poll is None or (now_bal - last_balance_poll).total_seconds() >= 60:
                    ai_loop = mt5.account_info()
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
                symbol = strat.get('symbol') or general_defaults['general']['default_symbol']
                timeframe = strat.get('timeframe') or general_defaults['general']['default_timeframe']
                key = f"{strat['name']}|{symbol}|{timeframe}"
                try:
                    now_loop = _server_time_utc(symbol)
                    mins = timeframe_to_minutes(timeframe)
                    day_start = now_loop.replace(hour=0,minute=0,second=0,microsecond=0)
                    mins_since = int((now_loop-day_start).total_seconds()//60)
                    current_boundary = (mins_since//mins)*mins
                    boundary_start = day_start + timedelta(minutes=current_boundary)
                    if (now_loop - boundary_start).total_seconds() < 1:
                        continue
                    if last_bar_boundary_seen.get(key) == pd.Timestamp(boundary_start):
                        continue
                    res = run_strategy_once(
                        strat['strategy_config'],
                        {'symbol':symbol, **strat},
                        general_defaults,
                        mt5a,
                        risk_cfg,
                        dry_run=args.dry_run,
                        max_positions=risk_cfg.get('max_concurrent_positions'),
                        last_closed_bar_ts=last_bar_seen.get(key)
                    )
                    if 'closed_bar_ts' in res and res['status'] not in ('no_data',):
                        last_bar_seen[key] = res['closed_bar_ts']
                        last_bar_boundary_seen[key] = pd.Timestamp(boundary_start)
                    if res['status'] == 'no_new_bar':
                        continue
                    parts = [f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | {res['status']}"]
                    if res.get('action') in ('BUY','SELL') or res['status'] in ('sent','dry_run','order_error','lot_below_min','insufficient_margin'):
                        if res.get('action'):
                            parts.append(str(res.get('action')))
                        if res.get('lot') is not None:
                            parts.append(f"lot={res.get('lot')}")
                        for key_name, decimals, label in [
                            ('entry', 5, 'entry'), ('sl', 5, 'sl'), ('tp', 5, 'tp'), ('spread', 5, 'spread'),
                        ]:
                            if res.get(key_name) is not None:
                                try:
                                    val = float(res.get(key_name))
                                    parts.append(f"{label}={val:.{decimals}f}")
                                except Exception:
                                    parts.append(f"{label}={res.get(key_name)}")
                    if 'closed_bar_ts' in res and res['closed_bar_ts'] is not None:
                        try:
                            bar_ts = pd.to_datetime(res['closed_bar_ts'])
                            parts.append(f"bar={bar_ts.strftime('%Y-%m-%d %H:%M:%S')}Z")
                        except Exception:
                            pass
                    prefix = '✅ ' if res.get('status')=='sent' else ('❌ ' if res.get('status')=='order_error' else '')
                    log_line(prefix + ' '.join(parts))

                    # Journal row
                    try:
                        order = res.get('order') if isinstance(res.get('order'),dict) else {}
                        _append_trade_journal(journal_path, {
                            'ts_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'strategy': strat['name'],
                            'symbol': symbol,
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
                sym0 = strategies[0].get('symbol') or general_defaults['general']['default_symbol']
                now = _server_time_utc(sym0)
                nxt = next_candle_close(now, tf)
                sec_to_close = (nxt - now).total_seconds()
                if -post_close_seconds <= sec_to_close <= pre_close_seconds:
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
