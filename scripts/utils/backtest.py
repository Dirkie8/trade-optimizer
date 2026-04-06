from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from functions.base_strategy import BaseStrategy
from scripts.utils.data_utils import pips_to_price, pip_size
# Robust tqdm import: prefer auto, fall back to base; else disable gracefully
try:  # pragma: no cover - environment-dependent
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:  # Fallback if auto not available
        from tqdm import tqdm as _tqdm  # type: ignore
    tqdm = _tqdm
except Exception:  # Final fallback: no progress bar available
    tqdm = None


@dataclass
class Trade:
    direction: str  # "BUY" or "SELL"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_profit_price: float
    size: float  # abstract units sized by risk
    # Optional debug/trace fields for transparency
    stop_distance_pips: Optional[float] = None
    size_pre_cap: Optional[float] = None
    size_after_leverage_cap: Optional[float] = None
    size_after_rounding: Optional[float] = None
    size_after_max_cap: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    equity_after: Optional[float] = None  # equity immediately after trade closes


def backtest_strategy(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[BaseStrategy],
    params: Dict[str, Any],
    account_cfg: Dict[str, Any],
    max_lookback: Optional[int] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a backtest with SL/TP supporting configurable concurrent positions.

    Behavior:
      - Trades are opened at the NEXT bar's open after a signal.
      - SL is evaluated before TP per bar for conservative fills.
      - Spread applied half at entry and half at exit. Optional slippage.
      - Commission applied on open and close.
      - Position sizing: risk_per_trade * equity / (stop_pips * pip_value_per_lot).
      - Concurrency: controlled via account_cfg['max_concurrent_positions'] (default 1 for legacy behavior).
        * If == 1 the logic is unchanged (only seek a new trade if none open).
        * If > 1 the strategy can add trades while others are still open up to the cap.
    """

    df = data.copy()
    df = df.sort_index()

    starting_balance = float(account_cfg.get("starting_balance", 10_000))
    risk_frac = float(account_cfg.get("risk_per_trade", 0.01))
    spread_pips = float(account_cfg.get("spread_pips", 0.0))
    commission = float(account_cfg.get("commission_per_trade", 0.0))
    leverage = float(account_cfg.get("leverage", 0.0))
    # Optional realism/risk controls (all optional)
    slippage_pips = float(account_cfg.get("slippage_pips", 0.0))  # unfavorable slip per fill
    min_stop_pips = float(account_cfg.get("min_stop_pips", 0.0))  # skip trades with too-tight SL
    min_size = float(account_cfg.get("min_size", 0.0))            # skip trades smaller than this size
    lot_step = float(account_cfg.get("lot_step", 0.0))             # round position size down to this increment if > 0
    max_drawdown_stop_pct = float(account_cfg.get("max_drawdown_stop_pct", 0.0))  # pause trading if exceeded
    max_lot_size = float(account_cfg.get("max_lot_size", 0.0))  # hard cap on position size (0 disables)
    # Optional entry cooldown controls
    cooldown_bars_after_exit = int(account_cfg.get("cooldown_bars_after_exit", 0) or 0)
    no_same_bar_reentry = bool(account_cfg.get("no_same_bar_reentry", False))

    spread_price = pips_to_price(spread_pips, symbol)
    slippage_price = pips_to_price(slippage_pips, symbol)
    pip = pip_size(symbol)
    # Lot and pip value modeling
    contract_size = float(account_cfg.get("contract_size", 100_000))  # units per 1.0 lot
    pip_value_override = account_cfg.get("pip_value_per_lot")
    if pip_value_override is not None:
        try:
            pip_value_per_lot = float(pip_value_override)
        except Exception:
            pip_value_per_lot = contract_size * pip  # fallback
    else:
        # Approximate pip value in account currency when quote matches account currency (e.g., EURUSD with USD account)
        pip_value_per_lot = contract_size * pip
    equity_rounding = float(account_cfg.get("equity_rounding", 0.0))  # e.g., 0.01 to round to cents

    equity = starting_balance
    peak_equity = starting_balance
    equity_curve: List[Dict[str, float]] = []
    # Track multiple open trades if allowed. For legacy single-trade behavior this list will be length 0 or 1.
    open_trades: List[Trade] = []
    trades: List[Trade] = []
    signal_debug: List[Dict[str, Any]] = []
    # Track exit timing for cooldown logic
    last_exit_bar_index: int | None = None

    # Determine warmup for strategies needing history
    warmup = max_lookback if max_lookback is not None else 200

    index = df.index
    iter_range = range(warmup, len(df) - 1)

    # Optional per-run progress bar
    manual_progress = False
    manual_step_every = None
    total_steps = (len(df) - 1 - warmup)
    if progress and tqdm is not None:
        desc = progress.get("desc") or "Backtest"
        position = int(progress.get("position", 0))
        leave = bool(progress.get("leave", False))
        # Force-enable display even if not detected as TTY; better UX in VS Code terminal
        bar = tqdm(
            iter_range,
            total=total_steps,
            desc=desc,
            position=position,
            leave=leave,
            unit="step",
            dynamic_ncols=True,
            disable=False,
        )
    else:
        bar = iter_range
        # If progress requested but tqdm unavailable, provide periodic textual updates
        if progress and tqdm is None and total_steps > 0:
            manual_progress = True
            # Print ~20 updates across the run
            manual_step_every = max(1, total_steps // 20)
            # Initial header for ASCII progress bar
            desc = progress.get('desc') or 'Backtest'
            sys.stdout.write(f"{desc}: {total_steps} steps...\n")
            sys.stdout.flush()

    max_concurrent = int(account_cfg.get("max_concurrent_positions", 1) or 1)

    for i in bar:
        now = index[i]
        nxt = index[i + 1]
        window = df.iloc[: i + 1]
        closed_this_bar = False

        # Manual progress prints when tqdm is unavailable (single-line ASCII bar)
        if manual_progress and manual_step_every is not None:
            done = (i - warmup)
            if done % manual_step_every == 0 or done == total_steps - 1:
                pct = int(round((done + 1) * 100.0 / total_steps))
                bar_len = 40
                filled = min(bar_len, max(0, int(bar_len * pct / 100)))
                bar_txt = "█" * filled + " " * (bar_len - filled)
                # carriage-return update in place
                sys.stdout.write(f"\r[{bar_txt}] {done + 1}/{total_steps} ({pct}%)")
                sys.stdout.flush()
                if done == total_steps - 1:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

        # Update equity curve at current bar close
        equity_curve.append({"time": now.isoformat(), "equity": equity})

        # Drawdown-based stop: stop trading once drawdown exceeds threshold
        if max_drawdown_stop_pct and peak_equity > 0:
            peak_equity = max(peak_equity, equity)
            dd_frac = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if dd_frac * 100.0 >= max_drawdown_stop_pct:
                # Append final point and exit early
                # Next append for last bar happens after loop; we exit now to stop further trades
                break

        # If there are open trades, evaluate exits for each independently.
        if open_trades:
            h = float(df.iloc[i + 1]["High"])
            l = float(df.iloc[i + 1]["Low"])
            # Iterate over a COPY so we can remove closed trades from original list.
            for ot in list(open_trades):
                exit_price = None
                if ot.direction == "BUY":
                    if l <= ot.stop_price:
                        exit_price = ot.stop_price - spread_price * 0.5 - slippage_price
                    elif h >= ot.take_profit_price:
                        exit_price = ot.take_profit_price - spread_price * 0.5 - slippage_price
                else:  # SELL
                    if h >= ot.stop_price:
                        exit_price = ot.stop_price + spread_price * 0.5 + slippage_price
                    elif l <= ot.take_profit_price:
                        exit_price = ot.take_profit_price + spread_price * 0.5 + slippage_price
                if exit_price is not None:
                    ot.exit_time = nxt
                    ot.exit_price = float(exit_price)
                    direction_mult = 1.0 if ot.direction == "BUY" else -1.0
                    price_diff = (ot.exit_price - ot.entry_price)
                    pips_signed = (price_diff / pip) * direction_mult if pip > 0 else 0.0
                    pnl = pips_signed * pip_value_per_lot * ot.size
                    pnl -= commission
                    ot.pnl = pnl
                    equity += pnl
                    if equity_rounding and equity_rounding > 0:
                        equity = round(equity / equity_rounding) * equity_rounding
                    ot.equity_after = equity
                    trades.append(ot)
                    open_trades.remove(ot)
                    closed_this_bar = True
                    last_exit_bar_index = i + 1

        # Open new trade(s) if capacity available.
        # Legacy behavior: only attempt if no open trades when max_concurrent == 1.
        can_open_now = ((max_concurrent == 1 and not open_trades) or (max_concurrent > 1 and len(open_trades) < max_concurrent))
        # Apply re-entry cooldown rules
        if can_open_now and (no_same_bar_reentry and closed_this_bar):
            signal_debug.append({"time": now.isoformat(), "reason": "same_bar_reentry_blocked"})
            can_open_now = False
        if can_open_now and cooldown_bars_after_exit and last_exit_bar_index is not None:
            # Entry occurs at nxt (i+1); ensure enough bars have passed since last_exit_bar_index
            if (i + 1) - last_exit_bar_index < cooldown_bars_after_exit:
                signal_debug.append({"time": now.isoformat(), "reason": "cooldown_active", "cooldown_bars_after_exit": cooldown_bars_after_exit})
                can_open_now = False

        if can_open_now:
            strat = strategy_cls(window, params)
            action, sl_pips, tp_pips = strat.generate_signals()
            if action in ("BUY", "SELL") and sl_pips is not None and tp_pips is not None:
                # Enforce minimum stop distance if configured
                if min_stop_pips and sl_pips < min_stop_pips:
                    signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "reason": "below_min_stop"})
                    continue
                entry_price = float(df.iloc[i + 1]["Open"])
                if action == "BUY":
                    # Buy fills at ask (mid + half-spread) with unfavorable slippage increasing price
                    entry_price += spread_price * 0.5 + slippage_price
                    stop_price = entry_price - pips_to_price(sl_pips, symbol)
                    take_profit_price = entry_price + pips_to_price(tp_pips, symbol)
                else:  # SELL
                    # Sell fills at bid (mid - half-spread); unfavorable slippage lowers sell price further
                    entry_price -= spread_price * 0.5 + slippage_price
                    stop_price = entry_price + pips_to_price(sl_pips, symbol)
                    take_profit_price = entry_price - pips_to_price(tp_pips, symbol)

                # Risk-based sizing in lots using pip value
                # lots = (equity * risk_per_trade) / (stop_pips * pip_value_per_lot)
                stop_dist = abs(entry_price - stop_price)
                if stop_dist <= 0:
                    signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "reason": "non_positive_stop_dist"})
                    continue
                risk_amount = equity * risk_frac
                stop_pips_eff = float(sl_pips) if sl_pips is not None else (stop_dist / pip if pip > 0 else 0.0)
                if stop_pips_eff <= 0:
                    signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "reason": "non_positive_stop_pips"})
                    continue
                # Pre-cap position size in lots (risk model)
                size = max(risk_amount / (stop_pips_eff * pip_value_per_lot), 0.0)
                size_pre_cap = size
                # Respect leverage/margin if provided (cap notional exposure)
                if leverage and leverage > 0 and entry_price > 0:
                    # allowable lots = (equity * leverage) / (contract_size * entry_price)
                    allowable_size = (equity * leverage) / (contract_size * entry_price)
                    if allowable_size <= 0:
                        signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "size_pre_cap": size_pre_cap, "reason": "non_positive_allowable_size"})
                        continue
                    size = min(size, allowable_size)
                size_after_lev_cap = size
                # Enforce lot step increments (round DOWN to avoid exceeding risk)
                if lot_step and lot_step > 0:
                    # Floor to nearest multiple of lot_step
                    size = (size // lot_step) * lot_step
                size_after_rounding = size
                # Hard cap on absolute size after rounding/leverage
                if max_lot_size and max_lot_size > 0 and size > max_lot_size:
                    size = max_lot_size
                size_after_max_cap = size
                # Skip trades that are too small to be meaningful
                if min_size and size < min_size:
                    # Allow rounding up to min_size if resulting risk does not exceed risk_amount by more than tolerance
                    risk_tolerance_pct = float(account_cfg.get("risk_rounding_tolerance_pct", 5.0))  # default 5% tolerance
                    risk_if_min = min_size * pip_value_per_lot * stop_pips_eff
                    if risk_if_min <= risk_amount * (1 + risk_tolerance_pct / 100.0) and min_size <= (max_lot_size if max_lot_size else min_size):
                        size = min_size
                        size_after_max_cap = size  # update
                    else:
                        signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "size_pre_cap": size_pre_cap, "size_after_rounding": size_after_rounding, "size_after_max_cap": size_after_max_cap, "reason": "below_min_size"})
                        continue
                new_trade = Trade(
                    direction=action,
                    entry_time=nxt,
                    entry_price=float(entry_price),
                    stop_price=float(stop_price),
                    take_profit_price=float(take_profit_price),
                    size=float(size),
                    stop_distance_pips=float(stop_pips_eff),
                    size_pre_cap=float(size_pre_cap),
                    size_after_leverage_cap=float(size_after_lev_cap),
                    size_after_rounding=float(size_after_rounding),
                    size_after_max_cap=float(size_after_max_cap),
                )
                equity -= commission  # commission on open
                if equity_rounding and equity_rounding > 0:
                    equity = round(equity / equity_rounding) * equity_rounding
                signal_debug.append({"time": now.isoformat(), "action": action, "sl_pips": sl_pips, "tp_pips": tp_pips, "size_final": size, "reason": "accepted"})
                open_trades.append(new_trade)

    # Final equity at last bar
    if len(df) > 0:
        equity_curve.append({"time": df.index[-1].isoformat(), "equity": equity})

    # Metrics
    trade_pnls = np.array([t.pnl for t in trades if t.pnl is not None], dtype=float)
    wins = (trade_pnls > 0).sum() if trade_pnls.size else 0
    losses = (trade_pnls < 0).sum() if trade_pnls.size else 0
    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    total_return = (equity / starting_balance) - 1.0

    # Equity returns per step (robust to zero/NaN equity values)
    eq = np.array([pt["equity"] for pt in equity_curve], dtype=float)
    if eq.size > 1:
        # Safe element-wise division to avoid divide-by-zero warnings
        rets_raw = np.divide(np.diff(eq), eq[:-1], out=np.zeros_like(eq[:-1], dtype=float), where=eq[:-1] != 0)
        # Drop any NaN/inf that might still sneak in
        rets = rets_raw[np.isfinite(rets_raw)]
    else:
        rets = np.array([])

    # Use NaN-safe stats and explicit guards to avoid RuntimeWarnings
    rets_std = float(np.nanstd(rets)) if rets.size > 0 else 0.0
    rets_mean = float(np.nanmean(rets)) if rets.size > 0 else 0.0
    sharpe = float(np.sqrt(252) * (rets_mean / rets_std)) if rets.size > 1 and rets_std > 0 else 0.0
    mdd = max_drawdown(eq) if eq.size > 0 else 0.0
    
    # Additional consistency metrics
    avg_dd = average_drawdown(eq) if eq.size > 0 else 0.0
    calmar = calmar_ratio(total_return * 100.0, mdd * 100.0) if mdd > 0 else 0.0
    sortino = sortino_ratio(rets) if rets.size > 1 else 0.0
    rolling_sharpe_cons = rolling_sharpe_consistency(rets, window=21) if rets.size > 42 else 0.0
    
    # Combined consistency score (you can adjust weights)
    # Higher is better - rewards return while penalizing inconsistency
    consistency_score = (
        sortino * 0.4 +           # Sortino favors consistent returns
        calmar * 0.3 +            # Calmar rewards return per drawdown
        rolling_sharpe_cons * 0.2 + # Consistent rolling performance
        (1.0 - avg_dd) * 100 * 0.1  # Low average drawdown (scaled to similar range)
    )

    result = {
        "starting_balance": starting_balance,
        "ending_balance": equity,
        "total_return_pct": total_return * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd * 100.0,
        "avg_drawdown_pct": avg_dd * 100.0,
        "calmar": calmar,
        "sortino": sortino,
        "rolling_sharpe_consistency": rolling_sharpe_cons,
        "consistency_score": consistency_score,
        "trades": len(trades),
        "win_rate_pct": win_rate * 100.0,
        "trades_detail": [trade_to_dict(t) for t in trades],
        "equity_curve": equity_curve,
        # Debug info for diagnostics
        "signal_debug": signal_debug,
    }
    return result


def trade_to_dict(t: Trade) -> Dict[str, Any]:
    return {
        "direction": t.direction,
        "entry_time": t.entry_time.isoformat() if t.entry_time is not None else None,
        "entry_price": t.entry_price,
        "stop_price": t.stop_price,
        "take_profit_price": t.take_profit_price,
        "exit_time": t.exit_time.isoformat() if t.exit_time is not None else None,
        "exit_price": t.exit_price,
        "size": t.size,
        # Debug/trace fields
        "stop_distance_pips": t.stop_distance_pips,
        "size_pre_cap": t.size_pre_cap,
        "size_after_leverage_cap": t.size_after_leverage_cap,
        "size_after_rounding": t.size_after_rounding,
        "size_after_max_cap": t.size_after_max_cap,
        "pnl": t.pnl,
        "equity_after": t.equity_after,
    }


def max_drawdown(equity: np.ndarray) -> float:
    """Return max drawdown as fraction (e.g., 0.2 for 20%)."""
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    # Safe division to avoid divide-by-zero when equity ever hits 0
    dd = np.divide(peaks - equity, peaks, out=np.zeros_like(equity, dtype=float), where=peaks != 0)
    return float(dd.max() if dd.size else 0.0)


def average_drawdown(equity: np.ndarray) -> float:
    """
    Return average drawdown - better for consistency than max drawdown.
    Captures typical drawdown experience, not just worst case.
    """
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    # Safe division to avoid divide-by-zero when equity ever hits 0
    dd = np.divide(peaks - equity, peaks, out=np.zeros_like(equity, dtype=float), where=peaks != 0)
    # Only count periods where we're in drawdown (dd > 0)
    active_dd = dd[dd > 0]
    return float(active_dd.mean() if active_dd.size > 0 else 0.0)


def calmar_ratio(total_return_pct: float, max_dd_pct: float) -> float:
    """
    Calmar Ratio = Total Return / Max Drawdown
    Directly captures return per unit of drawdown risk.
    Higher is better (more return per unit of drawdown).
    Capped at +/- 100 to prevent extreme values.
    """
    if max_dd_pct <= 0:
        # No drawdown - return high value if positive return
        if total_return_pct > 0:
            return 100.0  # Cap at very good
        elif total_return_pct < 0:
            return -100.0  # Cap at very bad
        else:
            return 0.0
    
    calmar = total_return_pct / max_dd_pct
    # Cap at +/- 100 to prevent extreme values
    return max(-100.0, min(100.0, calmar))


def sortino_ratio(rets: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Sortino Ratio - like Sharpe but only penalizes downside volatility.
    Better for consistency as it doesn't penalize upside moves.
    Capped at +/- 10 to prevent extreme values.
    """
    if rets.size == 0:
        return 0.0
    # Clean any non-finite values just in case
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return 0.0

    mean_return = float(np.nanmean(rets))
    downside_rets = rets[rets < risk_free_rate]
    
    # If no downside, return max positive value (very good!)
    if downside_rets.size == 0:
        return 10.0 if mean_return > 0 else 0.0
    
    downside_std = float(np.nanstd(downside_rets))
    if downside_std == 0:
        return 10.0 if mean_return > 0 else 0.0
    
    sortino = float(np.sqrt(252) * (mean_return - risk_free_rate) / downside_std)
    # Cap at +/- 10 to prevent extreme values
    return max(-10.0, min(10.0, sortino))


def rolling_sharpe_consistency(rets: np.ndarray, window: int = 21) -> float:
    """
    Measure consistency of rolling Sharpe ratios.
    Lower standard deviation = more consistent performance.
    Returns average rolling Sharpe minus its standard deviation.
    """
    if rets.size < window * 2:
        return 0.0
    
    rolling_sharpes = []
    for i in range(window, len(rets)):
        window_rets = rets[i-window:i]
        if window_rets.std() > 0:
            rs = window_rets.mean() / window_rets.std()
            rolling_sharpes.append(rs)
    
    if len(rolling_sharpes) == 0:
        return 0.0
    
    rs_array = np.array(rolling_sharpes)
    # Higher mean and lower std is better
    return float(rs_array.mean() - rs_array.std())
