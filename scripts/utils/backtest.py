from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from functions.base_strategy import BaseStrategy
from scripts.utils.data_utils import pips_to_price, pip_size
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class Trade:
    direction: str  # "BUY" or "SELL"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_profit_price: float
    size: float  # abstract units sized by risk
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


def backtest_strategy(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[BaseStrategy],
    params: Dict[str, Any],
    account_cfg: Dict[str, Any],
    max_lookback: Optional[int] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a simple one-position-at-a-time backtest with SL/TP.

    - Positions are opened at the next bar's open.
    - SL is checked before TP within each bar to be conservative.
    - Spread is applied half at entry and half at exit.
    - Commission is subtracted at close as a fixed amount.
    - Position sizing: risk_per_trade * balance divided by stop distance.
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
    max_drawdown_stop_pct = float(account_cfg.get("max_drawdown_stop_pct", 0.0))  # pause trading if exceeded

    spread_price = pips_to_price(spread_pips, symbol)
    slippage_price = pips_to_price(slippage_pips, symbol)
    pip = pip_size(symbol)

    equity = starting_balance
    peak_equity = starting_balance
    equity_curve: List[Dict[str, float]] = []
    open_trade: Optional[Trade] = None
    trades: List[Trade] = []

    # Determine warmup for strategies needing history
    warmup = max_lookback if max_lookback is not None else 200

    index = df.index
    iter_range = range(warmup, len(df) - 1)

    # Optional per-run progress bar
    if progress and tqdm is not None:
        desc = progress.get("desc") or "Backtest"
        position = int(progress.get("position", 0))
        leave = bool(progress.get("leave", False))
        bar = tqdm(iter_range, total=(len(df) - 1 - warmup), desc=desc, position=position, leave=leave, unit="step")
    else:
        bar = iter_range

    for i in bar:
        now = index[i]
        nxt = index[i + 1]
        window = df.iloc[: i + 1]

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

        # If trade is open, check if SL/TP hit within next bar
        if open_trade is not None:
            # Next bar's OHLC
            o = float(df.iloc[i + 1]["Open"])  # entry/exit use next bar open
            h = float(df.iloc[i + 1]["High"])
            l = float(df.iloc[i + 1]["Low"])

            # Determine exit
            exit_price = None
            if open_trade.direction == "BUY":
                # SL first
                if l <= open_trade.stop_price:
                    exit_price = open_trade.stop_price - spread_price * 0.5 - slippage_price
                elif h >= open_trade.take_profit_price:
                    exit_price = open_trade.take_profit_price - spread_price * 0.5 - slippage_price
            else:  # SELL
                if h >= open_trade.stop_price:
                    exit_price = open_trade.stop_price + spread_price * 0.5 + slippage_price
                elif l <= open_trade.take_profit_price:
                    exit_price = open_trade.take_profit_price + spread_price * 0.5 + slippage_price

            if exit_price is not None:
                # Close at decided price
                open_trade.exit_time = nxt
                open_trade.exit_price = float(exit_price)
                direction_mult = 1.0 if open_trade.direction == "BUY" else -1.0
                pnl = (open_trade.exit_price - open_trade.entry_price) * direction_mult * open_trade.size
                pnl -= commission  # commission on close
                open_trade.pnl = pnl
                equity += pnl
                trades.append(open_trade)
                open_trade = None

        # If no open trade, get signal at current bar close and open on next bar open
        if open_trade is None:
            strat = strategy_cls(window, params)
            action, sl_pips, tp_pips = strat.generate_signals()
            if action in ("BUY", "SELL") and sl_pips and tp_pips:
                # Enforce minimum stop distance if configured
                if min_stop_pips and sl_pips < min_stop_pips:
                    continue
                entry_price = float(df.iloc[i + 1]["Open"])
                if action == "BUY":
                    entry_price += spread_price * 0.5 + slippage_price
                    stop_price = entry_price - pips_to_price(sl_pips, symbol)
                    take_profit_price = entry_price + pips_to_price(tp_pips, symbol)
                else:  # SELL
                    entry_price -= spread_price * 0.5 - slippage_price
                    stop_price = entry_price + pips_to_price(sl_pips, symbol)
                    take_profit_price = entry_price - pips_to_price(tp_pips, symbol)

                stop_dist = abs(entry_price - stop_price)
                if stop_dist <= 0:
                    continue
                risk_amount = equity * risk_frac
                size = max(risk_amount / stop_dist, 0.0)
                # Respect leverage/margin if provided (cap notional exposure)
                if leverage and leverage > 0 and entry_price > 0:
                    allowable_size = (equity * leverage) / entry_price
                    if allowable_size <= 0:
                        continue
                    size = min(size, allowable_size)
                # Skip trades that are too small to be meaningful
                if min_size and size < min_size:
                    continue
                open_trade = Trade(
                    direction=action,
                    entry_time=nxt,
                    entry_price=float(entry_price),
                    stop_price=float(stop_price),
                    take_profit_price=float(take_profit_price),
                    size=float(size),
                )
                equity -= commission  # commission on open

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
        "pnl": t.pnl,
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
