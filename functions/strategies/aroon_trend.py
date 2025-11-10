"""
Aroon Trend Strategy

Uses Aroon Up/Down to detect emerging trends. Enters long when Aroon Up crosses
above Aroon Down and exceeds a threshold; enters short for the opposite.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _distance_since_last_true(mask: pd.Series) -> pd.Series:
    """Return number of bars since the last True in a boolean Series.
    Fast O(n) implementation to avoid slow rolling.apply.
    """
    arr = mask.fillna(False).to_numpy(dtype=bool)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    last = -1_000_000_000  # effectively -inf sentinel
    for i, b in enumerate(arr):
        if b:
            last = i
        if last > -1_000_000_000:
            out[i] = i - last
    return pd.Series(out, index=mask.index)


def _aroon(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Vectorized Aroon using distances since last rolling max/min.

    aroon_up = 100 * (period - 1 - bars_since_last_high) / (period - 1)
    aroon_down = 100 * (period - 1 - bars_since_last_low) / (period - 1)
    """
    high = df["High"]
    low = df["Low"]

    roll_max = high.rolling(period).max()
    roll_min = low.rolling(period).min()
    is_max = high == roll_max
    is_min = low == roll_min

    dist_high = _distance_since_last_true(is_max)
    dist_low = _distance_since_last_true(is_min)

    denom = max(period - 1, 1)
    aroon_up = 100 * (denom - dist_high) / denom
    aroon_down = 100 * (denom - dist_low) / denom
    # Clip to [0, 100] and keep NaNs where undefined
    aroon_up = aroon_up.clip(lower=0, upper=100)
    aroon_down = aroon_down.clip(lower=0, upper=100)
    # Leading region (< period) will naturally be NaN from rolling
    return aroon_up, aroon_down


class AroonTrend(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("aroon_period", 25))
        threshold = float(cfg.get("aroon_threshold", 60))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 2:
            return "HOLD", None, None

        up, down = _aroon(df, period)
        up_prev, up_last = up.iloc[-2], up.iloc[-1]
        down_prev, down_last = down.iloc[-2], down.iloc[-1]
        if any(pd.isna(x) for x in [up_prev, up_last, down_prev, down_last]):
            return "HOLD", None, None

        # Cross and threshold filter
        if up_prev <= down_prev and up_last > down_last and up_last >= threshold:
            return "BUY", sl_pips, tp_pips
        if down_prev <= up_prev and down_last > up_last and down_last >= threshold:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
