"""
Keltner Channel Breakout Strategy Trend Volume

EMA-based midline with ATR bands; signal on breakouts.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _bb(c: pd.Series, n: int, k: float):
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def _kc(df: pd.DataFrame, n: int, atr_mult: float):
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()
    ma = _ema(c, n)

    upper = ma + atr_mult * atr
    lower = ma - atr_mult * atr

    return ma, upper, lower, atr


class KeltnerSqueezeBreakTrend(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config

        period = int(cfg.get("period", 20))
        bb_k = float(cfg.get("bb_std", 2.0))
        kc_mult = float(cfg.get("kc_mult", 1.5))

        trend_period = int(cfg.get("trend_ema", 200))
        compression_thresh = float(cfg.get("compression_ratio", 0.7))
        breakout_atr_mult = float(cfg.get("breakout_atr_mult", 0.25))
        volume_mult = float(cfg.get("volume_mult", 1.2))

        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        c = df["Close"]

        if len(df) < max(period, trend_period, 20) + 5:
            return "HOLD", None, None

        # Indicators
        _, bb_u, bb_l = _bb(c, period, bb_k)
        _, kc_u, kc_l, atr = _kc(df, period, kc_mult)
        trend = _ema(c, trend_period)

        volume = df["Volume"]
        vol_ma = volume.rolling(20).mean()

        # Latest
        i = -1
        close = c.iloc[i]

        # Squeeze detection
        bb_width = bb_u - bb_l
        kc_width = kc_u - kc_l
        compression = bb_width / kc_width

        squeeze_now = (bb_u.iloc[i] <= kc_u.iloc[i]) and (bb_l.iloc[i] >= kc_l.iloc[i])
        squeeze_prev = (bb_u.iloc[i-1] <= kc_u.iloc[i-1]) and (bb_l.iloc[i-1] >= kc_l.iloc[i-1])

        # Must come OUT of a squeeze to trade: previously squeezed, now not squeezed
        if not (squeeze_prev and not squeeze_now):
            return "HOLD", None, None

        # Require strong compression (guard division by zero / degenerate widths)
        if kc_width.iloc[i-1] <= 0 or compression.iloc[i-1] > compression_thresh:
            return "HOLD", None, None

        # Breakout strength (guard ATR availability)
        if pd.isna(atr.iloc[i]) or abs(close - c.iloc[i-1]) < breakout_atr_mult * atr.iloc[i]:
            return "HOLD", None, None

        # Volume confirmation (guard MA availability)
        if pd.isna(vol_ma.iloc[i]) or volume.iloc[i] < volume_mult * vol_ma.iloc[i]:
            return "HOLD", None, None

        # Trend filter
        is_bull = close > trend.iloc[i]
        is_bear = close < trend.iloc[i]

        # Breakout direction
        if close > bb_u.iloc[i] and is_bull:
            return "BUY", sl_pips, tp_pips

        if close < bb_l.iloc[i] and is_bear:
            return "SELL", sl_pips, tp_pips

        return "HOLD", None, None
