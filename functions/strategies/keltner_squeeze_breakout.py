"""
Keltner Squeeze Breakout Strategy

When Bollinger Band width is below Keltner Channel width (squeeze), trade
breakouts above/below BB.
"""
from typing import Tuple, Optional
import pandas as pd

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
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    ma = _ema(c, n)
    upper = ma + atr_mult * atr
    lower = ma - atr_mult * atr
    return ma, upper, lower


class KeltnerSqueezeBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("period", 20))
        bb_k = float(cfg.get("bb_std", 2.0))
        kc_mult = float(cfg.get("kc_mult", 1.5))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        c = df["Close"]
        if len(c) < period + 2:
            return "HOLD", None, None

        _, bb_u, bb_l = _bb(c, period, bb_k)
        _, kc_u, kc_l = _kc(df, period, kc_mult)

        # Squeeze if BB inside Keltner (width comparison via containment)
        squeeze_prev = (bb_u.iloc[-2] <= kc_u.iloc[-2]) and (bb_l.iloc[-2] >= kc_l.iloc[-2])
        if any(pd.isna(x) for x in [bb_u.iloc[-1], bb_l.iloc[-1], kc_u.iloc[-1], kc_l.iloc[-1]]):
            return "HOLD", None, None

        cl = c.iloc[-1]
        if squeeze_prev and cl > bb_u.iloc[-1]:
            return "BUY", sl_pips, tp_pips
        if squeeze_prev and cl < bb_l.iloc[-1]:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
