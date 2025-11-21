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
    """Keltner squeeze breakout with dynamic SL/TP based on prior bar extremes.

    Contract: returns (action, sl_pips, tp_pips)
    - sl_pips and tp_pips are computed from ratios relative to prior bar high/low.
    - For BUY:
        base = max(Open[-1] - Low[-2], 0)
        SL price = Open[-1] - sl_ratio * base
        TP price = Open[-1] + tp_ratio * base
    - For SELL:
        base = max(High[-2] - Open[-1], 0)
        SL price = Open[-1] + sl_ratio * base
        TP price = Open[-1] - tp_ratio * base
    Distances are converted to pips using a simple FX heuristic:
        pip = 0.01 if price >= 10 else 0.0001
    """

    def _infer_pip(self, ref_price: float) -> float:
        # Heuristic for FX: JPY-like pairs ~ >10 price level use 0.01 pip, else 0.0001
        return 0.01 if float(ref_price) >= 10.0 else 0.0001

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("period", 20))
        bb_k = float(cfg.get("bb_std", 2.0))
        kc_mult = float(cfg.get("kc_mult", 1.5))
        sl_ratio = float(cfg.get("sl_ratio", 1.0))
        tp_ratio = float(cfg.get("tp_ratio", 2.0))

        df = self.data
        c = df["Close"]
        if len(c) < period + 2:
            return "HOLD", None, None

        _, bb_u, bb_l = _bb(c, period, bb_k)
        _, kc_u, kc_l = _kc(df, period, kc_mult)

        # Squeeze if BB inside Keltner (width comparison via containment)
        if any(pd.isna(x) for x in [bb_u.iloc[-1], bb_l.iloc[-1], kc_u.iloc[-1], kc_l.iloc[-1]]):
            return "HOLD", None, None
        squeeze_prev = (bb_u.iloc[-2] <= kc_u.iloc[-2]) and (bb_l.iloc[-2] >= kc_l.iloc[-2])

        # Current (last closed) bar and previous bar
        open_curr = float(df["Open"].iloc[-1])
        prev_low = float(df["Low"].iloc[-2])
        prev_high = float(df["High"].iloc[-2])
        pip = self._infer_pip(open_curr)

        cl = float(c.iloc[-1])
        if squeeze_prev and cl > float(bb_u.iloc[-1]):
            # BUY setup
            base = max(open_curr - prev_low, 0.0)
            if base <= 0:
                return "HOLD", None, None
            sl_dist = sl_ratio * base
            tp_dist = tp_ratio * base
            sl_pips = sl_dist / pip
            tp_pips = tp_dist / pip
            return "BUY", sl_pips, tp_pips

        if squeeze_prev and cl < float(bb_l.iloc[-1]):
            # SELL setup
            base = max(prev_high - open_curr, 0.0)
            if base <= 0:
                return "HOLD", None, None
            sl_dist = sl_ratio * base
            tp_dist = tp_ratio * base
            sl_pips = sl_dist / pip
            tp_pips = tp_dist / pip
            return "SELL", sl_pips, tp_pips

        return "HOLD", None, None
