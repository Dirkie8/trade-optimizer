"""
KST (Know Sure Thing) Crossover Strategy

Simplified KST implementation with four ROC components and SMA smoothing.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _roc(x: pd.Series, n: int) -> pd.Series:
    return x.pct_change(periods=n) * 100.0


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def _kst(c: pd.Series, n1: int, n2: int, n3: int, n4: int, s1: int, s2: int, s3: int, s4: int) -> pd.Series:
    rc1 = _sma(_roc(c, n1), s1)
    rc2 = _sma(_roc(c, n2), s2)
    rc3 = _sma(_roc(c, n3), s3)
    rc4 = _sma(_roc(c, n4), s4)
    return rc1 + 2 * rc2 + 3 * rc3 + 4 * rc4


class KSTCrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        n1 = int(cfg.get("n1", 10)); s1 = int(cfg.get("s1", 10))
        n2 = int(cfg.get("n2", 15)); s2 = int(cfg.get("s2", 10))
        n3 = int(cfg.get("n3", 20)); s3 = int(cfg.get("s3", 10))
        n4 = int(cfg.get("n4", 30)); s4 = int(cfg.get("s4", 15))
        sig = int(cfg.get("signal_period", 9))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        c = self.data["Close"]
        min_len = max(n1 + s1, n2 + s2, n3 + s3, n4 + s4) + sig + 2
        if len(c) < min_len:
            return "HOLD", None, None

        kst = _kst(c, n1, n2, n3, n4, s1, s2, s3, s4)
        signal = kst.ewm(span=sig, adjust=False).mean()

        k_prev, k_last = kst.iloc[-2], kst.iloc[-1]
        s_prev, s_last = signal.iloc[-2], signal.iloc[-1]
        if any(pd.isna(x) for x in [k_prev, k_last, s_prev, s_last]):
            return "HOLD", None, None

        if k_prev <= s_prev and k_last > s_last:
            return "BUY", sl_pips, tp_pips
        if k_prev >= s_prev and k_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
