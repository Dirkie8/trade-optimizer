"""
PPO Crossover Strategy

Percentage Price Oscillator (PPO) with signal line crossover.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


class PPOCrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        c = self.data["Close"]
        cfg = self.config
        fast = int(cfg.get("fast_period", 12))
        slow = int(cfg.get("slow_period", 26))
        signal_n = int(cfg.get("signal_period", 9))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        if slow <= fast:
            slow = fast + 1
        if len(c) < slow + signal_n + 2:
            return "HOLD", None, None

        ppo = (_ema(c, fast) - _ema(c, slow)) / _ema(c, slow) * 100.0
        sig = _ema(ppo, signal_n)
        p_prev, p_last = ppo.iloc[-2], ppo.iloc[-1]
        s_prev, s_last = sig.iloc[-2], sig.iloc[-1]
        if any(pd.isna(x) for x in [p_prev, p_last, s_prev, s_last]):
            return "HOLD", None, None

        if p_prev <= s_prev and p_last > s_last:
            return "BUY", sl_pips, tp_pips
        if p_prev >= s_prev and p_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
