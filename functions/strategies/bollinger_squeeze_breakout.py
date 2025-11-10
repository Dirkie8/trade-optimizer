"""
Bollinger Squeeze Breakout Strategy

Waits for low Bollinger Band width (squeeze) then trades breakouts above/below
the bands.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _bb(close: pd.Series, period: int, std_mult: float):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + std_mult * sd
    lower = ma - std_mult * sd
    width = (upper - lower) / ma
    return ma, upper, lower, width


class BollingerSqueezeBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("bb_period", 20))
        std_mult = float(cfg.get("bb_std", 2.0))
        width_threshold = float(cfg.get("width_threshold", 0.01))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        close = self.data["Close"]
        if len(close) < period + 2:
            return "HOLD", None, None

        ma, upper, lower, width = _bb(close, period, std_mult)
        if any(pd.isna(x) for x in [ma.iloc[-2], upper.iloc[-2], lower.iloc[-2], width.iloc[-2],
                                     ma.iloc[-1], upper.iloc[-1], lower.iloc[-1], width.iloc[-1]]):
            return "HOLD", None, None

        # Squeeze on previous bar, breakout on current close
        if width.iloc[-2] <= width_threshold and close.iloc[-1] > upper.iloc[-1]:
            return "BUY", sl_pips, tp_pips
        if width.iloc[-2] <= width_threshold and close.iloc[-1] < lower.iloc[-1]:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
