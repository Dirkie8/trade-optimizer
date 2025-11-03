"""
Ichimoku Kijun Cross Strategy

Simplified Ichimoku setup using Tenkan-sen and Kijun-sen cross.
Buy when Tenkan crosses above Kijun; sell on cross below.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class IchimokuKijunCross(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        tenkan = int(self.config.get("tenkan_period", 9))
        kijun = int(self.config.get("kijun_period", 26))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        if tenkan <= 0 or kijun <= 0:
            return "HOLD", None, None

        df = self.data.copy()
        if len(df) < max(tenkan, kijun) + 2:
            return "HOLD", None, None

        tenkan_line = (df["High"].rolling(tenkan).max() + df["Low"].rolling(tenkan).min()) / 2
        kijun_line = (df["High"].rolling(kijun).max() + df["Low"].rolling(kijun).min()) / 2

        prev_t, last_t = tenkan_line.iloc[-2], tenkan_line.iloc[-1]
        prev_k, last_k = kijun_line.iloc[-2], kijun_line.iloc[-1]
        if any(pd.isna(x) for x in [prev_t, last_t, prev_k, last_k]):
            return "HOLD", None, None

        if prev_t <= prev_k and last_t > last_k:
            return "BUY", sl_pips, tp_pips
        if prev_t >= prev_k and last_t < last_k:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
