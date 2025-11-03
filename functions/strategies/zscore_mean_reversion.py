"""
Z-Score Mean Reversion Strategy

Computes z-score of price relative to rolling mean and std.
Buys when z < -threshold, sells when z > threshold.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class ZScoreMeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        period = int(self.config["lookback"])
        z_thresh = float(self.config["z_threshold"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < period + 2:
            return "HOLD", None, None

        mean = df["Close"].rolling(period).mean()
        std = df["Close"].rolling(period).std()
        z = (df["Close"] - mean) / std.replace(0, pd.NA)

        prev_z, last_z = z.iloc[-2], z.iloc[-1]
        if pd.isna(prev_z) or pd.isna(last_z):
            return "HOLD", None, None

        if last_z < -abs(z_thresh):
            return "BUY", sl_pips, tp_pips
        if last_z > abs(z_thresh):
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
