"""
ROC Momentum Strategy

Uses Rate of Change (percent) over a lookback window.
Buy when ROC exceeds positive threshold; sell when below negative threshold.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class ROCMomentum(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        period = int(self.config["roc_period"])
        upper = float(self.config["upper_thresh"])
        lower = float(self.config["lower_thresh"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < period + 2:
            return "HOLD", None, None

        roc = df["Close"].pct_change(periods=period) * 100.0
        prev_roc, last_roc = roc.iloc[-2], roc.iloc[-1]
        if pd.isna(prev_roc) or pd.isna(last_roc):
            return "HOLD", None, None

        if prev_roc <= upper and last_roc > upper:
            return "BUY", sl_pips, tp_pips
        if prev_roc >= -abs(lower) and last_roc < -abs(lower):
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
