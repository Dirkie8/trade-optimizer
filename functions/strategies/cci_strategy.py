"""
CCI Strategy

Commodity Channel Index-based signals.
Buy when CCI crosses above +100 (from below), sell when crosses below -100 (from above).
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class CCIStrategy(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        period = int(self.config["cci_period"])
        thresh = float(self.config.get("threshold", 100))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < period + 2:
            return "HOLD", None, None

        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        sma_tp = tp.rolling(period).mean()
        mad = (tp - sma_tp).abs().rolling(period).mean()
        cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

        prev, last = cci.iloc[-2], cci.iloc[-1]
        if pd.isna(prev) or pd.isna(last):
            return "HOLD", None, None

        if prev <= thresh and last > thresh:
            return "BUY", sl_pips, tp_pips
        if prev >= -thresh and last < -thresh:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
