"""
Bollinger Band Mean Reversion Strategy

Contrarian approach: buy when price closes below lower band, sell when closes above upper band,
anticipating reversion toward the middle band.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class BollingerMeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        period = int(self.config["bb_period"])
        std_mult = float(self.config["bb_std_dev"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < period + 2:
            return "HOLD", None, None

        mid = df["Close"].rolling(period).mean()
        sd = df["Close"].rolling(period).std()
        upper = mid + std_mult * sd
        lower = mid - std_mult * sd

        prev_close, last_close = df["Close"].iloc[-2], df["Close"].iloc[-1]
        prev_upper, prev_lower = upper.iloc[-2], lower.iloc[-2]
        last_upper, last_lower = upper.iloc[-1], lower.iloc[-1]
        if pd.isna(last_upper) or pd.isna(last_lower) or pd.isna(prev_upper) or pd.isna(prev_lower):
            return "HOLD", None, None

        # Mean reversion entries on closes outside the bands (opposite of breakout)
        if prev_close >= prev_upper and last_close > last_upper:
            # Extended above upper band -> fade
            return "SELL", sl_pips, tp_pips
        if prev_close <= prev_lower and last_close < last_lower:
            # Extended below lower band -> fade
            return "BUY", sl_pips, tp_pips
        return "HOLD", None, None
