"""
RSI With MA Trend Filter

Mean-reversion entries gated by a moving average trend filter.
Buy oversold only if price is above MA (uptrend). Sell overbought only if price is below MA (downtrend).
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class RSIWithMAFilter(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        rsi_period = int(self.config["rsi_period"])
        ma_period = int(self.config["ma_period"])
        oversold = float(self.config["oversold"])
        overbought = float(self.config["overbought"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < max(rsi_period, ma_period) + 2:
            return "HOLD", None, None

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MA trend filter (EMA for responsiveness)
        df["ma"] = df["Close"].ewm(span=ma_period, adjust=False).mean()

        rsi_prev, rsi_last = df["rsi"].iloc[-2], df["rsi"].iloc[-1]
        close_prev, close_last = df["Close"].iloc[-2], df["Close"].iloc[-1]
        ma_prev, ma_last = df["ma"].iloc[-2], df["ma"].iloc[-1]

        if any(pd.isna(x) for x in [rsi_prev, rsi_last, ma_prev, ma_last]):
            return "HOLD", None, None

        uptrend = close_last > ma_last
        downtrend = close_last < ma_last

        # Buy if RSI crosses up from below oversold and uptrend
        if rsi_prev < oversold <= rsi_last and uptrend:
            return "BUY", sl_pips, tp_pips
        # Sell if RSI crosses down from above overbought and downtrend
        if rsi_prev > overbought >= rsi_last and downtrend:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
