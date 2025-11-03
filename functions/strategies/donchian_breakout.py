"""
Donchian Channel Breakout Strategy

Uses rolling highest high and lowest low over a lookback window.
Buy when Close breaks above upper channel; sell when Close breaks below lower channel.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class DonchianBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        lookback = int(self.config["lookback"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < lookback + 2:
            return "HOLD", None, None

        df["upper"] = df["High"].rolling(window=lookback, min_periods=lookback).max()
        df["lower"] = df["Low"].rolling(window=lookback, min_periods=lookback).min()

        last = df.iloc[-1]
        prev = df.iloc[-2]
        if pd.isna(last["upper"]) or pd.isna(last["lower"]) or pd.isna(prev["upper"]) or pd.isna(prev["lower"]):
            return "HOLD", None, None

        if prev["Close"] <= prev["upper"] and last["Close"] > last["upper"]:
            return "BUY", sl_pips, tp_pips
        if prev["Close"] >= prev["lower"] and last["Close"] < last["lower"]:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
