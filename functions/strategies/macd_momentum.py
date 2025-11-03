"""
MACD Momentum Strategy

Momentum strategy using MACD (Moving Average Convergence Divergence).
Trades based on MACD line crossing the signal line.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class MACDMomentum(BaseStrategy):
    """MACD momentum strategy.
    
    Logic:
    - Calculate MACD line (fast EMA - slow EMA)
    - Calculate signal line (EMA of MACD line)
    - Buy when MACD crosses above signal line (bullish momentum)
    - Sell when MACD crosses below signal line (bearish momentum)
    
    Parameters:
    - macd_fast: Fast EMA period (typically 12)
    - macd_slow: Slow EMA period (typically 26)
    - macd_signal: Signal line EMA period (typically 9)
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        macd_fast = int(self.config["macd_fast"])
        macd_slow = int(self.config["macd_slow"])
        macd_signal = int(self.config["macd_signal"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        df = self.data.copy()
        
        # Need enough data for MACD calculation
        min_periods = macd_slow + macd_signal + 2
        if len(df) < min_periods:
            return "HOLD", None, None
        
        # Calculate MACD
        ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["signal_line"] = df["macd_line"].ewm(span=macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["signal_line"]
        
        # Get current and previous values
        current_macd = df["macd_line"].iloc[-1]
        current_signal = df["signal_line"].iloc[-1]
        prev_macd = df["macd_line"].iloc[-2]
        prev_signal = df["signal_line"].iloc[-2]
        
        # Check for valid values
        if pd.isna(current_macd) or pd.isna(current_signal):
            return "HOLD", None, None
        
        # Signal logic: detect crosses
        # Buy when MACD crosses above signal line
        if prev_macd <= prev_signal and current_macd > current_signal:
            return "BUY", sl_pips, tp_pips
        
        # Sell when MACD crosses below signal line
        if prev_macd >= prev_signal and current_macd < current_signal:
            return "SELL", sl_pips, tp_pips
        
        return "HOLD", None, None
