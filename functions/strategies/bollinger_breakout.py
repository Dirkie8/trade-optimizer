"""
Bollinger Band Breakout Strategy

Breakout strategy using Bollinger Bands.
Enter positions when price breaks outside the bands, expecting continuation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class BollingerBreakout(BaseStrategy):
    """Bollinger Band breakout strategy.
    
    Logic:
    - Calculate Bollinger Bands (SMA ± std_dev * standard deviation)
    - Buy when price breaks above upper band (bullish breakout)
    - Sell when price breaks below lower band (bearish breakout)
    - Assumes strong momentum will continue in breakout direction
    
    Parameters:
    - bb_period: Period for moving average calculation
    - bb_std_dev: Number of standard deviations for bands
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        bb_period = int(self.config["bb_period"])
        bb_std_dev = float(self.config["bb_std_dev"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        df = self.data.copy()
        
        # Need enough data for BB calculation
        if len(df) < bb_period + 2:
            return "HOLD", None, None
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["Close"].rolling(window=bb_period).mean()
        df["bb_std"] = df["Close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * bb_std_dev)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * bb_std_dev)
        
        # Get current and previous values
        current_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        current_upper = df["bb_upper"].iloc[-1]
        current_lower = df["bb_lower"].iloc[-1]
        prev_upper = df["bb_upper"].iloc[-2]
        prev_lower = df["bb_lower"].iloc[-2]
        
        # Check for valid values
        if pd.isna(current_upper) or pd.isna(current_lower):
            return "HOLD", None, None
        
        # Signal logic: detect breakouts
        # Buy when price breaks above upper band
        if prev_close <= prev_upper and current_close > current_upper:
            return "BUY", sl_pips, tp_pips
        
        # Sell when price breaks below lower band
        if prev_close >= prev_lower and current_close < current_lower:
            return "SELL", sl_pips, tp_pips
        
        return "HOLD", None, None
