"""
RSI Mean Reversion Strategy

Classic mean reversion strategy using the Relative Strength Index (RSI).
Buy when RSI drops below oversold threshold, sell when it rises above overbought threshold.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy.
    
    Logic:
    - Calculate RSI over specified period
    - Buy when RSI crosses below oversold level (e.g., 30)
    - Sell when RSI crosses above overbought level (e.g., 70)
    - Mean reversion assumes price will return to average after extremes
    
    Parameters:
    - rsi_period: Lookback period for RSI calculation
    - oversold: RSI level below which we consider buying
    - overbought: RSI level above which we consider selling
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        rsi_period = int(self.config["rsi_period"])
        oversold = float(self.config["oversold"])
        overbought = float(self.config["overbought"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        df = self.data.copy()
        
        # Need enough data for RSI calculation
        if len(df) < rsi_period + 2:
            return "HOLD", None, None
        
        # Calculate RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Get current and previous RSI values
        current_rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2]
        
        # Check for valid RSI values
        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            return "HOLD", None, None
        
        # Signal logic: detect crosses
        # Buy when RSI crosses below oversold (was above, now below)
        if prev_rsi >= oversold and current_rsi < oversold:
            return "BUY", sl_pips, tp_pips
        
        # Sell when RSI crosses above overbought (was below, now above)
        if prev_rsi <= overbought and current_rsi > overbought:
            return "SELL", sl_pips, tp_pips
        
        return "HOLD", None, None
