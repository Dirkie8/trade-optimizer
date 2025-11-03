"""
Stochastic Oscillator Strategy

Momentum strategy using the Stochastic Oscillator.
Identifies overbought/oversold conditions and momentum shifts.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class StochasticOscillator(BaseStrategy):
    """Stochastic Oscillator momentum strategy.
    
    Logic:
    - Calculate %K line: (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - Calculate %D line: SMA of %K (signal line)
    - Buy when %K crosses above %D in oversold region
    - Sell when %K crosses below %D in overbought region
    
    Parameters:
    - stoch_k_period: Lookback period for %K calculation
    - stoch_d_period: Period for %D (SMA of %K)
    - oversold: Level below which we look for buy signals (typically 20)
    - overbought: Level above which we look for sell signals (typically 80)
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        k_period = int(self.config["stoch_k_period"])
        d_period = int(self.config["stoch_d_period"])
        oversold = float(self.config["oversold"])
        overbought = float(self.config["overbought"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        df = self.data.copy()
        
        # Need enough data for stochastic calculation
        if len(df) < k_period + d_period + 2:
            return "HOLD", None, None
        
        # Calculate %K line
        df["lowest_low"] = df["Low"].rolling(window=k_period).min()
        df["highest_high"] = df["High"].rolling(window=k_period).max()
        
        # Avoid division by zero
        denominator = df["highest_high"] - df["lowest_low"]
        denominator = denominator.replace(0, np.nan)
        
        df["stoch_k"] = ((df["Close"] - df["lowest_low"]) / denominator) * 100
        
        # Calculate %D line (signal line)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        
        # Get current and previous values
        current_k = df["stoch_k"].iloc[-1]
        current_d = df["stoch_d"].iloc[-1]
        prev_k = df["stoch_k"].iloc[-2]
        prev_d = df["stoch_d"].iloc[-2]
        
        # Check for valid values
        if pd.isna(current_k) or pd.isna(current_d):
            return "HOLD", None, None
        
        # Signal logic: crosses in extreme zones
        # Buy when %K crosses above %D in oversold region
        if current_k < oversold and prev_k <= prev_d and current_k > current_d:
            return "BUY", sl_pips, tp_pips
        
        # Sell when %K crosses below %D in overbought region
        if current_k > overbought and prev_k >= prev_d and current_k < current_d:
            return "SELL", sl_pips, tp_pips
        
        return "HOLD", None, None
