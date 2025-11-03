"""
SIMPLIFIED STRATEGY TEMPLATE - Quick 2-Minute Guide
====================================================

STEPS TO CREATE YOUR STRATEGY:
1. Copy this file to functions/strategies/my_strategy.py
2. Fill in the 3 sections marked below: params, indicators, logic
3. Create a config file at functions/configs/my_strategy.yaml
4. Set module in config to: "functions.strategies.my_strategy"
5. Run optimization

YOUR STRATEGY MUST:
- Inherit from BaseStrategy
- Implement generate_signals() method
- Return (action, take_profit, stop_loss) tuple
"""

from functions.base_strategy import BaseStrategy, Action
from typing import Tuple, Optional
import pandas as pd
import numpy as np


class MyStrategy(BaseStrategy):
    """Replace 'MyStrategy' with your strategy name"""
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_idx: int
    ) -> Tuple[Action, Optional[float], Optional[float]]:
        """
        Generate trading signal for current bar
        
        Args:
            data: DataFrame with OHLCV columns + DatetimeIndex
            current_idx: Current bar index in the dataframe
            
        Returns:
            (Action.BUY | Action.SELL | Action.HOLD, take_profit, stop_loss)
        """
        
        # ===== STEP 1: Get your parameters from config =====
        # Access parameters like: self.config['parameter_name']
        period = self.config.get('period', 14)
        threshold = self.config.get('threshold', 30)
        take_profit = self.config.get('take_profit_pips', 20)
        stop_loss = self.config.get('stop_loss_pips', 15)
        
        # ===== STEP 2: Calculate indicators (once per parameter set) =====
        # Use if statement to avoid recalculating every bar
        if 'my_indicator' not in data.columns:
            # Example: RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data['my_indicator'] = 100 - (100 / (1 + rs))
        
        # ===== STEP 3: Implement your trading logic =====
        # Get current values
        current_value = data['my_indicator'].iloc[current_idx]
        
        # Check if we have enough data
        if pd.isna(current_value):
            return Action.HOLD, None, None
        
        # Your entry conditions
        if current_value < threshold:
            return Action.BUY, take_profit, stop_loss
        elif current_value > (100 - threshold):
            return Action.SELL, take_profit, stop_loss
        
        # No signal
        return Action.HOLD, None, None


# ============================================================================
# EXAMPLE: Complete RSI Strategy (Copy & Modify This)
# ============================================================================

class ExampleRSI(BaseStrategy):
    """RSI mean reversion strategy - buy oversold, sell overbought"""
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get params
        rsi_period = self.config.get('rsi_period', 14)
        oversold = self.config.get('oversold', 30)
        overbought = self.config.get('overbought', 70)
        tp = self.config.get('take_profit_pips', 20)
        sl = self.config.get('stop_loss_pips', 15)
        
        # Calculate RSI once
        if 'rsi' not in data.columns:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # Get current RSI
        rsi = data['rsi'].iloc[current_idx]
        if pd.isna(rsi):
            return Action.HOLD, None, None
        
        # Trading logic
        if rsi < oversold:
            return Action.BUY, tp, sl
        elif rsi > overbought:
            return Action.SELL, tp, sl
        
        return Action.HOLD, None, None


# ============================================================================
# EXAMPLE: Bollinger Bands Strategy
# ============================================================================

class ExampleBollinger(BaseStrategy):
    """Buy at lower band, sell at upper band"""
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get params
        period = self.config.get('bb_period', 20)
        std_dev = self.config.get('bb_std_dev', 2.0)
        tp = self.config.get('take_profit_pips', 20)
        sl = self.config.get('stop_loss_pips', 15)
        
        # Calculate bands once
        if 'bb_upper' not in data.columns:
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            data['bb_upper'] = sma + (std * std_dev)
            data['bb_lower'] = sma - (std * std_dev)
        
        # Get current values
        close = data['close'].iloc[current_idx]
        upper = data['bb_upper'].iloc[current_idx]
        lower = data['bb_lower'].iloc[current_idx]
        
        if pd.isna(upper) or pd.isna(lower):
            return Action.HOLD, None, None
        
        # Trading logic: bounce off bands
        if close <= lower:
            return Action.BUY, tp, sl
        elif close >= upper:
            return Action.SELL, tp, sl
        
        return Action.HOLD, None, None


# ============================================================================
# AVAILABLE DATA & METHODS
# ============================================================================
# 
# DATA COLUMNS:
# - data['open'], data['high'], data['low'], data['close'], data['volume']
# - data.index is DatetimeIndex
# 
# ACCESSING CONFIG:
# - self.config['parameter_name']
# - self.config.get('parameter_name', default_value)
# 
# COMMON PATTERNS:
# - Calculate indicators: if 'indicator' not in data.columns: ...
# - Get current value: data['column'].iloc[current_idx]
# - Check for NaN: pd.isna(value) or np.isnan(value)
# - Rolling windows: data['close'].rolling(window=20).mean()
# 
# RETURN VALUES:
# - Action.BUY: Enter long position
# - Action.SELL: Enter short position  
# - Action.HOLD: Do nothing
# - take_profit: Pips to exit with profit (or None)
# - stop_loss: Pips to exit with loss (or None)
#
# ============================================================================
