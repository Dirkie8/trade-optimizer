"""
STRATEGY TEMPLATE
=================

This is a template for creating new trading strategies. Follow the instructions
below to implement your strategy logic.

QUICK START:
1. Copy this file to functions/strategies/my_strategy.py
2. Rename the class below to match your strategy (e.g., RSIStrategy, BollingerStrategy)
3. Implement your signal generation logic in the generate_signals() method
4. Create a config file in functions/configs/my_strategy.yaml
5. Set module in config to: "functions.strategies.my_strategy"
6. Run optimization!

STRATEGY CONTRACT:
- Your class MUST inherit from BaseStrategy
- Your class MUST implement generate_signals() method
- generate_signals() MUST return: (Action, Optional[float], Optional[float])
  where Action is "BUY", "SELL", or "HOLD"
  and the two floats are stop_loss_pips and take_profit_pips (or None for HOLD)

AVAILABLE DATA:
- self.data: pandas DataFrame with columns: Date, Open, High, Low, Close, Volume
- self.data.index: DatetimeIndex (already sorted)
- self.config: dict with your strategy parameters from the YAML config
- self.symbol: string like "frxEURUSD"

TIPS:
- Use self.data.copy() if you need to add indicator columns
- Always check for sufficient data before calculating indicators
- Return "HOLD", None, None if conditions aren't met
- Use pandas .rolling(), .shift(), etc. for indicators
- Access latest candle with: self.data.iloc[-1]
- Access previous candle with: self.data.iloc[-2]
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class MyStrategyTemplate(BaseStrategy):
    """
    [REPLACE THIS WITH YOUR STRATEGY DESCRIPTION]
    
    Example: RSI Mean Reversion Strategy
    - Buys when RSI crosses below oversold threshold
    - Sells when RSI crosses above overbought threshold
    - Uses fixed pip-based stop loss and take profit
    
    Required parameters (from config):
    - parameter1: Description of parameter1
    - parameter2: Description of parameter2
    - take_profit_pips: Take profit in pips
    - stop_loss_pips: Stop loss in pips
    """

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        """
        Generate trading signals based on your strategy logic.
        
        This method is called on each candle during backtesting.
        
        Returns:
            Tuple of (action, stop_loss_pips, take_profit_pips) where:
            - action: "BUY", "SELL", or "HOLD"
            - stop_loss_pips: Stop loss in pips (or None if HOLD)
            - take_profit_pips: Take profit in pips (or None if HOLD)
        """
        
        # ========================================================================
        # STEP 1: EXTRACT PARAMETERS FROM CONFIG
        # ========================================================================
        # These come from your YAML config file under 'parameters'
        # The optimizer will test different combinations of these values
        
        # Example parameters - replace with your own:
        # rsi_period = int(self.config["rsi_period"])
        # oversold = float(self.config["oversold"])
        # overbought = float(self.config["overbought"])
        
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        # ========================================================================
        # STEP 2: VALIDATE PARAMETERS (OPTIONAL BUT RECOMMENDED)
        # ========================================================================
        # Add any parameter validation or constraints here
        # Example:
        # if rsi_period <= 0:
        #     raise ValueError("RSI period must be positive")
        # if oversold >= overbought:
        #     return "HOLD", None, None
        
        # ========================================================================
        # STEP 3: CHECK IF WE HAVE ENOUGH DATA
        # ========================================================================
        # Make sure we have enough candles to calculate your indicators
        # Example: if you need 200 candles for a moving average:
        # min_required_candles = 200
        # if len(self.data) < min_required_candles:
        #     return "HOLD", None, None
        
        # ========================================================================
        # STEP 4: CALCULATE INDICATORS
        # ========================================================================
        # Use pandas to calculate your technical indicators
        # Work on a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Example: Calculate RSI
        # delta = df["Close"].diff()
        # gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        # loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        # rs = gain / loss
        # df["rsi"] = 100 - (100 / (1 + rs))
        
        # Example: Calculate Moving Averages
        # df["sma_20"] = df["Close"].rolling(window=20).mean()
        # df["sma_50"] = df["Close"].rolling(window=50).mean()
        
        # Example: Calculate Bollinger Bands
        # df["bb_middle"] = df["Close"].rolling(window=20).mean()
        # df["bb_std"] = df["Close"].rolling(window=20).std()
        # df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
        # df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
        
        # ========================================================================
        # STEP 5: GET CURRENT AND PREVIOUS VALUES
        # ========================================================================
        # Access the most recent candles for your logic
        last = df.iloc[-1]      # Current (most recent) candle
        prev = df.iloc[-2]      # Previous candle
        
        # Check if indicators are valid (not NaN)
        # if pd.isna(last["rsi"]) or pd.isna(prev["rsi"]):
        #     return "HOLD", None, None
        
        # ========================================================================
        # STEP 6: IMPLEMENT YOUR TRADING LOGIC
        # ========================================================================
        # Define your buy and sell conditions
        
        # Example: RSI Strategy
        # buy_condition = (prev["rsi"] >= oversold and last["rsi"] < oversold)
        # sell_condition = (prev["rsi"] <= overbought and last["rsi"] > overbought)
        
        # Example: Moving Average Crossover
        # buy_condition = (prev["sma_20"] < prev["sma_50"] and 
        #                  last["sma_20"] > last["sma_50"])
        # sell_condition = (prev["sma_20"] > prev["sma_50"] and 
        #                   last["sma_20"] < last["sma_50"])
        
        # Example: Bollinger Band Breakout
        # buy_condition = (prev["Close"] <= prev["bb_lower"] and 
        #                  last["Close"] > prev["bb_lower"])
        # sell_condition = (prev["Close"] >= prev["bb_upper"] and 
        #                   last["Close"] < prev["bb_upper"])
        
        # ========================================================================
        # STEP 7: RETURN YOUR SIGNAL
        # ========================================================================
        # Return one of:
        # - ("BUY", sl_pips, tp_pips): Open a long position
        # - ("SELL", sl_pips, tp_pips): Open a short position
        # - ("HOLD", None, None): No action
        
        # REPLACE THIS WITH YOUR LOGIC:
        # if buy_condition:
        #     return "BUY", sl_pips, tp_pips
        # elif sell_condition:
        #     return "SELL", sl_pips, tp_pips
        # else:
        #     return "HOLD", None, None
        
        # Placeholder - always HOLD until you implement your logic
        return "HOLD", None, None


# ============================================================================
# EXAMPLE STRATEGIES FOR REFERENCE
# ============================================================================

class ExampleRSIStrategy(BaseStrategy):
    """
    Example: Simple RSI Mean Reversion Strategy
    
    Buys when RSI drops below oversold level.
    Sells when RSI rises above overbought level.
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get parameters
        rsi_period = int(self.config["rsi_period"])
        oversold = float(self.config["oversold"])
        overbought = float(self.config["overbought"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        # Need enough data
        if len(self.data) < rsi_period + 2:
            return "HOLD", None, None
        
        # Calculate RSI
        df = self.data.copy()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Get current and previous
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN
        if pd.isna(last["rsi"]) or pd.isna(prev["rsi"]):
            return "HOLD", None, None
        
        # Trading logic: crossover detection
        if prev["rsi"] >= oversold and last["rsi"] < oversold:
            return "BUY", sl_pips, tp_pips
        elif prev["rsi"] <= overbought and last["rsi"] > overbought:
            return "SELL", sl_pips, tp_pips
        else:
            return "HOLD", None, None


class ExampleBollingerBandStrategy(BaseStrategy):
    """
    Example: Bollinger Band Bounce Strategy
    
    Buys when price touches lower band and bounces.
    Sells when price touches upper band and reverses.
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get parameters
        bb_period = int(self.config["bb_period"])
        bb_std = float(self.config["bb_std_dev"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        # Need enough data
        if len(self.data) < bb_period + 2:
            return "HOLD", None, None
        
        # Calculate Bollinger Bands
        df = self.data.copy()
        df["bb_middle"] = df["Close"].rolling(window=bb_period).mean()
        df["bb_std"] = df["Close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * df["bb_std"])
        df["bb_lower"] = df["bb_middle"] - (bb_std * df["bb_std"])
        
        # Get current and previous
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN
        if pd.isna(last["bb_upper"]) or pd.isna(last["bb_lower"]):
            return "HOLD", None, None
        
        # Trading logic: bounce from bands
        # Buy when price was below/at lower band and now moves up
        if prev["Close"] <= prev["bb_lower"] and last["Close"] > prev["bb_lower"]:
            return "BUY", sl_pips, tp_pips
        # Sell when price was above/at upper band and now moves down
        elif prev["Close"] >= prev["bb_upper"] and last["Close"] < prev["bb_upper"]:
            return "SELL", sl_pips, tp_pips
        else:
            return "HOLD", None, None


# ============================================================================
# AFTER IMPLEMENTING YOUR STRATEGY:
# ============================================================================
# 1. Add your class to functions/strategies.py:
#    - Open functions/strategies.py
#    - Copy your class definition
#    - Paste it at the bottom of that file
#    - Or import it: from ._strategy_template import MyStrategyTemplate
#
# 2. Create a config file in functions/configs/:
#    - Copy functions/configs/_template_new_strategy.yaml
#    - Rename it to match your strategy (e.g., my_strategy.yaml)
#    - Update the class name and parameters
#
# 3. Run optimization:
#    python scripts/optimize_strategy.py \
#        --strategy_config functions/configs/my_strategy.yaml \
#        --main_config configs/main_config.yaml \
#        --method grid \
#        --n_jobs 4
#
# 4. View results in:
#    results/YourStrategyName/optimizations/
#    results/YourStrategyName/evaluations/
# ============================================================================
