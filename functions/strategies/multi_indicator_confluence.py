"""
Multi-Indicator Confluence Strategy

Complex strategy combining RSI, MACD, and Bollinger Bands for confluence signals.
Requires agreement from multiple indicators before taking positions.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class MultiIndicatorConfluence(BaseStrategy):
    """Multi-indicator confluence strategy.
    
    Logic:
    - Calculate RSI, MACD, and Bollinger Bands
    - Buy only when multiple indicators agree on bullish signal
    - Sell only when multiple indicators agree on bearish signal
    - Requires minimum number of confirming indicators (confluence_required)
    
    Bullish signals:
    - RSI below oversold threshold
    - MACD line above signal line
    - Price below lower Bollinger Band
    
    Bearish signals:
    - RSI above overbought threshold
    - MACD line below signal line
    - Price above upper Bollinger Band
    
    Parameters:
    - rsi_period: RSI calculation period
    - rsi_oversold: RSI oversold level
    - rsi_overbought: RSI overbought level
    - macd_fast: MACD fast EMA period
    - macd_slow: MACD slow EMA period
    - macd_signal: MACD signal line period
    - bb_period: Bollinger Bands period
    - bb_std_dev: Bollinger Bands standard deviation multiplier
    - confluence_required: Minimum indicators that must agree (1-3)
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get parameters
        rsi_period = int(self.config["rsi_period"])
        rsi_oversold = float(self.config["rsi_oversold"])
        rsi_overbought = float(self.config["rsi_overbought"])
        macd_fast = int(self.config["macd_fast"])
        macd_slow = int(self.config["macd_slow"])
        macd_signal = int(self.config["macd_signal"])
        bb_period = int(self.config["bb_period"])
        bb_std_dev = float(self.config["bb_std_dev"])
        confluence_required = int(self.config["confluence_required"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        df = self.data.copy()
        
        # Need enough data for all indicators
        min_periods = max(rsi_period, macd_slow + macd_signal, bb_period) + 2
        if len(df) < min_periods:
            return "HOLD", None, None
        
        # === Calculate RSI ===
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # === Calculate MACD ===
        ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["signal_line"] = df["macd_line"].ewm(span=macd_signal, adjust=False).mean()
        
        # === Calculate Bollinger Bands ===
        df["bb_middle"] = df["Close"].rolling(window=bb_period).mean()
        df["bb_std"] = df["Close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * bb_std_dev)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * bb_std_dev)
        
        # Get current values
        current_close = df["Close"].iloc[-1]
        current_rsi = df["rsi"].iloc[-1]
        current_macd = df["macd_line"].iloc[-1]
        current_signal = df["signal_line"].iloc[-1]
        current_bb_upper = df["bb_upper"].iloc[-1]
        current_bb_lower = df["bb_lower"].iloc[-1]
        
        # Check for valid values
        if any(pd.isna(x) for x in [current_rsi, current_macd, current_signal, 
                                     current_bb_upper, current_bb_lower]):
            return "HOLD", None, None
        
        # === Count bullish signals ===
        bullish_signals = 0
        
        # RSI oversold
        if current_rsi < rsi_oversold:
            bullish_signals += 1
        
        # MACD bullish (line above signal)
        if current_macd > current_signal:
            bullish_signals += 1
        
        # Price below lower BB (potential bounce up)
        if current_close < current_bb_lower:
            bullish_signals += 1
        
        # === Count bearish signals ===
        bearish_signals = 0
        
        # RSI overbought
        if current_rsi > rsi_overbought:
            bearish_signals += 1
        
        # MACD bearish (line below signal)
        if current_macd < current_signal:
            bearish_signals += 1
        
        # Price above upper BB (potential drop down)
        if current_close > current_bb_upper:
            bearish_signals += 1
        
        # === Generate signals based on confluence ===
        if bullish_signals >= confluence_required:
            return "BUY", sl_pips, tp_pips
        elif bearish_signals >= confluence_required:
            return "SELL", sl_pips, tp_pips
        
        return "HOLD", None, None
