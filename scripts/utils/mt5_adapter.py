"""MT5 adapter utilities.

Provides thin wrappers around MetaTrader5 package functions plus
retry/backoff logic and bar retrieval aligned with a pandas workflow.
"""
from __future__ import annotations
import time
from typing import Optional, List, Dict

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover
    mt5 = None  # graceful degradation


class MT5Adapter:
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self.initialized = False

    def initialize(self) -> bool:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed. Install with: pip install MetaTrader5 (Windows recommended)")
        if not mt5.initialize():
            return False
        if self.login and self.password and self.server:
            authorized = mt5.login(self.login, password=self.password, server=self.server)
            if not authorized:
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        self.initialized = True
        return True

    def shutdown(self):  # pragma: no cover
        if mt5:
            mt5.shutdown()
        self.initialized = False

    def ensure_symbol(self, symbol: str) -> None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}")

    def fetch_recent_bars(self, symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        tf_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
        }
        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
        if rates is None:
            raise RuntimeError(f"Failed to fetch rates for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date').sort_index()
        return df

    def current_tick(self, symbol: str) -> Dict:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get tick for {symbol}")
        return tick._asdict()

    def send_market_order(self, *, symbol: str, volume: float, order_type: str, price: float, sl: float | None, tp: float | None, slippage_points: int, magic: int, comment: str) -> Dict:
        if order_type == 'BUY':
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            mt5_type = mt5.ORDER_TYPE_SELL
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': mt5_type,
            'price': price,
            'sl': sl,
            'tp': tp,
            'slippage': slippage_points,
            'magic': magic,
            'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result._asdict() if result else {'error': mt5.last_error()}

    def positions_for_symbol(self, symbol: str):  # pragma: no cover
        return mt5.positions_get(symbol=symbol)

