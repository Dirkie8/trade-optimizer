import json
import os
from typing import Optional

import pandas as pd


def timeframe_to_pandas_rule(timeframe: str) -> str:
    """Map timeframe like '1m','5m','15m','1h','4h','1d' to pandas offset alias.
    Raises ValueError if unsupported.
    """
    tf = timeframe.lower().strip()
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[tf]


def _timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    tf = timeframe.lower().strip()
    if tf.endswith('m'):
        return int(tf[:-1]) * 60
    elif tf.endswith('h'):
        return int(tf[:-1]) * 3600
    elif tf.endswith('d'):
        return int(tf[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def load_data(symbol: str, timeframe: str, path: str = "data/") -> pd.DataFrame:
    """Load candlestick data from JSON with expected filename pattern.

    The JSON should be a list of objects with keys: Date, Open, High, Low, Close, Volume.
    Date must be parseable by pandas.to_datetime.
    
    Supports both old format (SYMBOL_TIMEFRAME_candles.json) and new format 
    (candle_data_list_frxSYMBOL_SECONDSs.json).
    """
    # Try new format first
    symbol_for_filename = f"frx{symbol.upper()}" if not symbol.lower().startswith('frx') else symbol.upper()
    timeframe_seconds = _timeframe_to_seconds(timeframe)
    new_file_path = f"{path}candle_data_list_{symbol_for_filename}_{timeframe_seconds}s.json"
    
    # Try old format as fallback
    old_file_path = f"{path}{symbol}_{timeframe}_candles.json"
    
    file_path = None
    if os.path.exists(new_file_path):
        file_path = new_file_path
    elif os.path.exists(old_file_path):
        file_path = old_file_path
    else:
        raise FileNotFoundError(f"Could not find data file. Tried:\n  {new_file_path}\n  {old_file_path}")
    
    df = pd.read_json(file_path)
    if "Date" not in df.columns:
        # try raw json load fallback (in case orient changed)
        with open(file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.set_index("Date").sort_index()
    # ensure numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])  # Volume may be 0.0
    return df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV to a new timeframe using standard OHLC rules.
    Assumes df has a DateTimeIndex and columns Open, High, Low, Close, Volume.
    """
    rule = timeframe_to_pandas_rule(timeframe)
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule, label="right", closed="right").apply(ohlc).dropna()
    return out


def pip_size(symbol: str) -> float:
    """Return pip size for a given symbol (simplified forex rules).
    EURUSD, GBPUSD, etc. -> 0.0001, USDJPY -> 0.01.
    """
    sym = symbol.upper()
    if sym.endswith("JPY"):
        return 0.01
    return 0.0001


def pips_to_price(pips: float, symbol: str) -> float:
    return float(pips) * pip_size(symbol)
