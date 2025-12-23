#!/usr/bin/env python3
"""
Fetch historical candle data from local HistData cache and convert to desired format.

Examples:
  python scripts/fetch_data.py --symbol EURUSD --timeframe 1m --days 5
  python scripts/fetch_data.py --symbol EURUSD --timeframe 5m --startdate 2025-01-01 --enddate 2025-12-15
  python scripts/fetch_data.py --symbol EURUSD --timeframe 5m --startdate 2025-11-25 --enddate 2025-12-10
  python scripts/fetch_data.py --symbol EURUSD --timeframe 5m --startdate 2025-11-26 --enddate 2025-12-18
"""
import argparse
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd

# Ensure project root is on sys.path so `scripts` package is importable when executed as a file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _log(msg: str):
    """Simple timestamped logging."""
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    print(f"[{ts}] {msg}", flush=True)


def _map_symbol_for_histdata(symbol: str) -> str:
    """Map symbol to HistData format. Strip 'frx' prefix if present."""
    if symbol.lower().startswith('frx'):
        return symbol[3:].upper()
    return symbol.upper()


def _iter_months(start_dt: datetime, end_dt: datetime):
    """Yield (year, month) pairs covering [start_dt, end_dt)."""
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur < end_dt:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def _find_histdata_files(symbol: str, cache_dir: str = "data/raw/histdata") -> List[str]:
    """Find all HistData zip files for a symbol in the cache directory."""
    sym = _map_symbol_for_histdata(symbol)
    found_files = []
    
    # Search in multiple possible locations
    search_paths = [
        os.path.join(cache_dir, sym, "M1"),
        os.path.join(cache_dir, sym),
        cache_dir,
    ]
    
    seen_paths = set()
    for root_path in search_paths:
        if not os.path.isdir(root_path):
            continue
            
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.lower().endswith('.zip'):
                    continue
                    
                filepath = os.path.join(dirpath, filename)
                if filepath in seen_paths:
                    continue
                seen_paths.add(filepath)
                
                # Check if this looks like a HistData file for our symbol
                if sym.lower() in filename.lower():
                    found_files.append(filepath)
    
    return found_files


def _read_histdata_csv(content_bytes: bytes) -> Optional[pd.DataFrame]:
    """Parse HistData CSV bytes to DataFrame with columns: Date, Open, High, Low, Close, Volume."""
    # Try semicolon-delimited format first: Date Time;Open;High;Low;Close;Volume
    for sep in [';', ',']:
        try:
            df = pd.read_csv(io.BytesIO(content_bytes), header=None, sep=sep, engine='python')
            
            if df.shape[1] >= 6:
                # Format: Date, Time, Open, High, Low, Close, Volume
                date_col = df.iloc[:, 0]
                time_col = df.iloc[:, 1]
                o = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                h = pd.to_numeric(df.iloc[:, 3], errors='coerce')
                l = pd.to_numeric(df.iloc[:, 4], errors='coerce')
                c = pd.to_numeric(df.iloc[:, 5], errors='coerce')
                v = pd.to_numeric(df.iloc[:, 6] if df.shape[1] >= 7 else 0, errors='coerce').fillna(0)
                
                # Build datetime
                dt_str = date_col.astype(str).str.replace('.', '-', regex=False) + ' ' + time_col.astype(str)
                dt = pd.to_datetime(dt_str, utc=True, errors='coerce')
                
                if dt.isna().all():
                    # Try single datetime column format
                    dt = pd.to_datetime(df.iloc[:, 0].astype(str).str.replace('.', '-', regex=False), 
                                      utc=True, errors='coerce')
                    o = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                    h = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                    l = pd.to_numeric(df.iloc[:, 3], errors='coerce')
                    c = pd.to_numeric(df.iloc[:, 4], errors='coerce')
                    v = pd.to_numeric(df.iloc[:, 5] if df.shape[1] >= 6 else 0, errors='coerce').fillna(0)
                
                result = pd.DataFrame({
                    'Date': dt,
                    'Open': o,
                    'High': h,
                    'Low': l,
                    'Close': c,
                    'Volume': v
                }).dropna(subset=['Date'])
                
                if not result.empty:
                    return result
                    
        except Exception:
            continue
    
    # Try with header
    try:
        df = pd.read_csv(io.BytesIO(content_bytes))
        cols = {c.lower(): c for c in df.columns}
        
        # Find datetime column
        dt_series = None
        for key in ['datetime', 'date', 'time', 'timestamp']:
            if key in cols:
                dt_series = df[cols[key]]
                break
        
        if dt_series is not None:
            dt = pd.to_datetime(dt_series, utc=True, errors='coerce')
            result = pd.DataFrame({
                'Date': dt,
                'Open': pd.to_numeric(df.get('Open') or df.get('open'), errors='coerce'),
                'High': pd.to_numeric(df.get('High') or df.get('high'), errors='coerce'),
                'Low': pd.to_numeric(df.get('Low') or df.get('low'), errors='coerce'),
                'Close': pd.to_numeric(df.get('Close') or df.get('close'), errors='coerce'),
                'Volume': pd.to_numeric(df.get('Volume') or df.get('volume'), errors='coerce').fillna(0)
            }).dropna(subset=['Date'])
            
            if not result.empty:
                return result
                
    except Exception:
        pass
    
    return None


def _load_histdata_from_cache(symbol: str, start_dt: datetime, end_dt: datetime, 
                             cache_dir: str = "data/raw/histdata") -> pd.DataFrame:
    """Load M1 data from HistData cache files for the given symbol and date range."""
    sym = _map_symbol_for_histdata(symbol)
    
    # Find all available files
    zip_files = _find_histdata_files(symbol, cache_dir)
    
    if not zip_files:
        _log(f"No HistData files found for {sym} in {cache_dir}")
        return pd.DataFrame()
    
    _log(f"Found {len(zip_files)} HistData files for {sym}")
    
    # Track which months we need vs what we found
    needed_months = set(_iter_months(start_dt, end_dt))
    found_months = set()
    missing_months = []
    
    month_frames = []
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = [name for name in zf.namelist() 
                          if name.lower().endswith('.csv') or name.lower().endswith('.txt')]
                
                if not members:
                    _log(f"No CSV/TXT files in {zip_path}")
                    continue
                
                parsed_count = 0
                for member in members:
                    try:
                        with zf.open(member) as f:
                            raw = f.read()
                        
                        df = _read_histdata_csv(raw)
                        if df is None or df.empty:
                            continue
                        
                        # Filter to requested date range
                        df_filtered = df[(df['Date'] >= start_dt) & (df['Date'] < end_dt)]
                        if df_filtered.empty:
                            continue
                        
                        month_frames.append(df_filtered)
                        parsed_count += len(df_filtered)
                        
                        # Track which months this file covered
                        if not df_filtered.empty:
                            first_month = (df_filtered['Date'].min().year, df_filtered['Date'].min().month)
                            last_month = (df_filtered['Date'].max().year, df_filtered['Date'].max().month)
                            found_months.add(first_month)
                            found_months.add(last_month)
                        
                    except Exception as e:
                        _log(f"Error parsing {member} in {zip_path}: {e}")
                        continue
                
                if parsed_count > 0:
                    _log(f"Loaded {parsed_count} rows from {os.path.basename(zip_path)}")
        
        except Exception as e:
            _log(f"Error reading {zip_path}: {e}")
            continue
    
    # Check for missing months
    for needed_year, needed_month in needed_months:
        if (needed_year, needed_month) not in found_months:
            missing_months.append((needed_year, needed_month))
    
    if missing_months:
        missing_str = ', '.join([f"{y}-{m:02d}" for y, m in missing_months])
        _log(f"WARNING: Missing data for months: {missing_str}")
        _log(f"Place HistData monthly M1 zip files for {sym} in: {cache_dir}")
        _log(f"Expected filename patterns: {sym}_M1_YYYYMM.zip or {sym}_M1_YYYY_MM.zip")
    
    if not month_frames:
        _log(f"No data found for {sym} in requested range {start_dt.date()} to {end_dt.date()}")
        return pd.DataFrame()
    
    # Combine and sort
    result = pd.concat(month_frames, axis=0).sort_values('Date').drop_duplicates(subset=['Date'])
    _log(f"Loaded {len(result)} total M1 candles for {sym}")
    
    return result


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


def _resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1m data to the target timeframe."""
    if timeframe == '1m':
        return df
    
    seconds = _timeframe_to_seconds(timeframe)
    
    # Map to pandas frequency string
    freq_map = {
        300: '5min',      # 5m
        900: '15min',     # 15m
        1800: '30min',    # 30m
        3600: '1h',       # 1h
        14400: '4h',      # 4h
        86400: '1D'       # 1d
    }
    
    freq = freq_map.get(seconds)
    if not freq:
        raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")
    
    df_indexed = df.set_index('Date')
    
    # Resample using OHLC rules
    resampled = df_indexed.resample(freq, label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Fetch historical candle data from HistData cache")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., EURUSD)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, help="Number of days back from today")
    parser.add_argument("--startdate", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--enddate", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", default="data/raw/histdata", help="HistData cache directory")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.startdate and args.enddate:
        start_dt = datetime.strptime(args.startdate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.enddate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.days:
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=args.days)
    else:
        parser.error("Either --days or both --startdate and --enddate must be provided")
    
    _log(f"Fetching {args.symbol} {args.timeframe} data from {start_dt.date()} to {end_dt.date()}")
    
    # Load M1 data from cache
    m1_data = _load_histdata_from_cache(args.symbol, start_dt, end_dt, args.cache_dir)
    
    if m1_data.empty:
        _log("ERROR: No data available for the requested symbol and date range")
        _log(f"Check that HistData files are available in: {args.cache_dir}")
        return 1
    
    # Resample to target timeframe
    try:
        resampled_data = _resample_to_timeframe(m1_data, args.timeframe)
    except ValueError as e:
        _log(f"ERROR: {e}")
        return 1
    
    if resampled_data.empty:
        _log("ERROR: No data after resampling to target timeframe")
        return 1
    
    # Convert to the output format
    output_records = []
    for _, row in resampled_data.iterrows():
        output_records.append({
            "Date": row['Date'].strftime("%Y-%m-%d %H:%M:%S+0000"),
            "Open": float(row['Open']),
            "High": float(row['High']), 
            "Low": float(row['Low']),
            "Close": float(row['Close']),
            "Volume": float(row['Volume'])
        })
    
    # Determine output filename
    symbol_for_filename = f"frx{args.symbol.upper()}" if not args.symbol.lower().startswith('frx') else args.symbol.upper()
    timeframe_seconds = _timeframe_to_seconds(args.timeframe)
    output_filename = f"data/candle_data_list_{symbol_for_filename}_{timeframe_seconds}s.json"
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(output_records, f, indent=2)
    
    _log(f"Successfully saved {len(output_records)} candles to {output_filename}")
    _log(f"Date range: {output_records[0]['Date']} to {output_records[-1]['Date']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
