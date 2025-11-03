#!/usr/bin/env python3
"""
Fetch candle data from Deriv WebSocket and save a candle_data_list_*.pkl

Usage:
  python scripts/fetch_data.py --config configs/data/fetch_eurusd_5m.yaml
  python scripts/fetch_data.py --config configs/data/fetch_eurusd_1h.yaml
Notes:
- If api_token is not provided in the config, we will read DERIV_API_TOKEN from env.
- For unauthenticated public history, Deriv may allow limited ranges; auth recommended.
"""
import argparse
import os
import sys
import pickle
from datetime import datetime, timezone, timedelta
import yaml
import io
import zipfile
import requests
import pandas as pd
def _ts():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)

# (Removed Dukascopy LZMA/http helpers)

try:
    import websocket  # comes from lambdas/requirements.txt
except Exception:
    websocket = None

# Allow running as a script without installing as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.data_handler import DerivHistoricalDataClient


def _month_segments(start_dt: datetime, end_dt: datetime):
    """Yield (seg_start, seg_end) month-aligned segments covering [start_dt, end_dt)."""
    # normalize to first-of-month UTC for iteration
    s = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    # if start inside month, first segment starts at start_dt
    cur_start = start_dt
    # compute first day of next month
    def first_of_next_month(d: datetime) -> datetime:
        y, m = d.year, d.month
        if m == 12:
            return datetime(y + 1, 1, 1, tzinfo=timezone.utc)
        return datetime(y, m + 1, 1, tzinfo=timezone.utc)

    while cur_start < end_dt:
        month_end = first_of_next_month(cur_start)
        seg_end = min(month_end, end_dt)
        yield (cur_start, seg_end)
        cur_start = seg_end


def _map_symbol_for_histdata(symbol: str) -> str:
    """HistData uses symbols like 'EURUSD'. Strip 'frx' prefix if present."""
    if symbol.lower().startswith('frx'):
        return symbol[3:]
    return symbol


def _iter_months(start_dt: datetime, end_dt: datetime):
    """Yield (year, month) pairs covering [start_dt, end_dt)."""
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur < end_dt:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def _histdata_url_candidates(symbol: str, year: int, month: int):
    """Return plausible URL candidates for HistData monthly M1 zip.
    HistData changes paths; try a few common patterns.
    """
    s = symbol.upper()
    yyyy = f"{year:04d}"
    mm = f"{month:02d}"
    # Common filename variants
    fn1 = f"{s}_M1_{yyyy}{mm}.zip"
    fn2 = f"{s}_M1_{yyyy}_{mm}.zip"
    # Likely folder structures in wp-content/uploads (varies by year/month)
    return [
        f"https://www.histdata.com/wp-content/uploads/{yyyy}/{mm}/{fn1}",
        f"https://www.histdata.com/wp-content/uploads/{yyyy}/{fn1}",
        f"https://www.histdata.com/wp-content/uploads/{yyyy}/{mm}/{fn2}",
        f"https://www.histdata.com/wp-content/uploads/{yyyy}/{fn2}",
    ]


def _read_histdata_csv(content_bytes: bytes):
    """Parse HistData CSV bytes to a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    Accepts semicolon/comma separators and various datetime layouts.
    """
    # Try semicolon-delimited DAT_ASCII: Date Time;Open;High;Low;Close;Volume
    for sep in [';', ',']:
        try:
            df = pd.read_csv(io.BytesIO(content_bytes), header=None, sep=sep, engine='python')
            # Try to infer columns
            if df.shape[1] >= 6:
                # Handle [Date, Time, O, H, L, C, Vol?]
                date_col = df.iloc[:, 0]
                time_col = df.iloc[:, 1]
                o = df.iloc[:, 2].astype(float)
                h = df.iloc[:, 3].astype(float)
                l = df.iloc[:, 4].astype(float)
                c = df.iloc[:, 5].astype(float)
                v = df.iloc[:, 6] if df.shape[1] >= 7 else 0
                # Build datetime
                dt = pd.to_datetime(date_col.astype(str).str.replace('.', '-', regex=False) + ' ' + time_col.astype(str), utc=True, errors='coerce')
                if dt.isna().all():
                    # Some files have single datetime in first column
                    dt = pd.to_datetime(df.iloc[:, 0].astype(str).str.replace('.', '-', regex=False), utc=True, errors='coerce')
                    o = df.iloc[:, 1].astype(float)
                    h = df.iloc[:, 2].astype(float)
                    l = df.iloc[:, 3].astype(float)
                    c = df.iloc[:, 4].astype(float)
                    v = df.iloc[:, 5] if df.shape[1] >= 6 else 0
                out = pd.DataFrame({
                    'Date': dt,
                    'Open': o.astype(float), 'High': h.astype(float), 'Low': l.astype(float), 'Close': c.astype(float),
                    'Volume': pd.to_numeric(v, errors='coerce').fillna(0).astype(float)
                }).dropna(subset=['Date'])
                return out
        except Exception:
            continue
    # Fallback: try headered CSV
    try:
        df = pd.read_csv(io.BytesIO(content_bytes))
        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        # Try to find datetime column
        dt_series = None
        for key in ['datetime', 'date', 'time', 'timestamp']:
            if key in cols:
                dt_series = df[cols[key]]
                break
        if dt_series is None:
            return None
        dt = pd.to_datetime(dt_series, utc=True, errors='coerce')
        out = pd.DataFrame({
            'Date': dt,
            'Open': pd.to_numeric(df.get('Open') or df.get('open'), errors='coerce'),
            'High': pd.to_numeric(df.get('High') or df.get('high'), errors='coerce'),
            'Low': pd.to_numeric(df.get('Low') or df.get('low'), errors='coerce'),
            'Close': pd.to_numeric(df.get('Close') or df.get('close'), errors='coerce'),
            'Volume': pd.to_numeric(df.get('Volume') or df.get('volume'), errors='coerce').fillna(0)
        }).dropna(subset=['Date'])
        return out
    except Exception:
        return None


def fetch_via_histdata(symbol, granularity, start_iso, end_iso, cache_dir=None):
    """Fetch candles via HistData monthly M1 files, resample to desired granularity.
    If download fails (site gating), will use any matching zip in cache_dir.
    """
    sym = _map_symbol_for_histdata(symbol)
    start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00')).astimezone(timezone.utc)
    end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00')).astimezone(timezone.utc)
    os.makedirs(cache_dir or 'data/raw/histdata', exist_ok=True)
    month_frames = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Referer': 'https://www.histdata.com/'
    })

    missing_months = []
    requested_months = []
    for y, m in _iter_months(start_dt, end_dt):
        requested_months.append((y, m))
        fetched = False
        content_zip = None
        # Try download into cache
        for url in _histdata_url_candidates(sym, y, m):
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 200 and r.content and len(r.content) > 1000:
                    content_zip = r.content
                    _log(f"HistData: HIT {url}")
                    fetched = True
                    break
                else:
                    pass
            except Exception:
                continue
        # If not fetched, try cache files
        zname1 = f"{sym}_M1_{y:04d}{m:02d}.zip"
        zname2 = f"{sym}_M1_{y:04d}_{m:02d}.zip"
        for zname in (zname1, zname2):
            zpath = os.path.join(cache_dir or 'data/raw/histdata', sym, 'M1', zname)
            if os.path.exists(zpath):
                with open(zpath, 'rb') as f:
                    content_zip = f.read()
                    _log(f"HistData: CACHE {zpath}")
                    fetched = True
                    break
        if not fetched or not content_zip:
            _log(f"HistData: MISS for {sym} {y}-{m:02d}. Provide cached zip under {(cache_dir or 'data/raw/histdata')}/{sym}/M1/{zname1}")
            missing_months.append((y, m))
            continue
        # Extract CSV/TXT members and parse (handle zips with multiple files, including annual packs)
        try:
            with zipfile.ZipFile(io.BytesIO(content_zip)) as zf:
                members = [name for name in zf.namelist() if name.lower().endswith('.csv') or name.lower().endswith('.txt')]
                if not members:
                    _log(f"HistData: No CSV/TXT found in zip for {sym} {y}-{m:02d}")
                    continue
                parsed_any = False
                for member in members:
                    try:
                        with zf.open(member) as f:
                            raw = f.read()
                        df = _read_histdata_csv(raw)
                        if df is None or df.empty:
                            continue
                        # Filter to [start, end)
                        df = df[(df['Date'] >= start_dt) & (df['Date'] < end_dt)]
                        if df.empty:
                            continue
                        month_frames.append(df)
                        parsed_any = True
                    except Exception:
                        continue
                if not parsed_any:
                    _log(f"HistData: No rows in range for {sym} {y}-{m:02d} from available CSV/TXT members")
                    continue
        except Exception as e:
            _log(f"HistData: zip parse error for {sym} {y}-{m:02d}: {e}")
            continue

    if not month_frames:
        # Global cache scan fallback: parse any zip under cache_dir (recursively)
        cache_root_candidates = [
            os.path.join(cache_dir or 'data/raw/histdata', sym, 'M1'),
            os.path.join(cache_dir or 'data/raw/histdata', sym),
            cache_dir or 'data/raw/histdata',
        ]
        seen_paths = set()
        for root in cache_root_candidates:
            if not os.path.isdir(root):
                continue
            for dirpath, _dirnames, filenames in os.walk(root):
                for name in filenames:
                    if not name.lower().endswith('.zip'):
                        continue
                    zpath = os.path.join(dirpath, name)
                    if zpath in seen_paths:
                        continue
                    seen_paths.add(zpath)
                    try:
                        with zipfile.ZipFile(zpath, 'r') as zf:
                            members = [fn for fn in zf.namelist() if fn.lower().endswith('.csv') or fn.lower().endswith('.txt')]
                            if not members:
                                continue
                            parsed_count = 0
                            for member in members:
                                try:
                                    with zf.open(member) as f:
                                        raw = f.read()
                                    df = _read_histdata_csv(raw)
                                    if df is None or df.empty:
                                        continue
                                    df = df[(df['Date'] >= start_dt) & (df['Date'] < end_dt)]
                                    if df.empty:
                                        continue
                                    month_frames.append(df)
                                    parsed_count += len(df)
                                except Exception:
                                    continue
                            if parsed_count:
                                _log(f"HistData: CACHE-SCAN used {name} → {parsed_count} rows in range")
                    except Exception:
                        continue
    if not month_frames:
        if missing_months:
            miss_str = ', '.join([f"{y}-{m:02d}" for y, m in missing_months])
            sample_zip1 = f"{sym}_M1_{missing_months[0][0]:04d}{missing_months[0][1]:02d}.zip"
            sample_zip2 = f"{sym}_M1_{missing_months[0][0]:04d}_{missing_months[0][1]:02d}.zip"
            cache_hint = os.path.join(cache_dir or 'data/raw/histdata', sym, 'M1')
            _log("HistData: Missing months summary:")
            _log(f"  - Months missing: {miss_str}")
            _log(f"  - Place monthly M1 zips in: {cache_hint}")
            _log(f"  - Accepted names: {sample_zip1}, {sample_zip2}, or annual packs like HISTDATA_COM_MT_{sym}_M12011.zip (any *.zip with CSV/TXT inside works)")
        return []
    m1 = pd.concat(month_frames, axis=0).sort_values('Date').drop_duplicates(subset=['Date'])
    # Resample to desired granularity
    if int(granularity) == 60:
        df = m1
    else:
        freq_map = {300: '5T', 900: '15T', 1800: '30T', 3600: '1H', 14400: '4H', 86400: '1D'}
        freq = freq_map.get(int(granularity))
        if not freq:
            raise RuntimeError(f"Unsupported granularity for HistData: {granularity}")
        g = m1.set_index('Date')
        o = g['Open'].resample(freq).first()
        h = g['High'].resample(freq).max()
        l = g['Low'].resample(freq).min()
        c = g['Close'].resample(freq).last()
        v = g['Volume'].resample(freq).sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna().reset_index()
    out = []
    for _, r in df.iterrows():
        dt = r['Date'] if isinstance(r['Date'], datetime) else pd.to_datetime(r['Date'], utc=True).to_pydatetime()
        out.append({
            'Date': dt.isoformat().replace('+00:00', 'Z'),
            'Open': float(r['Open']), 'High': float(r['High']), 'Low': float(r['Low']),
            'Close': float(r['Close']), 'Volume': float(r.get('Volume', 0.0))
        })
    return out


def fetch_via_deriv_segmented(symbol, granularity, start_iso, end_iso, app_id=None, api_token=None, segment_mode='monthly'):
    """Fetch candles via Deriv in segments and merge/dedup."""
    # Parse for segmentation
    start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
    segments = [(start_dt, end_dt)] if segment_mode == 'none' else list(_month_segments(start_dt, end_dt))
    _log(f"Deriv segmented fetch: {len(segments)} segment(s) from {start_iso} to {end_iso}")
    all = []
    seen = set()
    for i, (seg_s, seg_e) in enumerate(segments, 1):
        ss = seg_s.isoformat().replace('+00:00', 'Z')
        ee = seg_e.isoformat().replace('+00:00', 'Z')
        _log(f"Segment {i}/{len(segments)}: {ss} → {ee}")
        client = DerivHistoricalDataClient(symbol, granularity, ss, ee, timezone_in_minutes=0, api_token=api_token)
        if app_id:
            endpoint = "wss://ws.binaryws.com/websockets/v3"
            client.ws_url = f"{endpoint}?app_id={str(app_id)}"
        _times, _prices, candles = client.get_historical_data(timeout=180, max_retries=5, sleep_between_batches=1)
        _log(f"Segment {i}/{len(segments)}: received {len(candles)} candles")
        for c in candles:
            dt = c.get('Date')
            if isinstance(dt, datetime):
                ep = int(dt.timestamp())
                if ep in seen:
                    continue
                seen.add(ep)
                all.append({
                    'Date': dt.isoformat().replace('+00:00', 'Z'),
                    'Open': float(c['Open']), 'High': float(c['High']), 'Low': float(c['Low']),
                    'Close': float(c['Close']), 'Volume': float(c.get('Volume', 0))
                })
    all.sort(key=lambda x: x['Date'])
    return all


def parse_iso8601(v):
    """Accept ISO8601 string, datetime, or numeric epoch and return epoch seconds (int)."""
    # Already epoch seconds
    if isinstance(v, (int, float)):
        return int(v)
    # datetime object (YAML may auto-parse timestamps)
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    # string
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        # Replace 'Z' with '+00:00' for fromisoformat
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    raise TypeError(f"Unsupported datetime value type: {type(v)}")


def to_iso_str(v):
    """Return an ISO8601 string with Z for given epoch/datetime/string value."""
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(int(v), tz=timezone.utc).isoformat().replace('+00:00', 'Z')
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')
    if isinstance(v, str):
        s = v.strip()
        # If numeric string epoch
        if s.isdigit():
            return datetime.fromtimestamp(int(s), tz=timezone.utc).isoformat().replace('+00:00', 'Z')
        # Normalize Z suffix
        if s.endswith('Z'):
            return s
        try:
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat().replace('+00:00', 'Z')
        except Exception:
            return s
    raise TypeError(f"Unsupported datetime value type: {type(v)}")


def main():
    ap = argparse.ArgumentParser(description='Fetch candle data via Deriv API')
    ap.add_argument('--config', default='configs/data/fetch_eurusd_5m.yaml')
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file {args.config} not found")
        return 1
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    symbol = cfg.get('symbol', 'frxEURUSD')
    granularity = int(cfg.get('granularity', 300))
    source = (cfg.get('source') or 'auto').lower()
    segment_mode = (cfg.get('segment') or 'monthly').lower()
    merge_existing = bool(cfg.get('merge_existing', False))
    start = parse_iso8601(cfg.get('start')) if cfg.get('start') else None
    end = parse_iso8601(cfg.get('end')) if cfg.get('end') else None
    outfile = cfg.get('outfile', f"data/candle_data_list_{symbol}_{granularity}s.pkl")
    histdata_cache_dir = cfg.get('histdata_cache_dir') or 'data/raw/histdata'

    api_token = cfg.get('api_token') or os.getenv('DERIV_API_TOKEN')
    app_id = cfg.get('app_id') or os.getenv('DERIV_APP_ID')

    # Use historical client with robust batching/retry logic
    start_iso = to_iso_str(cfg.get('start')) if cfg.get('start') else None
    end_iso = to_iso_str(cfg.get('end')) if cfg.get('end') else None
    if not start_iso or not end_iso:
        print("ERROR: start/end must be provided in the config (ISO8601 or epoch).")
        return 1
    _log(f"Fetch start: source={source}, symbol={symbol}, tf={granularity}s, range={start_iso} → {end_iso}, outfile={outfile}, merge_existing={merge_existing}")

    all_candles = []
    # If merge_existing and outfile exists, load to detect existing coverage
    existing = []
    if merge_existing and os.path.exists(outfile):
        try:
            with open(outfile, 'rb') as f:
                existing = pickle.load(f)
        except Exception:
            existing = []
        if existing:
            try:
                ef = existing[0]['Date']
                el = existing[-1]['Date']
                _log(f"Existing outfile detected. Span: {ef} → {el}. Will backfill missing earlier portion if needed.")
            except Exception:
                _log("Existing outfile detected but could not parse dates; proceeding without merge optimization.")
    try:
        desired_start = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        desired_end = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))

        # Determine missing ranges relative to existing file
        missing_left = None
        missing_right = None
        existing_first_dt = None
        existing_last_dt = None
        if existing:
            try:
                existing_first_dt = datetime.fromisoformat(existing[0]['Date'].replace('Z', '+00:00'))
                existing_last_dt = datetime.fromisoformat(existing[-1]['Date'].replace('Z', '+00:00'))
                if desired_start < existing_first_dt:
                    missing_left = (start_iso, existing_first_dt.isoformat().replace('+00:00', 'Z'))
                if desired_end > existing_last_dt:
                    # Start a hair after the last existing candle to avoid duplication
                    right_start_dt = existing_last_dt + timedelta(seconds=1)
                    missing_right = (right_start_dt.isoformat().replace('+00:00', 'Z'), end_iso)
            except Exception:
                # Fall back to fetching full range if parsing failed
                missing_left = (start_iso, end_iso)
        else:
            # No existing file → fetch full range
            missing_left = (start_iso, end_iso)

        # Provider selection per missing range
        def choose_provider(range_start_dt: datetime):
            if source == 'histdata':
                return 'histdata'
            if source == 'deriv':
                return 'deriv'
            # auto: use histdata if older than ~360 days, else deriv
            now_dt = datetime.now(timezone.utc)
            if (now_dt - range_start_dt).days > 360:
                return 'histdata'
            return 'deriv'

        if source not in ('deriv', 'auto', 'histdata'):
            print(f"ERROR: Unknown source '{source}'. Use one of [auto|deriv|histdata].")
            return 1

        pieces = []
        fetched_left = 0
        fetched_right = 0
        if missing_left is not None:
            ls, le = missing_left
            _log(f"Deriv segmented: fetching left/backfill range {ls} → {le}")
            left_start_dt = datetime.fromisoformat(ls.replace('Z', '+00:00'))
            provider = choose_provider(left_start_dt)
            if provider == 'histdata':
                _log("Provider: HistData (left/backfill)")
                left_part = fetch_via_histdata(symbol, granularity, ls, le, cache_dir=histdata_cache_dir)
            else:
                _log("Provider: Deriv (left/backfill)")
                left_part = fetch_via_deriv_segmented(symbol, granularity, ls, le, app_id=app_id, api_token=api_token, segment_mode=segment_mode)
            fetched_left = len(left_part)
            pieces.extend(left_part)
        else:
            _log("Existing file covers requested left boundary; no backfill needed.")

        if missing_right is not None:
            rs, re = missing_right
            _log(f"Deriv segmented: fetching right/extension range {rs} → {re}")
            right_start_dt = datetime.fromisoformat(rs.replace('Z', '+00:00'))
            provider = choose_provider(right_start_dt)
            if provider == 'histdata':
                _log("Provider: HistData (right/extension)")
                right_part = fetch_via_histdata(symbol, granularity, rs, re, cache_dir=histdata_cache_dir)
            else:
                _log("Provider: Deriv (right/extension)")
                right_part = fetch_via_deriv_segmented(symbol, granularity, rs, re, app_id=app_id, api_token=api_token, segment_mode=segment_mode)
            fetched_right = len(right_part)
            pieces.extend(right_part)

        all_candles = pieces
        if not existing and not pieces:
            # Safety: ensure we fetch if nothing was collected
            provider = choose_provider(desired_start)
            if provider == 'histdata':
                all_candles = fetch_via_histdata(symbol, granularity, start_iso, end_iso, cache_dir=histdata_cache_dir)
                print("Source: HistData")
            else:
                all_candles = fetch_via_deriv_segmented(symbol, granularity, start_iso, end_iso, app_id=app_id, api_token=api_token, segment_mode=segment_mode)
                print("Source: Deriv (segmented)")
        else:
            print("Source: mixed (Deriv/HistData)" if (fetched_left and fetched_right and provider) else "Source: Deriv/HistData")
    except Exception as e:
        import traceback
        print("Fetch failed:", e)
        traceback.print_exc()
        return 1

    # Merge with existing if needed
    if existing:
        combined = []
        seen = set()
        # Add new first (older part), then existing
        for c in (all_candles + existing):
            dt = c.get('Date')
            if isinstance(dt, str):
                key = dt
            else:
                key = datetime.fromisoformat(str(dt).replace('Z', '+00:00')).isoformat()
            if key in seen:
                continue
            seen.add(key)
            combined.append(c)
        combined.sort(key=lambda x: x['Date'])
        all_candles = combined
        _log(f"Merged: now have {len(all_candles)} candles after de-dup and sort")

    # Save
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    # Input already normalized (Deriv path ensures ISO); re-normalize defensively
    normalized = []
    for c in all_candles:
        dt = c.get('Date')
        if isinstance(dt, datetime):
            c = dict(c)
            c['Date'] = dt.isoformat().replace('+00:00', 'Z')
        elif isinstance(dt, str) and not dt.endswith('Z'):
            try:
                _dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                c = dict(c)
                c['Date'] = _dt.isoformat().replace('+00:00', 'Z')
            except Exception:
                pass
        normalized.append(c)
    with open(outfile, 'wb') as f:
        pickle.dump(normalized, f)
    print(f"Saved {len(normalized)} candles to {outfile}")
    # Coverage summary
    try:
        req_first = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        req_last = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
        have_first = datetime.fromisoformat(normalized[0]['Date'].replace('Z', '+00:00')) if normalized else None
        have_last = datetime.fromisoformat(normalized[-1]['Date'].replace('Z', '+00:00')) if normalized else None

        # Compute coverage inside requested range only
        within = [c for c in normalized if req_first <= datetime.fromisoformat(c['Date'].replace('Z', '+00:00')) < req_last]
        approx_expected = int((req_last - req_first).total_seconds() // granularity)

        print("\nCoverage summary:")
        print(f"- Requested: {req_first.isoformat().replace('+00:00','Z')} → {req_last.isoformat().replace('+00:00','Z')} (~{(req_last-req_first).total_seconds()/86400.0:.1f} days)")
        if 'existing' in locals() and existing:
            print(f"- Pre-existing file span: {existing_first_dt.isoformat().replace('+00:00','Z')} → {existing_last_dt.isoformat().replace('+00:00','Z')} ({len(existing)} candles)")
        print(f"- Fetched this run: left/backfill={fetched_left}, right/extension={fetched_right}")
        if have_first and have_last:
            print(f"- Final file span: {have_first.isoformat().replace('+00:00','Z')} → {have_last.isoformat().replace('+00:00','Z')} ({len(normalized)} candles)")
        print(f"- Candles within requested range: {len(within)} / ~{approx_expected} expected")
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
