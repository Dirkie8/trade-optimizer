import MetaTrader5 as mt5

from dotenv import load_dotenv
import os
import csv
from datetime import datetime, timezone
from pathlib import Path

load_dotenv()
LOGIN = int(os.getenv("MT5_LOGIN"))
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")

SYMBOL = "EURUSDm"

# Where to save balance/equity history
OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
BAL_CSV = OUT_DIR / 'account_balance.csv'

def _append_account_info(ai):
    try:
        row = {
            'ts_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'balance': getattr(ai, 'balance', None),
            'equity': getattr(ai, 'equity', None),
            'margin': getattr(ai, 'margin', None),
            'margin_free': getattr(ai, 'margin_free', None),
            'leverage': getattr(ai, 'leverage', None),
        }
    except Exception:
        return
    write_header = not BAL_CSV.exists()
    try:
        with BAL_CSV.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Failed to write account csv: {e}")

# --- INITIALIZE CONNECTION ---
if not mt5.initialize():
    print("Initialize() failed, error code =", mt5.last_error())
    quit()

# Optional explicit login
if not mt5.login(LOGIN, PASSWORD, SERVER):
    print("Login failed, error code =", mt5.last_error())
    mt5.shutdown()
    quit()

print("Connected to account:", mt5.account_info().name)

# Log initial account snapshot
ai = mt5.account_info()
if ai is not None:
    _append_account_info(ai)

# --- CHECK SYMBOL ---
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(f"Symbol {SYMBOL} not found.")
    mt5.shutdown()
    quit()

if not symbol_info.visible:
    print(f"Making {SYMBOL} visible...")
    mt5.symbol_select(SYMBOL, True)

# --- GET TICK AND PLACE ORDER ---
tick = mt5.symbol_info_tick(SYMBOL)
if tick is None:
    print(f"Failed to get tick data for {SYMBOL}.")
    mt5.shutdown()
    quit()

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "deviation": 20,
    "magic": 123456,
    "comment": "Test trade from Python",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print("Trade result:", result)

# Log account snapshot after attempted trade
ai2 = mt5.account_info()
if ai2 is not None:
    _append_account_info(ai2)

mt5.shutdown()
