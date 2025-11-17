import MetaTrader5 as mt5

# Connect to terminal
if not mt5.initialize():
    print("Initialize() failed, error code =", mt5.last_error())
else:
    print("Connected to MT5:", mt5.version())

# Optional: login to a specific account
# mt5.login(login=YOUR_LOGIN, password='YOUR_PASSWORD', server='Exness-MT5Real7')
