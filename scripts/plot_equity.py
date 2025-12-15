"""Simple plotter for equity_balance_history.csv

Usage:
    python scripts/plot_equity.py [path/to/equity_balance_history.csv]

If no path is given, it will look for scripts/outputs/equity_balance_history.csv
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT = Path(__file__).parent / 'outputs' / 'equity_balance_history.csv'

def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if 'ts_utc' in df.columns:
        df['ts_utc'] = pd.to_datetime(df['ts_utc'])
        df = df.set_index('ts_utc')
    return df


def plot(df: pd.DataFrame):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10,5))
    if 'balance' in df.columns:
        ax.plot(df.index, df['balance'], label='Balance', marker='o')
    if 'equity' in df.columns:
        ax.plot(df.index, df['equity'], label='Equity', marker='o')
    ax.set_title('MT5 Account Balance & Equity')
    ax.set_xlabel('UTC time')
    ax.set_ylabel('Amount')
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    df = load_csv(p)
    print(df.tail())
    plot(df)
