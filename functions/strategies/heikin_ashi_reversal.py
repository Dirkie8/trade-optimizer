"""
Heikin Ashi Reversal Strategy

Detects HA candle color change with simple confirmation.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha['HA_Close'].iloc[i-1]) / 2)
    ha['HA_Open'] = pd.Series(ha_open, index=df.index)
    ha['HA_High'] = pd.concat([df['High'], ha['HA_Open'], ha['HA_Close']], axis=1).max(axis=1)
    ha['HA_Low'] = pd.concat([df['Low'], ha['HA_Open'], ha['HA_Close']], axis=1).min(axis=1)
    return ha


class HeikinAshiReversal(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        ema_filter = int(cfg.get('ema_filter', 50))
        tp_pips = float(cfg['take_profit_pips'])
        sl_pips = float(cfg['stop_loss_pips'])

        df = self.data
        if len(df) < ema_filter + 3:
            return 'HOLD', None, None

        ha = _heikin_ashi(df)
        ema = df['Close'].ewm(span=ema_filter, adjust=False).mean()

        ha_prev_bull = ha['HA_Close'].iloc[-2] > ha['HA_Open'].iloc[-2]
        ha_prev_bear = ha['HA_Close'].iloc[-2] < ha['HA_Open'].iloc[-2]
        ha_last_bull = ha['HA_Close'].iloc[-1] > ha['HA_Open'].iloc[-1]
        ha_last_bear = ha['HA_Close'].iloc[-1] < ha['HA_Open'].iloc[-1]
        ema_last = ema.iloc[-1]
        cl = df['Close'].iloc[-1]

        if ha_prev_bear and ha_last_bull and cl > ema_last:
            return 'BUY', sl_pips, tp_pips
        if ha_prev_bull and ha_last_bear and cl < ema_last:
            return 'SELL', sl_pips, tp_pips
        return 'HOLD', None, None
