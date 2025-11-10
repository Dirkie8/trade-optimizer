"""
Ichimoku Cloud Breakout Strategy

Signals when price closes above the cloud (bullish) or below the cloud (bearish),
optionally requiring Tenkan above/below Kijun for confirmation.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int):
    high = df["High"]
    low = df["Low"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b


class IchimokuCloudBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        tenkan = int(cfg.get("tenkan", 9))
        kijun = int(cfg.get("kijun", 26))
        senkou_b = int(cfg.get("senkou_b", 52))
        require_tk_confirm = bool(cfg.get("require_tk_confirm", True))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < max(tenkan, kijun, senkou_b) + kijun + 2:
            return "HOLD", None, None

        tenkan_sen, kijun_sen, sa, sb = _ichimoku(df, tenkan, kijun, senkou_b)
        close = df["Close"]
        sa_last, sb_last = sa.iloc[-1], sb.iloc[-1]
        c_last = close.iloc[-1]
        if any(pd.isna(x) for x in [sa_last, sb_last, c_last]):
            return "HOLD", None, None

        cloud_top = max(sa_last, sb_last)
        cloud_bottom = min(sa_last, sb_last)

        long_ok = c_last > cloud_top
        short_ok = c_last < cloud_bottom

        if require_tk_confirm:
            tk_prev = (tenkan_sen.iloc[-2], kijun_sen.iloc[-2])
            tk_last = (tenkan_sen.iloc[-1], kijun_sen.iloc[-1])
            if any(pd.isna(x) for x in [*tk_prev, *tk_last]):
                return "HOLD", None, None
            long_ok = long_ok and tk_last[0] > tk_last[1]
            short_ok = short_ok and tk_last[0] < tk_last[1]

        if long_ok:
            return "BUY", sl_pips, tp_pips
        if short_ok:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
