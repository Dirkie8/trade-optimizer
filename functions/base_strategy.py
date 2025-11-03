import pandas as pd
from typing import Tuple, Optional, Literal, Dict

Action = Literal["HOLD", "BUY", "SELL"]


class BaseStrategy:
    """Base class for trading strategies.

    Contract:
    - Inputs:
        - data: pandas DataFrame with columns [Open, High, Low, Close, Volume] and DateTimeIndex
        - config: dict of parameters for the strategy
    - Output:
        - (action, stop_loss_pips, take_profit_pips)
            - action in {"HOLD", "BUY", "SELL"}
            - stop_loss_pips, take_profit_pips are positive numbers (pips)
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data index must be a DatetimeIndex")
        self.data = data.sort_index()
        self.config = config

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        """Return a trading signal for the latest bar.

        Should return (action, stop_loss_pips, take_profit_pips).
        Implementations MUST always provide both SL and TP pips when action is BUY or SELL.
        """
        raise NotImplementedError("Strategy must implement generate_signals()")
