#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Iterable


def equity_reward(
    equity_curve: np.ndarray | Iterable[float],
    k: float = 5.0,
    eps: float = 1e-12,
) -> float:
    """
    Smoothness-aware equity reward.

    equity_curve: array of equity values over time (monotonic time)
    k: drawdown pain multiplier
    eps: small constant to avoid division by zero
    """
    equity = np.asarray(list(equity_curve), dtype=float)
    if equity.size < 2:
        return -np.inf

    # Log return (scale-free)
    start = equity[0] + eps
    end = equity[-1]
    if start <= 0 or end <= 0:
        return -np.inf
    log_return = np.log(end / start)

    # Running max and drawdowns
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    # Fraction of time underwater
    underwater_ratio = float(np.mean(drawdown > eps))

    # Drawdown area normalized by equity std
    dd_area = float(np.sum(drawdown))
    denom = float(np.std(equity) + eps)
    dd_area_norm = dd_area / denom

    reward = log_return * (1.0 - underwater_ratio) * np.exp(-k * dd_area_norm)
    if not np.isfinite(reward):
        return -np.inf
    return float(reward)


def reward_with_constraints(
    equity_curve: np.ndarray | Iterable[float],
    max_dd_frac_of_start: float = 0.2,
    k: float = 5.0,
    eps: float = 1e-12,
) -> float:
    """
    Apply hard constraints (DD and non-positive terminal growth) before equity_reward.

    max_dd_frac_of_start: e.g. 0.2 means reject if max DD exceeds 20% of initial equity.
    """
    equity = np.asarray(list(equity_curve), dtype=float)
    if equity.size < 2:
        return -np.inf

    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    max_dd = float(np.max(drawdown)) if drawdown.size else 0.0
    start = equity[0]
    end = equity[-1]

    if start <= 0:
        return -np.inf
    if max_dd > max_dd_frac_of_start * start:
        return -np.inf
    if end <= start:
        return -np.inf

    return equity_reward(equity, k=k, eps=eps)


def extract_equity_array(backtest_result: Dict[str, Any]) -> np.ndarray:
    """
    Convert a backtest result's equity_curve into a numeric array of equity values.
    Accepts either a list of dicts with key 'equity' or a list/array of numbers.
    """
    eq = backtest_result.get('equity_curve', [])
    if not isinstance(eq, (list, tuple, np.ndarray)):
        return np.asarray([], dtype=float)
    if len(eq) == 0:
        return np.asarray([], dtype=float)
    first = eq[0]
    if isinstance(first, dict) and 'equity' in first:
        return np.asarray([float(p.get('equity', 0.0)) for p in eq], dtype=float)
    # assume numeric already
    try:
        return np.asarray(eq, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def aggregate_rewards(rewards: List[float], mode: str = 'mean', penalty_lambda: float = 0.0) -> float:
    """
    Aggregate fold rewards.
    - mode 'mean': simple average
    - mode 'median': median
    - mode 'mean_var_penalty': mean - penalty_lambda * std
    """
    vals = np.asarray([r for r in rewards if np.isfinite(r)], dtype=float)
    if vals.size == 0:
        return -np.inf
    if mode == 'median':
        return float(np.median(vals))
    if mode == 'mean_var_penalty':
        return float(np.mean(vals) - penalty_lambda * np.std(vals))
    return float(np.mean(vals))


def _max_drawdown_fraction(equity: np.ndarray, eps: float = 1e-12) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    denom = np.where(running_max > 0, running_max, running_max + eps)
    dd_frac = (running_max - equity) / denom
    try:
        return float(np.max(dd_frac))
    except Exception:
        return 0.0


def legacy_reward(
    trades_detail: List[Dict[str, Any]] | None,
    equity_curve: np.ndarray | Iterable[float] | List[Any],
    metrics: Dict[str, Any] | None,
    reward_type: str = 'balanced',
) -> float:
    """
    Legacy reward options compatibility:
    - 'balanced': chunked Sharpe stability minus drawdown penalty
    - 'consistency': mean/std of chunk PnL sums
    - 'sharpe' | 'sortino' | 'calmar': from metrics dict
    """
    metrics = metrics or {}
    eq = np.asarray(list(equity_curve), dtype=float) if not isinstance(equity_curve, np.ndarray) else equity_curve

    if reward_type == 'sharpe':
        return float(metrics.get('sharpe', -np.inf))
    if reward_type == 'sortino':
        return float(metrics.get('sortino', -np.inf))
    if reward_type == 'calmar':
        return float(metrics.get('calmar', -np.inf))

    # Extract pnl from trades_detail if available
    pnl: List[float] = []
    if isinstance(trades_detail, list) and trades_detail:
        for t in trades_detail:
            try:
                v = t.get('pnl') if isinstance(t, dict) else None
                if v is not None:
                    pnl.append(float(v))
            except Exception:
                continue
    if len(pnl) < 5:
        return -np.inf
    arr = np.asarray(pnl, dtype=float)
    if np.all(arr == 0):
        return -np.inf

    chunks = np.array_split(arr, min(5, len(arr)))
    if reward_type == 'consistency':
        sums = [float(np.sum(c)) for c in chunks if len(c) > 0]
        if not sums:
            return -np.inf
        m = float(np.mean(sums))
        s = float(np.std(sums))
        return m / (s + 1e-6) if s > 0 else m

    # 'balanced'
    sharpes: List[float] = []
    for c in chunks:
        if len(c) > 0:
            s = float(np.std(c))
            if s > 0:
                sharpes.append(float(np.mean(c)) / s)
    if not sharpes:
        return -np.inf
    sharpes_np = np.asarray(sharpes, dtype=float)
    mean_sharpe = float(np.mean(sharpes_np))
    sharpe_std = float(np.std(sharpes_np))
    stability = 1.0 - min(sharpe_std, 1.0)
    dd_frac = _max_drawdown_fraction(eq)
    reward = mean_sharpe * stability - 0.2 * dd_frac
    return float(reward)
