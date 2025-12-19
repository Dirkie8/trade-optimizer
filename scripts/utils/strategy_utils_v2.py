#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any


def infer_max_lookback(params: Dict[str, Any], default: int = 250) -> int:
    """Infer max lookback from params by scanning for period/window/lookback keys."""
    candidates = []
    for k, v in params.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if 'period' in kl or 'window' in kl or 'lookback' in kl:
            try:
                candidates.append(int(v))
            except Exception:
                pass
    return (max(candidates) + 50) if candidates else int(default)
