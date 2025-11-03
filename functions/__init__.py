"""Strategy package.

Keep lightweight to avoid import-time errors from optional strategy modules.
Expose only stable core symbols here.
"""

from .base_strategy import BaseStrategy  # re-export for convenience

__all__ = ["BaseStrategy"]
