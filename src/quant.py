"""Backwards-compatible shim — canonical implementation lives in quantstream/.

Prefer ``from quantstream.<module> import ...`` in new code.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantstream.quant import sma, rsi, momentum_signals, mean_reversion_signals, backtest, summarize, main

__all__ = ['sma', 'rsi', 'momentum_signals', 'mean_reversion_signals', 'backtest', 'summarize', 'main']
