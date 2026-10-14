"""Backwards-compatible shim — canonical implementation lives in quantstream/.

Prefer ``from quantstream.<module> import ...`` in new code.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantstream.generate_sample_data import generate_ticks, write_csv, main, SYMBOLS, SEED

__all__ = ['generate_ticks', 'write_csv', 'main', 'SYMBOLS', 'SEED']
