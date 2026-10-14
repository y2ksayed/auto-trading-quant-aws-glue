"""Backwards-compatible shim — canonical implementation lives in quantstream/.

Prefer ``from quantstream.<module> import ...`` in new code.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantstream.market_producer import tick, make_producer, send_with_backoff, run, main

__all__ = ['tick', 'make_producer', 'send_with_backoff', 'run', 'main']
