"""Backwards-compatible shim — canonical implementation lives in quantstream/.

Prefer ``from quantstream.<module> import ...`` in new code.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantstream.glue_streaming import TICK_SCHEMA, get_spark, read_ticks, split_quarantine, add_features, start_stream, log_quarantine_metrics, main

__all__ = ['TICK_SCHEMA', 'get_spark', 'read_ticks', 'split_quarantine', 'add_features', 'start_stream', 'log_quarantine_metrics', 'main']
