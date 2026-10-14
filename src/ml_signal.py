"""Backwards-compatible shim — canonical implementation lives in quantstream/.

Prefer ``from quantstream.<module> import ...`` in new code.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantstream.ml_signal import FEATURE_COLS, LABEL_COL, GLOBAL_SEED, make_labels, validate_training_data, build_pipeline, tune_model, evaluate_predictions, profit_evaluation, config_hash, train_walkforward, load_latest_model, main

__all__ = ['FEATURE_COLS', 'LABEL_COL', 'GLOBAL_SEED', 'make_labels', 'validate_training_data', 'build_pipeline', 'tune_model', 'evaluate_predictions', 'profit_evaluation', 'config_hash', 'train_walkforward', 'load_latest_model', 'main']
