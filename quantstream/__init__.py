"""quantstream — real-time quant research and signal pipeline.

Public API:
    Signals:      sma, rsi, momentum_signals, mean_reversion_signals
    Backtesting:  backtest, summarize
    Config:       Settings
    ML:           make_labels, walk_forward_splits, validate_training_data,
                  train_walkforward, load_latest_model
"""

from quantstream.config import Settings, configure_logging
from quantstream.quant import (
    backtest,
    mean_reversion_signals,
    momentum_signals,
    rsi,
    sma,
    summarize,
)

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "configure_logging",
    "sma",
    "rsi",
    "momentum_signals",
    "mean_reversion_signals",
    "backtest",
    "summarize",
    "__version__",
]
