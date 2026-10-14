"""Tests for the pure-Python quant research code.

No Spark required — everything here runs on pandas/NumPy so the suite stays
fast and can run in CI without a cluster.
"""

import numpy as np
import pandas as pd
import pytest

from quantstream.quant import (
    backtest,
    mean_reversion_signals,
    momentum_signals,
    rsi,
    sma,
    summarize,
)
from quantstream.splits import walk_forward_splits


def _prices(n=200, seed=7, drift=0.0005, vol=0.01):
    """Deterministic random-walk price frame for tests."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"price": prices})


# --- signals ---------------------------------------------------------------

def test_sma_shape_and_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert len(out) == len(s)
    assert out.iloc[:2].isna().all()  # min_periods -> NaN until window fills
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_momentum_signal_shape_and_dtype():
    df = momentum_signals(_prices(), fast=5, slow=20)
    assert set(df["signal"].unique()) <= {0, 1}
    assert {"sma_fast", "sma_slow", "signal"} <= set(df.columns)
    assert len(df) == 200


def test_mean_reversion_signal_bounds():
    df = mean_reversion_signals(_prices(), window=20, z_entry=1.0)
    assert set(df["signal"].unique()) <= {-1, 0, 1}
    assert "zscore" in df.columns


def test_rsi_bounds():
    out = rsi(_prices()["price"], window=14)
    valid = out.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_momentum_rejects_bad_windows():
    with pytest.raises(ValueError):
        momentum_signals(_prices(), fast=30, slow=10)


# --- backtester ------------------------------------------------------------

def test_flat_prices_zero_trades():
    df = pd.DataFrame({"price": [100.0] * 100, "signal": [1] * 100})
    res = backtest(df, commission_bps=0, slippage_bps=0)
    assert res["n_trades"] == 0
    assert res["total_return"] == pytest.approx(0.0, abs=1e-6)


def test_commission_reduces_returns():
    df = momentum_signals(_prices())
    free = backtest(df, commission_bps=0, slippage_bps=0)
    costly = backtest(df, commission_bps=50, slippage_bps=0)
    assert costly["total_return"] < free["total_return"]


def test_slippage_reduces_returns():
    df = momentum_signals(_prices())
    free = backtest(df, commission_bps=0, slippage_bps=0)
    slipped = backtest(df, commission_bps=0, slippage_bps=50)
    assert slipped["total_return"] < free["total_return"]


def test_sharpe_nan_safe_on_constant_series():
    # Zero-variance returns must not produce inf/nan Sharpe.
    df = pd.DataFrame({"price": [100.0] * 50, "signal": [0] * 50})
    res = backtest(df)
    assert np.isfinite(res["sharpe"])


def test_summarize_returns_expected_keys():
    res = summarize(_prices().assign(signal=1), "test", commission_bps=0)
    assert set(res) == {"total_return", "sharpe", "max_drawdown", "n_trades", "final_equity"}


def test_backtest_rejects_missing_signal():
    with pytest.raises(ValueError):
        backtest(pd.DataFrame({"price": [1.0, 2.0]}))


# --- walk-forward splits ---------------------------------------------------

def test_walk_forward_folds_do_not_overlap():
    stamps = list(range(20))
    folds = walk_forward_splits(stamps, n_folds=3)
    assert len(folds) == 3
    for train, val in folds:
        assert not set(train) & set(val), "train and validation overlap"
        assert max(train) < min(val), "validation must be strictly after training"


def test_walk_forward_is_deterministic_and_covers_all():
    stamps = list(range(17))
    a = walk_forward_splits(stamps, n_folds=3)
    b = walk_forward_splits(list(reversed(stamps)), n_folds=3)
    assert a == b  # input order doesn't matter
    seen = set()
    for train, val in a:
        seen |= set(train) | set(val)
    assert seen == set(stamps)


def test_walk_forward_rejects_too_few_timestamps():
    with pytest.raises(ValueError):
        walk_forward_splits([1, 2], n_folds=3)


# --- sample data -----------------------------------------------------------

def test_sample_ticks_csv():
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ticks.csv")
    df = pd.read_csv(path)
    assert len(df) > 0
    assert {"ts", "symbol", "price", "bid", "ask", "size", "exchange"} <= set(df.columns)
    assert (df["price"] > 0).all()
