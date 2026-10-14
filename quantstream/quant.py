"""Quant signal research: momentum, mean reversion, RSI, and a backtester.

This is the offline research side — pandas-based, fast to iterate on in the
notebook. The signals here are textbook examples, not production alphas, and
everything is evaluated on synthetic data. Do not trade on this.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ANNUALIZATION = 252  # trading days per year, for Sharpe annualization


def _require_price_col(df: pd.DataFrame) -> pd.Series:
    """Validate the input frame and return the price series as float."""
    if df.empty:
        raise ValueError("input DataFrame is empty")
    if "price" not in df.columns:
        raise ValueError("input DataFrame must contain a 'price' column")
    prices = df["price"].astype(float)
    if (prices <= 0).any():
        raise ValueError("prices must be positive")
    return prices


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average; ``min_periods`` avoids NaNs at the start."""
    _positive_int(window, "window")
    return series.rolling(window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic RSI: 100 - 100 / (1 + RS), RS = avg gain / avg loss."""
    _positive_int(window, "window")
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def momentum_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    """SMA-crossover signals: 1 (long) when fast SMA > slow SMA, else 0."""
    _positive_int(fast, "fast")
    _positive_int(slow, "slow")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    prices = _require_price_col(df)
    out = df.copy()
    out["sma_fast"] = sma(prices, fast)
    out["sma_slow"] = sma(prices, slow)
    out["signal"] = np.where(out["sma_fast"] > out["sma_slow"], 1, 0)
    return out


def mean_reversion_signals(
    df: pd.DataFrame, window: int = 20, z_entry: float = 1.5
) -> pd.DataFrame:
    """Z-score mean-reversion signals: -1 short / 1 long / 0 flat.

    Short when price is stretched above the rolling mean, long when stretched
    below, flat in between. Signals are bounded to {-1, 0, 1}.
    """
    _positive_int(window, "window")
    if z_entry <= 0:
        raise ValueError(f"z_entry must be positive, got {z_entry!r}")
    prices = _require_price_col(df)
    out = df.copy()
    roll_mean = sma(prices, window)
    roll_std = prices.rolling(window, min_periods=window).std()
    z = (prices - roll_mean) / roll_std.replace(0, np.nan)
    out["zscore"] = z
    out["signal"] = np.where(z < -z_entry, 1, np.where(z > z_entry, -1, 0))
    return out


def backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000,
    max_position_pct: float = 0.1,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> dict:
    """Vectorized backtest of a signal column.

    Each bar holds ``signal * max_position_pct`` of capital. Commissions and
    slippage (both in basis points) are charged on absolute position changes.
    Strategy returns use the *lagged* position — you only earn the return
    *after* taking the position, which is the key anti-lookahead rule.

    Args:
        df: Frame with ``price`` and ``signal`` columns.
        initial_capital: Starting equity, must be positive.
        max_position_pct: Fraction of capital per unit signal, in (0, 1].
        commission_bps: Commission per trade in basis points, >= 0.
        slippage_bps: Slippage per trade in basis points, >= 0.

    Returns:
        Dict with total_return, sharpe, max_drawdown, n_trades, final_equity.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")
    if not 0 < max_position_pct <= 1:
        raise ValueError(f"max_position_pct must be in (0, 1], got {max_position_pct}")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("commission_bps and slippage_bps must be >= 0")
    prices = _require_price_col(df)
    if "signal" not in df.columns:
        raise ValueError("input DataFrame must contain a 'signal' column")

    rets = prices.pct_change().fillna(0)
    signal = df["signal"].astype(float)

    position = signal * max_position_pct
    # Trade size drives both costs: commissions plus slippage on the fill.
    trades = position.diff().abs().fillna(0)
    costs = trades * (commission_bps + slippage_bps) / 10_000

    strat_rets = position.shift(1).fillna(0) * rets - costs
    equity = initial_capital * (1 + strat_rets).cumprod()

    total_ret = equity.iloc[-1] / initial_capital - 1
    std = strat_rets.std()
    # NaN-safe Sharpe: constant series have zero variance, not infinite edge.
    sharpe = float(strat_rets.mean() / std * np.sqrt(ANNUALIZATION)) if std > 0 else 0.0
    max_dd = float(((equity.cummax() - equity) / equity.cummax()).max())
    n_trades = int((trades > 0).sum())

    return {
        "total_return": round(float(total_ret), 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def summarize(df: pd.DataFrame, name: str, **bt_kwargs) -> dict:
    """Run the backtest and log a one-line summary; return the metrics dict."""
    res = backtest(df, **bt_kwargs)
    log.info(
        "[%s] return=%.2f%% sharpe=%.2f max_dd=%.2f%% trades=%d equity=%.2f",
        name,
        res["total_return"] * 100,
        res["sharpe"],
        res["max_drawdown"] * 100,
        res["n_trades"],
        res["final_equity"],
    )
    return res


def main() -> None:
    """CLI: backtest textbook signals on the sample tick CSV."""
    parser = argparse.ArgumentParser(description="Backtest signals on sample data")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "sample_ticks.csv",
        ),
        help="tick CSV with a price column",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    df = pd.read_csv(args.csv)
    log.info("loaded %d ticks from %s", len(df), args.csv)
    # Backtest per symbol (ticks interleave symbols, so a single pooled
    # series would be meaningless) and report the mean across symbols.
    results = []
    for symbol, frame in df.groupby("symbol"):
        res = summarize(momentum_signals(frame), f"momentum/{symbol}")
        results.append(res["total_return"])
    log.info(
        "mean total return across %d symbols: %.2f%%",
        len(results),
        float(np.mean(results)) * 100,
    )


if __name__ == "__main__":
    main()
