"""Quant signal research: momentum, mean reversion, and a simple backtester.

This is the offline research side - pandas-based, fast to iterate on in the
notebook. The signals here are textbook examples, not production alphas, and
everything is evaluated on synthetic data. Do not trade on this.
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    # Simple moving average; min_periods avoids NaN at the start.
    return series.rolling(window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # Classic RSI: 100 - 100 / (1 + RS), where RS = avg gain / avg loss.
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def momentum_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    # Golden cross style: fast SMA above slow SMA = long, else flat.
    out = df.copy()
    out["sma_fast"] = sma(out["price"], fast)
    out["sma_slow"] = sma(out["price"], slow)
    out["signal"] = np.where(out["sma_fast"] > out["sma_slow"], 1, 0)
    return out


def mean_reversion_signals(df: pd.DataFrame, window: int = 20, z_entry: float = 1.5) -> pd.DataFrame:
    # Z-score of price vs rolling mean: short when stretched high, long when
    # stretched low, flat in between. Textbook mean reversion.
    out = df.copy()
    roll_mean = sma(out["price"], window)
    roll_std = out["price"].rolling(window, min_periods=window).std()
    z = (out["price"] - roll_mean) / roll_std.replace(0, np.nan)
    out["zscore"] = z
    out["signal"] = np.where(z < -z_entry, 1, np.where(z > z_entry, -1, 0))
    return out


def backtest(df: pd.DataFrame, initial_capital: float = 100_000,
             max_position_pct: float = 0.1, commission_bps: float = 2.0) -> dict:
    # Simple vectorized backtest: each bar we hold `signal` fraction of a
    # max-sized position. Commissions charged on position changes.
    prices = df["price"].astype(float)
    rets = prices.pct_change().fillna(0)
    signal = df["signal"].astype(float)

    position = signal * max_position_pct
    trades = position.diff().abs().fillna(0)
    commission = trades * commission_bps / 10_000

    strat_rets = position.shift(1).fillna(0) * rets - commission
    equity = initial_capital * (1 + strat_rets).cumprod()

    total_ret = equity.iloc[-1] / initial_capital - 1
    # Sharpe on per-bar returns, annualized crudely with 252 trading days.
    sharpe = strat_rets.mean() / (strat_rets.std() + 1e-12) * np.sqrt(252)
    max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
    n_trades = int((trades > 0).sum())

    return {
        "total_return": round(float(total_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "n_trades": n_trades,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def summarize(df: pd.DataFrame, name: str, **bt_kwargs) -> dict:
    # Convenience wrapper: run the backtest and print a one-line summary.
    res = backtest(df, **bt_kwargs)
    print(f"[{name}] return={res['total_return']:.2%} sharpe={res['sharpe']:.2f} "
          f"max_dd={res['max_drawdown']:.2%} trades={res['n_trades']}")
    return res
