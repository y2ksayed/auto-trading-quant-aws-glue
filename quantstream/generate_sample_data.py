"""Generate a small, reproducible sample of synthetic tick data.

Writes ``data/sample_ticks.csv`` with a fixed random seed so the file is
identical on every run. The sample lets anyone exercise the streaming schema,
the backtester, and the notebook without running Kafka or the producer.

Schema (one row per tick):
    ts       ISO-8601 timestamp (UTC)
    symbol   ticker symbol
    price    last trade price (> 0)
    bid/ask  best bid/ask around the price
    size     tick size in shares
    exchange venue
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

SYMBOLS = ["AAPL", "MSFT", "NVDA"]
EXCHANGES = ["NASDAQ", "NYSE", "ARCA"]
SEED = 42


def generate_ticks(rows: int, seed: int = SEED) -> list[dict]:
    """Generate ``rows`` synthetic ticks with a fixed seed (reproducible)."""
    rng = random.Random(seed)
    prices = {s: rng.uniform(120, 380) for s in SYMBOLS}
    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    ticks = []
    for i in range(rows):
        symbol = SYMBOLS[i % len(SYMBOLS)]
        # Gentle random walk per symbol so prices trend realistically.
        prices[symbol] = max(1.0, prices[symbol] * (1 + rng.gauss(0, 0.002)))
        price = round(prices[symbol], 2)
        spread = round(price * rng.uniform(1, 6) / 10_000, 2)
        ticks.append(
            {
                "ts": (start + timedelta(seconds=i * 3)).isoformat(),
                "symbol": symbol,
                "price": price,
                "bid": round(price - spread / 2, 2),
                "ask": round(price + spread / 2, 2),
                "size": rng.randint(100, 5000),
                "exchange": rng.choice(EXCHANGES),
            }
        )
    return ticks


def write_csv(ticks: list[dict], out_path: str) -> None:
    """Write ticks to CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ts", "symbol", "price", "bid", "ask", "size", "exchange"]
        )
        writer.writeheader()
        writer.writerows(ticks)
    log.info("wrote %d ticks to %s (seed=%d)", len(ticks), out_path, SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample tick data")
    parser.add_argument("--rows", type=int, default=500, help="number of ticks")
    parser.add_argument("--out", default="data/sample_ticks.csv", help="output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.rows <= 0:
        raise ValueError(f"--rows must be positive, got {args.rows}")
    write_csv(generate_ticks(args.rows), args.out)


if __name__ == "__main__":
    main()
