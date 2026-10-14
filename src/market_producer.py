"""Simulated market tick data producer.

Generates random-walk tick data for a set of symbols and publishes it to
Kafka. This is synthetic data - good enough to exercise the streaming and
quant code, but nothing like real market microstructure. Never trade on this.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer

from config import Settings


def make_producer(bootstrap: str) -> KafkaProducer:
    # JSON serializer keeps the payload readable on the topic.
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=5,
    )


def tick(symbol: str, price: float, ts: datetime) -> dict:
    # A simple tick: last price, a spread, and some size. Real venues give
    # you a lot more (book depth, trade flags), but this is enough for the
    # feature engineering downstream.
    spread_bps = random.uniform(1, 8)
    spread = price * spread_bps / 10_000
    return {
        "symbol": symbol,
        "ts": ts.isoformat(),
        "price": round(price, 2),
        "bid": round(price - spread / 2, 2),
        "ask": round(price + spread / 2, 2),
        "size": random.randint(1, 5000),
        "exchange": random.choice(["NASDAQ", "NYSE", "ARCA"]),
    }


def run(settings: Settings, rate: float, duration: int) -> None:
    producer = make_producer(settings.kafka_bootstrap)
    # Random walk starting points per symbol.
    prices = {s: random.uniform(80, 400) for s in settings.symbol_list()}
    deadline = time.time() + duration if duration > 0 else float("inf")
    sent = 0
    try:
        while time.time() < deadline:
            ts = datetime.now(timezone.utc)
            for symbol in settings.symbol_list():
                # Geometric-ish random walk so prices don't go negative.
                drift = random.gauss(0, 0.0008)
                prices[symbol] = max(1.0, prices[symbol] * (1 + drift))
                producer.send(settings.ticks_topic, tick(symbol, prices[symbol], ts))
                sent += 1
            producer.flush()
            time.sleep(rate)
    finally:
        producer.close()
        print(f"Sent {sent} ticks to {settings.ticks_topic}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate market ticks to Kafka")
    parser.add_argument("--rate", type=float, default=0.5, help="seconds between ticks")
    parser.add_argument("--duration", type=int, default=300, help="seconds to run, 0 = forever")
    args = parser.parse_args()

    settings = Settings()
    print(f"Producing ticks: {settings.summary()}")
    run(settings, rate=args.rate, duration=args.duration)


if __name__ == "__main__":
    main()
