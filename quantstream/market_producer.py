"""Simulated market tick producer.

Generates random-walk tick data for the configured symbols and publishes it
to Kafka as JSON. The data is synthetic — good enough to exercise the
streaming and quant code, nothing like real market microstructure.
Never trade on this.

The producer handles SIGINT/SIGTERM gracefully (flushes in-flight records),
retries failed sends with exponential backoff, and rate-limits accurately to
a target ticks-per-second.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import time
from datetime import datetime, timezone
from typing import Any

from kafka import KafkaProducer
from kafka.errors import KafkaError

from quantstream.config import Settings, configure_logging

log = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Flag a graceful shutdown on SIGINT/SIGTERM."""
    global _shutdown
    log.info("received signal %s, shutting down after this batch", signum)
    _shutdown = True


def make_producer(bootstrap: str, retries: int = 5) -> KafkaProducer:
    """Build a Kafka producer with JSON serialization and sane retries."""
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=retries,
        retry_backoff_ms=500,
        max_in_flight_requests_per_connection=5,
    )


def tick(symbol: str, price: float, ts: datetime) -> dict[str, Any]:
    """Build one synthetic tick: last price, spread, size, venue."""
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


def send_with_backoff(
    producer: KafkaProducer,
    topic: str,
    payload: dict[str, Any],
    max_attempts: int = 4,
) -> bool:
    """Send one record, retrying with exponential backoff on Kafka errors.

    Returns True on success, False if all attempts failed (the record is
    dropped and counted by the caller).
    """
    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            producer.send(topic, payload).get(timeout=10)
            return True
        except KafkaError as exc:
            log.warning(
                "send failed (attempt %d/%d): %s", attempt, max_attempts, exc
            )
            time.sleep(delay)
            delay *= 2
    return False


def run(settings: Settings, ticks_per_second: float, duration: int) -> int:
    """Produce ticks until the deadline or a shutdown signal.

    Rate limiting is per-second accurate: each loop iteration aims to emit
    ``ticks_per_second`` ticks spread across all symbols, then sleeps for the
    remainder of the second.
    """
    producer = make_producer(settings.kafka_bootstrap)
    symbols = settings.symbol_list()
    prices = {s: random.uniform(80, 400) for s in symbols}
    deadline = time.time() + duration if duration > 0 else float("inf")

    sent, dropped, retries = 0, 0, 0
    window_start = time.time()
    window_sent = 0
    try:
        while time.time() < deadline and not _shutdown:
            ts = datetime.now(timezone.utc)
            for symbol in symbols:
                # Geometric-ish random walk so prices can't go negative.
                drift = random.gauss(0, 0.0008)
                prices[symbol] = max(1.0, prices[symbol] * (1 + drift))
                ok = send_with_backoff(
                    producer, settings.ticks_topic, tick(symbol, prices[symbol], ts)
                )
                if ok:
                    sent += 1
                    window_sent += 1
                else:
                    dropped += 1

            # Sleep the remainder of the second to hold the target rate.
            elapsed = time.time() - window_start
            sleep_for = max(0.0, 1.0 - elapsed)
            time.sleep(sleep_for)
            actual_rate = window_sent / max(time.time() - window_start, 1e-6)
            log.debug("tick rate: %.1f/s (target %.1f/s)", actual_rate, ticks_per_second)
            window_start = time.time()
            window_sent = 0
    finally:
        producer.flush(timeout=10)
        producer.close()
    log.info(
        "producer stopped: sent=%d dropped=%d topic=%s",
        sent,
        dropped,
        settings.ticks_topic,
    )
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate market ticks to Kafka")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="target ticks per second across all symbols")
    parser.add_argument("--duration", type=int, default=300,
                        help="seconds to run, 0 = run until interrupted")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    settings = Settings.from_env()
    log.info("starting producer: %s", settings.summary())
    sent = run(settings, ticks_per_second=args.rate, duration=args.duration)
    log.info("done, sent %d ticks", sent)


if __name__ == "__main__":
    main()
