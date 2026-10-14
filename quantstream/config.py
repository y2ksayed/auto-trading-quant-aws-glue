"""Central configuration for the auto-trading quant pipeline.

Every setting comes from an environment variable with a sane local default,
so the pipeline runs out of the box with ``docker compose`` and flips to AWS
by changing env vars. No secrets live here — AWS credentials come from the
standard boto3 chain (instance role, profile, env) outside the code.

Use :func:`Settings.from_env` to build settings; it validates immediately and
fails fast with a clear message instead of dying mid-stream on a bad value.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,TSLA,SPY"


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass
class Settings:
    """All pipeline settings. Build via :meth:`from_env`, never directly."""

    kafka_bootstrap: str = "localhost:9092"
    ticks_topic: str = "market-ticks"
    signals_topic: str = "trade-signals"
    symbols: str = DEFAULT_SYMBOLS

    watermark: str = "2 minutes"
    window: str = "5 minutes"
    slide: str = "1 minute"
    trigger_seconds: int = 30

    s3_bucket: str = "s3://my-quant-bucket"
    bronze_prefix: str = "bronze/market-ticks"
    silver_prefix: str = "silver/features"
    gold_prefix: str = "gold/signals"
    checkpoint_root: str = "./checkpoints"

    glue_job_name: str = "quant-streaming-job"

    initial_capital: float = 100_000.0
    max_position_pct: float = 0.1
    commission_bps: float = 2.0
    slippage_bps: float = 1.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the environment and validate them."""
        settings = cls(
            kafka_bootstrap=_str("KAFKA_BOOTSTRAP", "localhost:9092"),
            ticks_topic=_str("TICKS_TOPIC", "market-ticks"),
            signals_topic=_str("SIGNALS_TOPIC", "trade-signals"),
            symbols=_str("SYMBOLS", DEFAULT_SYMBOLS),
            watermark=_str("WATERMARK", "2 minutes"),
            window=_str("WINDOW", "5 minutes"),
            slide=_str("SLIDE", "1 minute"),
            trigger_seconds=_int("TRIGGER_SECONDS", 30),
            s3_bucket=_str("S3_BUCKET", "s3://my-quant-bucket"),
            checkpoint_root=_str("CHECKPOINT_ROOT", "./checkpoints"),
            glue_job_name=_str("GLUE_JOB_NAME", "quant-streaming-job"),
            initial_capital=_float("INITIAL_CAPITAL", 100_000.0),
            max_position_pct=_float("MAX_POSITION_PCT", 0.1),
            commission_bps=_float("COMMISSION_BPS", 2.0),
            slippage_bps=_float("SLIPPAGE_BPS", 1.0),
        )
        settings.validate()
        return settings

    def symbol_list(self) -> list[str]:
        """Parse ``symbols`` into a cleaned, uppercased list."""
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    def s3_path(self, prefix: str) -> str:
        """Join the bucket with a key prefix."""
        return f"{self.s3_bucket.rstrip('/')}/{prefix.strip('/')}"

    def checkpoint_path(self, stream: str) -> str:
        """Checkpoint dir for one stream.

        One directory per stream — sharing a checkpoint between streams is a
        classic way to corrupt streaming state.
        """
        return f"{self.checkpoint_root.rstrip('/')}/{stream}"

    def kafka_consumer_opts(self) -> dict[str, str]:
        """Shared Kafka consumer options so local and Glue runs agree."""
        return {
            "kafka.bootstrap.servers": self.kafka_bootstrap,
            "subscribe": self.ticks_topic,
            "startingOffsets": "latest",
        }

    def validate(self) -> None:
        """Fail fast on invalid settings with a clear message."""
        if not self.symbol_list():
            raise ValueError("SYMBOLS must contain at least one symbol")
        if self.initial_capital <= 0:
            raise ValueError(
                f"INITIAL_CAPITAL must be positive, got {self.initial_capital}"
            )
        if not 0 < self.max_position_pct <= 1:
            raise ValueError(
                "MAX_POSITION_PCT must be in (0, 1], "
                f"got {self.max_position_pct}"
            )
        if self.trigger_seconds <= 0:
            raise ValueError(
                f"TRIGGER_SECONDS must be positive, got {self.trigger_seconds}"
            )
        if self.commission_bps < 0:
            raise ValueError(
                f"COMMISSION_BPS must be >= 0, got {self.commission_bps}"
            )
        if self.slippage_bps < 0:
            raise ValueError(
                f"SLIPPAGE_BPS must be >= 0, got {self.slippage_bps}"
            )
        if not self.ticks_topic.strip():
            raise ValueError("TICKS_TOPIC must not be empty")

    def summary(self) -> str:
        """One-line human-readable summary for startup logs."""
        return (
            f"symbols={self.symbol_list()} kafka={self.kafka_bootstrap} "
            f"topics={self.ticks_topic}->{self.signals_topic} s3={self.s3_bucket}"
        )


def configure_logging(level: str = "INFO") -> None:
    """Set up structured-ish console logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    configure_logging()
    settings = Settings.from_env()
    print(settings.summary())
