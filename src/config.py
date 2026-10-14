"""Shared configuration for the auto trading quant pipeline.

Everything comes from environment variables with sane defaults, so you can
run the whole thing locally with docker compose and flip over to AWS by just
changing env vars. No secrets live in this file - AWS creds come from the
normal boto3 chain (instance role, profile, env) outside the code.
"""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    # Small helper so defaults stay in one place and tests can override them.
    return os.getenv(name, default)


@dataclass
class Settings:
    # Kafka / MSK - local broker by default, point at MSK in AWS.
    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092"))
    ticks_topic: str = field(default_factory=lambda: _env("TICKS_TOPIC", "market-ticks"))
    signals_topic: str = field(default_factory=lambda: _env("SIGNALS_TOPIC", "trade-signals"))

    # Symbols we simulate / subscribe to.
    symbols: str = field(default_factory=lambda: _env("SYMBOLS", "AAPL,MSFT,NVDA,TSLA,SPY"))

    # Streaming knobs.
    watermark: str = field(default_factory=lambda: _env("WATERMARK", "2 minutes"))
    window: str = field(default_factory=lambda: _env("WINDOW", "5 minutes"))
    slide: str = field(default_factory=lambda: _env("SLIDE", "1 minute"))
    trigger_seconds: int = int(_env("TRIGGER_SECONDS", "30"))

    # S3 / Glue outputs (purely placeholders until you set them).
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "s3://my-quant-bucket"))
    bronze_prefix: str = "bronze/market-ticks"
    silver_prefix: str = "silver/features"
    gold_prefix: str = "gold/signals"
    checkpoint_root: str = field(default_factory=lambda: _env("CHECKPOINT_ROOT", "./checkpoints"))

    # Glue streaming job name - used for logging only.
    glue_job_name: str = field(default_factory=lambda: _env("GLUE_JOB_NAME", "quant-streaming-job"))

    # Backtest settings.
    initial_capital: float = float(_env("INITIAL_CAPITAL", "100000"))
    max_position_pct: float = float(_env("MAX_POSITION_PCT", "0.1"))
    commission_bps: float = float(_env("COMMISSION_BPS", "2"))

    def symbol_list(self) -> list:
        # Split on commas and drop empties so "AAPL,, MSFT" still works.
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    def s3_path(self, prefix: str) -> str:
        return f"{self.s3_bucket.rstrip('/')}/{prefix}"

    def checkpoint_path(self, stream: str) -> str:
        # One checkpoint dir per stream keeps restarts isolated - sharing a
        # checkpoint between streams is a classic way to corrupt state.
        return f"{self.checkpoint_root.rstrip('/')}/{stream}"

    def kafka_consumer_opts(self) -> dict:
        # Shared consumer options so local runs and Glue runs agree.
        return {
            "kafka.bootstrap.servers": self.kafka_bootstrap,
            "subscribe": self.ticks_topic,
            "startingOffsets": "latest",
        }

    def validate(self) -> None:
        # Fail fast on obviously bad config instead of dying mid-stream.
        assert self.symbol_list(), "SYMBOLS must contain at least one symbol"
        assert 0 < self.max_position_pct <= 1, "MAX_POSITION_PCT must be in (0, 1]"
        assert self.trigger_seconds > 0, "TRIGGER_SECONDS must be positive"

    def summary(self) -> str:
        return (
            f"symbols={self.symbol_list()} kafka={self.kafka_bootstrap} "
            f"topics={self.ticks_topic}->{self.signals_topic} s3={self.s3_bucket}"
        )
