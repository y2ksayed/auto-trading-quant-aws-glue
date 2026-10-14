"""AWS Glue streaming job for quant feature engineering.

Reads raw market ticks from Kafka, computes per-symbol rolling features on a
sliding window, and writes Parquet to S3 in bronze/silver layers. Malformed
rows go to a quarantine sink instead of killing the stream.

Designed to run as an AWS Glue 4.0 streaming job; it also runs on plain
Spark locally — the Glue-specific bits are guarded so they don't break
local runs.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from quantstream.config import Settings, configure_logging

log = logging.getLogger(__name__)

# Explicit schema, never inferSchema in streaming — a single malformed batch
# can otherwise silently change column types and corrupt downstream features.
TICK_SCHEMA = StructType(
    [
        StructField("symbol", StringType(), True),
        StructField("ts", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("bid", DoubleType(), True),
        StructField("ask", DoubleType(), True),
        StructField("size", LongType(), True),
        StructField("exchange", StringType(), True),
    ]
)


def get_spark(settings: Settings) -> SparkSession:
    """Return a Spark session, Glue-aware when running on AWS Glue.

    On Glue, initializes the Glue job context from ``--JOB_NAME``; locally it
    falls back to a plain Spark session.
    """
    try:
        from awsglue.context import GlueContext  # type: ignore
        from awsglue.job import Job  # type: ignore
        from awsglue.utils import getResolvedOptions  # type: ignore

        args = getResolvedOptions(sys.argv, ["JOB_NAME"])
        sc = SparkSession.builder.getOrCreate().sparkContext
        glue_ctx = GlueContext(sc)
        Job(glue_ctx).init(args["JOB_NAME"], args)
        log.info("running as Glue job: %s", args["JOB_NAME"])
        return glue_ctx.spark_session
    except ImportError:
        log.info("awsglue not installed — running as local Spark job")
        return (
            SparkSession.builder.appName(settings.glue_job_name)
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )


def read_ticks(spark: SparkSession, settings: Settings) -> DataFrame:
    """Read raw ticks from Kafka and parse the JSON payload.

    Args:
        spark: Active Spark session.
        settings: Validated pipeline settings.

    Returns:
        DataFrame of parsed ticks plus an ``event_time`` timestamp column.
        Malformed payloads surface as rows with null ``symbol``.
    """
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.ticks_topic)
        .option("startingOffsets", "latest")
        .load()
    )
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), TICK_SCHEMA).alias("tick")
    )
    return parsed.select("tick.*").withColumn(
        "event_time", F.to_timestamp("ts")
    )


def split_quarantine(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split parsed ticks into valid rows and malformed (quarantine) rows."""
    good = df.filter(F.col("symbol").isNotNull())
    bad = df.filter(F.col("symbol").isNull())
    return good, bad


def add_features(df: DataFrame, settings: Settings) -> DataFrame:
    """Compute sliding-window per-symbol features.

    Watermarking keeps the aggregation state bounded: late data beyond the
    watermark is dropped instead of growing state forever.
    """
    windowed = (
        df.withWatermark("event_time", settings.watermark)
        .groupBy(
            "symbol",
            F.window("event_time", settings.window, settings.slide),
        )
        .agg(
            F.avg("price").alias("avg_price"),
            F.stddev("price").alias("volatility"),
            F.count("*").alias("tick_count"),
            F.sum(F.col("price") * F.col("size")).alias("px_size"),
            F.sum("size").alias("total_size"),
        )
    )
    return (
        windowed.withColumn(
            "vwap", F.col("px_size") / F.greatest(F.col("total_size"), F.lit(1))
        )
        .withColumn("window_start", F.col("window.start"))
        .drop("window")
    )


def start_stream(
    df: DataFrame,
    path: str,
    checkpoint: str,
    name: str,
    trigger_seconds: int,
    partition_by: list[str] | None = None,
) -> "StreamingQuery":
    """Start one Parquet streaming query with its own checkpoint.

    Raises:
        RuntimeError: if the query fails to start (bad path, bad options).
    """
    from pyspark.sql.streaming import StreamingQuery

    try:
        writer = (
            df.writeStream.format("parquet")
            .option("path", path)
            .option("checkpointLocation", checkpoint)
            .outputMode("append")
            .trigger(processingTime=f"{trigger_seconds} seconds")
            .queryName(name)
        )
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        query = writer.start()
        log.info("stream started: name=%s path=%s", name, path)
        return query
    except Exception as exc:
        raise RuntimeError(f"failed to start stream {name!r}: {exc}") from exc


def log_quarantine_metrics(bad: DataFrame, settings: Settings) -> None:
    """Log the quarantine rate each trigger so data-quality issues surface."""
    def _log(batch: DataFrame, batch_id: int) -> None:
        count = batch.count()
        if count:
            log.warning("quarantine: %d malformed rows in batch %d", count, batch_id)

    bad.writeStream.foreachBatch(_log).trigger(
        processingTime=f"{settings.trigger_seconds} seconds"
    ).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Glue streaming feature job")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = Settings.from_env()
    log.info("starting streaming job: %s", settings.summary())

    spark = get_spark(settings)
    spark.sparkContext.setLogLevel("WARN")

    ticks = read_ticks(spark, settings)
    good, bad = split_quarantine(ticks)

    queries = []
    try:
        # Bronze: raw ticks, one stream.
        queries.append(
            start_stream(
                good,
                settings.s3_path(settings.bronze_prefix),
                settings.checkpoint_path("bronze"),
                name="bronze",
                trigger_seconds=settings.trigger_seconds,
                partition_by=["symbol"],
            )
        )
        # Quarantine: malformed rows never kill the stream; log their rate.
        log_quarantine_metrics(bad, settings)

        # Silver: engineered features, second stream.
        features = add_features(good, settings)
        queries.append(
            start_stream(
                features,
                settings.s3_path(settings.silver_prefix),
                settings.checkpoint_path("silver"),
                name="silver",
                trigger_seconds=settings.trigger_seconds,
            )
        )
    except RuntimeError:
        log.exception("stream startup failed, stopping any started queries")
        for q in queries:
            q.stop()
        raise

    log.info(
        "streaming %s -> bronze/silver under %s",
        settings.ticks_topic,
        settings.s3_bucket,
    )
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
