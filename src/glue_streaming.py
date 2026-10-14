"""AWS Glue streaming job for quant feature engineering.

Reads raw market ticks from Kafka, computes per-symbol rolling features on a
sliding window, and writes parquet to S3 in bronze/silver layers. Designed to
run as an AWS Glue streaming job, but it also runs locally on Spark - the
Glue-specific bits are just the job init wrapper, guarded so they don't break
local runs.
"""

import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from config import Settings


def get_spark(settings: Settings) -> SparkSession:
    # Try the Glue context first; fall back to plain Spark for local dev.
    try:
        from awsglue.context import GlueContext  # type: ignore
        from awsglue.job import Job  # type: ignore
        from awsglue.utils import getResolvedOptions  # type: ignore

        args = getResolvedOptions(sys.argv, ["JOB_NAME"])
        sc = SparkSession.builder.getOrCreate().sparkContext
        glue_ctx = GlueContext(sc)
        Job(glue_ctx).init(args["JOB_NAME"], args)
        print(f"Running as Glue job: {args['JOB_NAME']}")
        return glue_ctx.spark_session
    except ImportError:
        print("awsglue not installed - running as local Spark job")
        return (
            SparkSession.builder.appName(settings.glue_job_name)
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )


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


def read_ticks(spark: SparkSession, settings: Settings) -> DataFrame:
    # Read from Kafka, parse the JSON payload with an explicit schema.
    # Explicit schema beats inferSchema in streaming - it fails fast on bad
    # data instead of silently drifting.
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.ticks_topic)
        .option("startingOffsets", "latest")
        .load()
    )
    parsed = raw.select(F.from_json(F.col("value").cast("string"), TICK_SCHEMA).alias("t"))
    return parsed.select("t.*").withColumn("event_time", F.to_timestamp("ts"))


def add_features(df: DataFrame, settings: Settings) -> DataFrame:
    # Sliding-window per-symbol features: VWAP proxy, return, volatility,
    # and tick count. Watermark keeps state bounded.
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
    return windowed.withColumn(
        "vwap", F.col("px_size") / F.greatest(F.col("total_size"), F.lit(1))
    ).withColumn("window_start", F.col("window.start")).drop("window")


def main() -> None:
    settings = Settings()
    spark = get_spark(settings)
    spark.sparkContext.setLogLevel("WARN")

    ticks = read_ticks(spark, settings)

    # Bronze: raw ticks to S3, one stream.
    bronze_q = (
        ticks.writeStream.format("parquet")
        .option("path", settings.s3_path(settings.bronze_prefix))
        .option("checkpointLocation", f"{settings.checkpoint_root}/bronze")
        .outputMode("append")
        .trigger(processingTime=f"{settings.trigger_seconds} seconds")
        .start()
    )

    # Silver: engineered features to S3, second stream.
    features = add_features(ticks, settings)
    silver_q = (
        features.writeStream.format("parquet")
        .option("path", settings.s3_path(settings.silver_prefix))
        .option("checkpointLocation", f"{settings.checkpoint_root}/silver")
        .outputMode("append")
        .trigger(processingTime=f"{settings.trigger_seconds} seconds")
        .start()
    )

    print(f"Streaming {settings.ticks_topic} -> {settings.s3_path(settings.silver_prefix)}")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
