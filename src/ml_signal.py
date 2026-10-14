"""Spark ML signal prediction.

Trains a classifier to predict the direction of the next window's return
from the streaming features (avg_price, volatility, vwap, tick_count).
Labels come from a simple momentum rule on historical windows - this is a
demo label, replace it with your real research target before trusting it.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from config import Settings


def make_labels(df: DataFrame) -> DataFrame:
    # Label = 1 if next window's avg_price went up, else 0. Using lead() keeps
    # the label strictly in the future relative to the features - no leakage.
    from pyspark.sql.window import Window as W
    spec = W.partitionBy("symbol").orderBy("window_start")
    return df.withColumn(
        "label",
        F.when(F.lead("avg_price").over(spec) > F.col("avg_price"), 1).otherwise(0),
    ).na.drop()


def build_pipeline(model) -> Pipeline:
    # Assemble numeric features, scale them, then classify. RandomForest
    # handles non-linear interactions; LogisticRegression is the baseline.
    assembler = VectorAssembler(
        inputCols=["avg_price", "volatility", "vwap", "tick_count"],
        outputCol="raw_features",
    )
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    return Pipeline(stages=[assembler, scaler, model])


def train(df: DataFrame) -> dict:
    labeled = make_labels(df)
    train_df, test_df = labeled.randomSplit([0.8, 0.2], seed=42)

    results = {}
    models = {
        "logreg": LogisticRegression(featuresCol="features", labelCol="label", maxIter=50),
        "rf": RandomForestClassifier(featuresCol="features", labelCol="label",
                                     numTrees=100, seed=42),
    }
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    for name, model in models.items():
        pipe = build_pipeline(model).fit(train_df)
        preds = pipe.transform(test_df)
        auc = evaluator.evaluate(preds)
        results[name] = {"auc": round(float(auc), 4)}
        print(f"[{name}] AUC={auc:.4f}")
    return results


def main() -> None:
    settings = Settings()
    # Validate early so a bad env var fails fast instead of mid-training.
    settings.validate()
    spark = SparkSession.builder.appName("ml-signal-train").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Read the silver features written by the streaming job.
    df = spark.read.parquet(settings.s3_path(settings.silver_prefix))
    print(f"Training on {df.count()} feature rows")
    results = train(df)
    print("Results:", results)
    spark.stop()


if __name__ == "__main__":
    main()
