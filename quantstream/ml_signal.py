"""Quant-firm-grade ML signal system.

Trains and compares classifiers that predict the direction of the next
window's return from streaming features. Labels come from a simple momentum
rule on historical windows — a demo label; replace it with a real research
target before trusting it.

What makes this production-grade:

- Walk-forward validation: no random train/test splits on time-series data.
  Every validation fold is strictly in the future relative to its training
  fold, mimicking how models are actually retrained and deployed.
- Per-model hyperparameter tuning with seeded ``TrainValidationSplit``.
- Experiment tracking: every run writes ``runs/<timestamp>/run.json`` with
  params, features, data window, seed, config hash, and per-fold metrics.
- Model registry: the winner lands in
  ``models/registry/<model_name>/v<date>/`` with ``metadata.json``; a
  ``LATEST`` pointer always resolves the current champion.
- Profit-based evaluation: each model's predictions are fed through the
  cost-aware backtester, so selection considers net P&L, not just AUC.

All models are Spark MLlib — no sklearn-in-Spark hacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from quantstream.config import Settings, configure_logging
from quantstream.splits import fold_summary, walk_forward_splits

log = logging.getLogger(__name__)

# Single source of truth for model inputs — persisted with each artifact.
FEATURE_COLS = ["avg_price", "volatility", "vwap", "tick_count"]
LABEL_COL = "label"

# One global seed threaded through every randomized step.
GLOBAL_SEED = 42
MIN_TRAIN_ROWS = 100
N_FOLDS = 3

MODEL_SPECS: dict[str, dict] = {
    # Small grids on purpose: tuning must stay fast enough to run on every
    # retrain, not just on a research cluster.
    "logreg": {
        "estimator": lambda: LogisticRegression(
            featuresCol="features", labelCol=LABEL_COL, maxIter=50
        ),
        "grid": {"regParam": [0.01, 0.1], "elasticNetParam": [0.0, 0.5]},
    },
    "random_forest": {
        "estimator": lambda: RandomForestClassifier(
            featuresCol="features", labelCol=LABEL_COL, seed=GLOBAL_SEED
        ),
        "grid": {"numTrees": [50, 100], "maxDepth": [5, 8]},
    },
    "gbt": {
        "estimator": lambda: GBTClassifier(
            featuresCol="features", labelCol=LABEL_COL, seed=GLOBAL_SEED
        ),
        "grid": {"maxIter": [20, 50], "maxDepth": [3, 5]},
    },
    "linear_svc": {
        "estimator": lambda: LinearSVC(
            featuresCol="features", labelCol=LABEL_COL, maxIter=50
        ),
        "grid": {"regParam": [0.01, 0.1]},
    },
}


def make_labels(df: DataFrame) -> DataFrame:
    """Label = 1 if the *next* window's avg_price rose, else 0.

    Using ``lead()`` partitioned by symbol keeps the label strictly in the
    future relative to the features — no lookahead leakage.
    """
    spec = Window.partitionBy("symbol").orderBy("window_start")
    return (
        df.withColumn(
            LABEL_COL,
            F.when(F.lead("avg_price").over(spec) > F.col("avg_price"), 1).otherwise(0),
        )
        .na.drop()
    )


def validate_training_data(df: DataFrame, min_rows: int = MIN_TRAIN_ROWS) -> None:
    """Fail fast on data problems that would silently wreck training.

    Checks: required columns present, no nulls in feature/label columns,
    both label classes present, and enough rows to split into folds.
    """
    required = set(FEATURE_COLS + [LABEL_COL, "window_start", "symbol"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training data missing columns: {sorted(missing)}")

    n = df.count()
    if n < min_rows:
        raise ValueError(f"need at least {min_rows} rows, got {n}")

    null_counts = (
        df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLS + [LABEL_COL]])
        .collect()[0]
        .asDict()
    )
    bad = {c: v for c, v in null_counts.items() if v}
    if bad:
        raise ValueError(f"null values in training columns: {bad}")

    label_counts = {r[LABEL_COL]: r["count"] for r in df.groupBy(LABEL_COL).count().collect()}
    if len(label_counts) < 2:
        raise ValueError(
            f"label column has only one class {label_counts} — "
            "a classifier cannot learn from a single class"
        )
    minority = min(label_counts.values()) / n
    log.info("label balance: %s (minority share %.3f)", label_counts, minority)
    if minority < 0.05:
        log.warning("minority class share %.3f is very low; metrics may be noisy", minority)


def build_pipeline(estimator) -> Pipeline:
    """Assemble numeric features, scale them, then classify."""
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    return Pipeline(stages=[assembler, scaler, estimator])


def _build_param_grid(estimator, grid: dict) -> list:
    """Translate a {param_name: values} dict into a Spark param grid."""
    builder = ParamGridBuilder()
    for name, values in grid.items():
        builder = builder.addGrid(getattr(estimator, name), values)
    return builder.build()


def tune_model(
    name: str,
    spec: dict,
    train_df: DataFrame,
    seed: int = GLOBAL_SEED,
) -> tuple[PipelineModel, dict]:
    """Tune one model with seeded TrainValidationSplit; return best + params."""
    estimator = spec["estimator"]()
    param_maps = _build_param_grid(estimator, spec["grid"])
    tvs = TrainValidationSplit(
        estimator=build_pipeline(estimator),
        estimatorParamMaps=param_maps,
        evaluator=BinaryClassificationEvaluator(
            labelCol=LABEL_COL, metricName="areaUnderROC"
        ),
        trainRatio=0.8,
        seed=seed,
    )
    tvs_model = tvs.fit(train_df)
    best_params = {
        p.name: tvs_model.bestModel.stages[-1].getOrDefault(p)
        for p in sum([list(m.keys()) for m in param_maps], [])
    }
    log.info("[%s] best params: %s", name, best_params)
    return tvs_model.bestModel, best_params


def evaluate_predictions(predictions: DataFrame) -> dict:
    """AUC, accuracy, precision, recall, F1 on a predictions DataFrame."""
    auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderROC"
    ).evaluate(predictions)
    multi = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction")
    return {
        "auc": round(float(auc), 4),
        "accuracy": round(float(multi.evaluate(predictions, {multi.metricName: "accuracy"})), 4),
        "precision": round(float(multi.evaluate(predictions, {multi.metricName: "weightedPrecision"})), 4),
        "recall": round(float(multi.evaluate(predictions, {multi.metricName: "weightedRecall"})), 4),
        "f1": round(float(multi.evaluate(predictions, {multi.metricName: "f1"})), 4),
    }


def profit_evaluation(
    predictions: DataFrame,
    settings: Settings,
) -> dict:
    """Feed each model's predictions through the cost-aware backtester.

    Per symbol, the predicted class becomes the position signal on that
    window's avg_price series. Returns mean net P&L and Sharpe across symbols —
    a model with 52% accuracy but positive net P&L beats a 55% model that
    loses money after costs.
    """
    from quantstream.quant import backtest  # local import: pandas-only, no Spark

    rows = (
        predictions.select("symbol", "window_start", "avg_price", "prediction")
        .orderBy("symbol", "window_start")
        .collect()
    )
    by_symbol: dict[str, list] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], []).append(r)

    results = []
    for symbol, rs in by_symbol.items():
        import pandas as pd

        frame = pd.DataFrame(
            {"price": [r["avg_price"] for r in rs], "signal": [r["prediction"] for r in rs]}
        )
        if len(frame) < 2:
            continue
        metrics = backtest(
            frame,
            initial_capital=settings.initial_capital,
            max_position_pct=settings.max_position_pct,
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
        )
        results.append(metrics)

    if not results:
        return {"mean_net_pnl": 0.0, "mean_sharpe": 0.0, "n_symbols": 0}
    import statistics

    return {
        "mean_net_pnl": round(
            statistics.mean(r["final_equity"] - settings.initial_capital for r in results), 2
        ),
        "mean_sharpe": round(statistics.mean(r["sharpe"] for r in results), 3),
        "n_symbols": len(results),
    }


def config_hash(settings: Settings, seed: int) -> str:
    """Short deterministic hash of the training configuration."""
    payload = json.dumps(
        {
            "features": FEATURE_COLS,
            "label": LABEL_COL,
            "n_folds": N_FOLDS,
            "seed": seed,
            "symbols": settings.symbol_list(),
            "window": settings.window,
            "slide": settings.slide,
            "watermark": settings.watermark,
            "commission_bps": settings.commission_bps,
            "slippage_bps": settings.slippage_bps,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def train_walkforward(
    df: DataFrame,
    settings: Settings,
    models_dir: str = "models",
    runs_dir: str = "runs",
    n_folds: int = N_FOLDS,
    seed: int = GLOBAL_SEED,
) -> dict:
    """Walk-forward train/tune/evaluate all models; register the winner.

    Fold ``i`` trains on all windows up to a cutoff and validates on the next
    block — validation is always in the future relative to training.
    """
    validate_training_data(df)
    labeled = make_labels(df).cache()
    log.info("labeled rows: %d", labeled.count())

    bounds = [
        r[0]
        for r in labeled.select("window_start").distinct().orderBy("window_start").collect()
    ]
    folds = walk_forward_splits(bounds, n_folds=n_folds)
    for s in fold_summary(folds):
        log.info(
            "fold %(fold)d: train_end=%(train_end)s val=%(val_start)s..%(val_end)s "
            "(n_train=%(n_train)d, n_val=%(n_val)d)", s,
        )

    data_start, data_end = str(bounds[0]), str(bounds[-1])
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = config_hash(settings, seed)

    per_model: dict[str, dict] = {}
    for name, spec in MODEL_SPECS.items():
        fold_metrics: list[dict] = []
        best_params: dict = {}
        for fold_i, (train_ts, val_ts) in enumerate(folds):
            train_df = labeled.filter(F.col("window_start").isin(train_ts))
            val_df = labeled.filter(F.col("window_start").isin(val_ts))
            model, best_params = tune_model(name, spec, train_df, seed=seed)
            preds = model.transform(val_df)
            metrics = evaluate_predictions(preds)
            metrics.update(profit_evaluation(preds, settings))
            fold_metrics.append({"fold": fold_i, **metrics})
            log.info("[%s] fold %d: %s", name, fold_i, metrics)

        import statistics

        mean_auc = statistics.mean(m["auc"] for m in fold_metrics)
        std_auc = statistics.stdev([m["auc"] for m in fold_metrics]) if len(fold_metrics) > 1 else 0.0
        mean_pnl = statistics.mean(m["mean_net_pnl"] for m in fold_metrics)
        per_model[name] = {
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "mean_net_pnl": round(mean_pnl, 2),
            "best_params": best_params,
            "folds": fold_metrics,
        }

    # Selection: highest mean AUC; net P&L breaks ties — a profitable model
    # beats a slightly higher-AUC model that loses money after costs.
    best_name = max(
        per_model,
        key=lambda n: (per_model[n]["mean_auc"], per_model[n]["mean_net_pnl"]),
    )
    rationale = (
        f"selected {best_name}: highest walk-forward mean AUC "
        f"({per_model[best_name]['mean_auc']}) with mean net P&L "
        f"${per_model[best_name]['mean_net_pnl']:,.2f} after costs"
    )
    log.info(rationale)

    # Refit the winner on all labeled data for the registry artifact.
    winner_spec = MODEL_SPECS[best_name]
    winner_estimator = winner_spec["estimator"]()
    for pname, pval in per_model[best_name]["best_params"].items():
        winner_estimator = winner_estimator.set(getattr(winner_estimator, pname), pval)
    winner_model = build_pipeline(winner_estimator).fit(labeled)

    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    registry_dir = os.path.join(models_dir, "registry", best_name, f"v{version}")
    os.makedirs(registry_dir, exist_ok=True)
    winner_model.write().overwrite().save(os.path.join(registry_dir, "model"))
    metadata = {
        "model_name": best_name,
        "version": f"v{version}",
        "params": per_model[best_name]["best_params"],
        "features": FEATURE_COLS,
        "label": LABEL_COL,
        "metrics": {
            k: v for k, v in per_model[best_name].items() if k != "folds"
        },
        "data_window": {"start": data_start, "end": data_end},
        "seed": seed,
        "config_hash": cfg_hash,
        "trained_at": run_ts,
    }
    with open(os.path.join(registry_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # LATEST pointer: always resolves the current champion.
    latest_path = os.path.join(models_dir, "registry", "LATEST")
    with open(latest_path, "w") as f:
        f.write(f"{best_name}/v{version}")
    log.info("registered %s -> %s", best_name, registry_dir)

    # Experiment-tracking run manifest.
    run_dir = os.path.join(runs_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    run_manifest = {
        "run_id": run_ts,
        "seed": seed,
        "config_hash": cfg_hash,
        "n_folds": n_folds,
        "features": FEATURE_COLS,
        "label": LABEL_COL,
        "data_window": {"start": data_start, "end": data_end},
        "models": per_model,
        "selection_rationale": rationale,
        "registry_artifact": registry_dir,
    }
    run_path = os.path.join(run_dir, "run.json")
    with open(run_path, "w") as f:
        json.dump(run_manifest, f, indent=2)

    summary_path = os.path.join(models_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"best_model": best_name, "comparison": per_model, "run": run_path}, f, indent=2
        )
    log.info("run manifest: %s; summary: %s", run_path, summary_path)
    return {
        "best_model": best_name,
        "registry_dir": registry_dir,
        "run_manifest": run_path,
        "comparison": per_model,
    }


def load_latest_model(models_dir: str = "models") -> tuple[PipelineModel, dict]:
    """Load the current champion from the registry LATEST pointer."""
    latest_path = os.path.join(models_dir, "registry", "LATEST")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"no registry found at {latest_path}; train first")
    with open(latest_path) as f:
        rel = f.read().strip()
    artifact_dir = os.path.join(models_dir, "registry", rel)
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        metadata = json.load(f)
    model = PipelineModel.load(os.path.join(artifact_dir, "model"))
    log.info("loaded %s from %s", metadata["model_name"], artifact_dir)
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward ML signal training")
    parser.add_argument("--models-dir", default="models", help="registry root")
    parser.add_argument("--runs-dir", default="runs", help="experiment tracking root")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS, help="walk-forward folds")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="global seed")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = Settings.from_env()

    spark = SparkSession.builder.appName("ml-signal-train").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.read.parquet(settings.s3_path(settings.silver_prefix))
        log.info("loaded %d silver feature rows", df.count())
        result = train_walkforward(
            df,
            settings,
            models_dir=args.models_dir,
            runs_dir=args.runs_dir,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        log.info("best model: %s", result["best_model"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
