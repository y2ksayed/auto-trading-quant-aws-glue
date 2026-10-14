"""Walk-forward validation splits for time-series model evaluation.

Random train/test splits are a credibility killer on time-series data: they
let the model train on the future and validate on the past, which inflates
metrics and hides lookahead bias. Walk-forward validation instead rolls a
training window forward in time — every validation fold is strictly *after*
its training fold, mimicking how a model is actually retrained and deployed.

This module is intentionally Spark-free so the split logic is unit-testable.
"""

from __future__ import annotations

from datetime import datetime


def walk_forward_splits(
    timestamps: list,
    n_folds: int = 3,
) -> list[tuple[list, list]]:
    """Split an ordered timeline into walk-forward (train, validation) folds.

    The sorted unique timestamps are divided into ``n_folds + 1`` contiguous
    blocks. Fold ``i`` trains on blocks ``[0..i]`` and validates on block
    ``i + 1``. Folds never overlap and validation is always in the future
    relative to training.

    Args:
        timestamps: Ordered (or unordered — they get sorted) timestamps.
        n_folds: Number of validation folds, must be >= 1.

    Returns:
        List of ``(train_timestamps, val_timestamps)`` tuples, one per fold.

    Raises:
        ValueError: if there aren't enough distinct timestamps for the folds.
    """
    if not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError(f"n_folds must be a positive int, got {n_folds!r}")
    ordered = sorted(set(timestamps))
    n_blocks = n_folds + 1
    if len(ordered) < n_blocks:
        raise ValueError(
            f"need at least {n_blocks} distinct timestamps for {n_folds} folds, "
            f"got {len(ordered)}"
        )

    # Distribute timestamps into contiguous blocks as evenly as possible.
    block_size, remainder = divmod(len(ordered), n_blocks)
    blocks: list[list] = []
    start = 0
    for i in range(n_blocks):
        size = block_size + (1 if i < remainder else 0)
        blocks.append(ordered[start : start + size])
        start += size

    folds = []
    for i in range(n_folds):
        train_ts = [t for block in blocks[: i + 1] for t in block]
        val_ts = blocks[i + 1]
        folds.append((train_ts, val_ts))
    return folds


def fold_summary(folds: list[tuple[list, list]]) -> list[dict]:
    """Human-readable summary of folds for logging."""
    return [
        {
            "fold": i,
            "n_train": len(train),
            "n_val": len(val),
            "train_end": str(train[-1]),
            "val_start": str(val[0]),
            "val_end": str(val[-1]),
        }
        for i, (train, val) in enumerate(folds)
    ]


def _parse_ts(value) -> datetime:
    """Best-effort timestamp parsing for fold boundary logging."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
