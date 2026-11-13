"""Review-level coding and store-level aggregation."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

DIMENSIONS = (
    "Availability",
    "Accessibility",
    "Accommodation",
    "Affordability",
    "Acceptability",
)

STAR_SIGNAL = {1: -1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0}


def star_to_signal(stars: pd.Series) -> pd.Series:
    """Map Google stars to the study's negative/neutral/positive signal."""
    numeric = pd.to_numeric(stars, errors="coerce")
    invalid = numeric.notna() & ~numeric.isin(STAR_SIGNAL)
    if invalid.any():
        values = sorted(numeric.loc[invalid].unique().tolist())
        raise ValueError(f"Stars must be integers from 1 to 5; found {values}")
    return numeric.map(STAR_SIGNAL).astype(float)


def dimension_columns(frame: pd.DataFrame) -> list[str]:
    """Return pred_dim columns in rank order."""
    columns = [column for column in frame if re.fullmatch(r"pred_dim\d*", column)]

    def rank(column: str) -> int:
        suffix = column.removeprefix("pred_dim")
        return int(suffix) if suffix else 1

    return sorted(columns, key=rank)


def reviews_to_long(
    reviews: pd.DataFrame,
    *,
    dimensions: Iterable[str] = DIMENSIONS,
) -> pd.DataFrame:
    """Expand multi-label review assignments to one row per review–dimension."""
    required = {"placeId", "stars"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labels = dimension_columns(reviews)
    if not labels:
        raise ValueError("No pred_dim, pred_dim2, ... columns found")

    working = reviews.copy()
    working["dim_signal"] = star_to_signal(working["stars"])
    working["dim_sentiment"] = pd.cut(
        working["dim_signal"],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["Low", "Neutral", "High"],
    )
    identity = [column for column in working if column not in labels]
    long = working.melt(
        id_vars=identity,
        value_vars=labels,
        var_name="label_rank",
        value_name="dim",
    )
    long = long.loc[long["dim"].isin(tuple(dimensions))].copy()
    long["label_k"] = (
        long["label_rank"]
        .str.extract(r"pred_dim(\d*)", expand=False)
        .replace("", "1")
        .astype(int)
    )
    return long


def compute_store_scores(
    reviews: pd.DataFrame,
    *,
    dimensions: Iterable[str] = DIMENSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute store × dimension raw scores, counts, and standard errors.

    Returns the wide table used by the partial-pooling model and its long-form
    precursor. A review with multiple labels contributes once to each assigned
    dimension.
    """
    dimensions = tuple(dimensions)
    long = reviews_to_long(reviews, dimensions=dimensions)
    grouped = (
        long.groupby(["placeId", "dim"], observed=True, as_index=False)
        .agg(
            mean_signal=("dim_signal", "mean"),
            n_reviews=("dim_signal", "size"),
            n_low=("dim_sentiment", lambda values: (values == "Low").sum()),
            n_neutral=("dim_sentiment", lambda values: (values == "Neutral").sum()),
            n_high=("dim_sentiment", lambda values: (values == "High").sum()),
        )
    )
    grouped["p_low"] = grouped["n_low"] / grouped["n_reviews"]
    grouped["p_high"] = grouped["n_high"] / grouped["n_reviews"]
    grouped["var_y"] = (
        grouped["p_low"]
        + grouped["p_high"]
        - grouped["mean_signal"].pow(2)
    )
    grouped["SE_mean"] = np.sqrt(grouped["var_y"] / grouped["n_reviews"]).clip(
        lower=1e-6
    )

    pieces = []
    for value, suffix in (
        ("mean_signal", "mean"),
        ("n_reviews", "n"),
        ("SE_mean", "se"),
    ):
        piece = grouped.pivot(index="placeId", columns="dim", values=value).reindex(
            columns=dimensions
        )
        piece.columns = [f"{column}_{suffix}" for column in piece.columns]
        pieces.append(piece)

    wide = pieces[0].join(pieces[1:], how="outer")
    count_columns = [f"{dimension}_n" for dimension in dimensions]
    wide[count_columns] = wide[count_columns].fillna(0).astype(int)
    return wide, grouped

