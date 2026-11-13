#!/usr/bin/env python3
"""Partially pool store scores across stores for all five dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from food_access_reviews.pooling import partial_pool_one_dimension
from food_access_reviews.scoring import DIMENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    scores = pd.read_csv(args.input, index_col="placeId")
    summaries = []
    for dimension in DIMENSIONS:
        scores, _, summary = partial_pool_one_dimension(
            scores,
            dimension,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=args.seed,
        )
        summaries.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output)
    args.diagnostics.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Wrote partially pooled scores for {len(scores)} stores to {args.output}")


if __name__ == "__main__":
    main()
