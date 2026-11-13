#!/usr/bin/env python3
"""Compute raw store × food-access-dimension scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from food_access_reviews.scoring import compute_store_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--long-output", type=Path)
    args = parser.parse_args()

    scores, long_scores = compute_store_scores(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output)
    if args.long_output:
        args.long_output.parent.mkdir(parents=True, exist_ok=True)
        long_scores.to_csv(args.long_output, index=False)
    print(f"Wrote {len(scores)} stores to {args.output}")


if __name__ == "__main__":
    main()

