#!/usr/bin/env python3
"""Assign dimensions from a CSV of topic–definition cosine similarities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from food_access_reviews.assignment import similarities_to_assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=0.11)
    parser.add_argument("--margin", type=float, default=0.17)
    args = parser.parse_args()

    similarities = pd.read_csv(args.input, index_col=0)
    assigned = similarities_to_assignments(
        similarities,
        temperature=args.temperature,
        margin_threshold=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.output)
    print(f"Wrote {len(assigned)} topic assignments to {args.output}")


if __name__ == "__main__":
    main()

