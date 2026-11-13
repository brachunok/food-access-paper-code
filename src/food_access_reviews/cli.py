"""Small command-line interface for store scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .scoring import compute_store_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prepared review CSV")
    parser.add_argument("output", type=Path, help="Store-score CSV to create")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reviews = pd.read_csv(args.input)
    scores, _ = compute_store_scores(reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output)
    print(f"Wrote {len(scores)} stores to {args.output}")


if __name__ == "__main__":
    main()

