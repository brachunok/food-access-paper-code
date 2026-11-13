# Validation record

## Store-scoring reconstruction

Date: 2026-11-13

The refactored `compute_store_scores` implementation was run on the private
prepared Raleigh review file and compared with the archived store-score table.

- Stores reconstructed: 49
- Stores in archived reference: 49
- Store identifiers matched: 49
- Fields compared: raw mean, contributing-review count, and standard error for
  each of the five food-access dimensions (15 fields total)
- Maximum absolute difference: 0 for every compared field

The validation uses private source data that are intentionally excluded from
the repository. The synthetic file under `data/example/` exercises the same
public code path without redistributing review text or reviewer metadata.

