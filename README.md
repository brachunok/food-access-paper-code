# Perceived food access from online grocery reviews

This repository contains the analysis code for **“Online Grocery Reviews
Enable Population-Level Measurement of Perceived Food Access”** by Teresa
Groton and Benjamin Rachunok. The case study covers grocery stores in and near
Raleigh, North Carolina.

> **Repository provenance:** This public-facing repository is an AI-assisted
> cleanup and refactoring of the authors' existing analysis code and notebooks;
> it is not a newly designed analysis. The cleaned implementation has been
> tested and functions as documented. In particular, it exactly reproduces the
> archived raw store-level means, review counts, and standard errors for all 49
> Raleigh stores across all five food-access dimensions. The authors reviewed
> and remain responsible for the code, analysis, and reported results.

The release is being reconstructed from the analysis notebooks used for the
manuscript. Version 0.1 makes the central, previously notebook-only choices
explicit and testable: topic-to-dimension assignment, review sentiment coding,
store-level aggregation, and Bayesian partial pooling.

## Analysis in brief

1. Review text is embedded with `all-MiniLM-L12-v2`.
2. BERTopic identifies recurring review topics using UMAP and HDBSCAN.
3. Each topic centroid is compared by cosine similarity with the five access
   definitions in `config/access_definitions.json`.
4. Similarities are converted to probabilities by a softmax with temperature
   0.11. The top dimension is always retained; second and third dimensions are
   retained when their probability lies within 0.17 of the top probability.
5. Ratings are coded 1 star = −1, 2–4 stars = 0, and 5 stars = +1. Scores are
   averaged for each store and dimension.
6. Store means are partially pooled with a normal–normal hierarchical model.

All fixed parameters are recorded in `config/analysis.yml`.

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
make example
```

The example uses invented records and writes
`outputs/tables/example_store_scores.csv`.

To score an authorized prepared dataset:

```bash
python scripts/02_compute_store_scores.py \
  --input data/raw/reviews_with_dimensions.csv \
  --output data/derived/store_scores.csv \
  --long-output data/derived/store_dimension_scores_long.csv
```

Install `.[modeling]` to run the PyMC partial-pooling model, or create the full
Conda environment with `conda env create -f environment.yml`.

## Data access and privacy

Raw review text and reviewer metadata are deliberately excluded from this
repository. See `data/README.md` for the prepared-input schema and a synthetic
example. Do not commit raw downloads, reviewer identifiers, review URLs, photo
URLs, fitted model binaries, or API credentials.

## Repository status

This is an initial reproducibility scaffold. Remaining release work includes:

- extracting the topic-model fitting step into a command-line script;
- attaching the exact Raleigh store-boundary input and its provenance;
- porting the manuscript's figure and table generators;
- validating all released outputs against the submitted manuscript; and
- replacing the citation and code-availability URL placeholders.

The store-scoring reconstruction has been checked against the archived Raleigh
analysis. It reproduces the raw means, counts, and standard errors for all 49
stores exactly; see `VALIDATION.md`.

## License

Code is released under the MIT License. Any deposited data may have separate
terms; consult its accompanying metadata.


