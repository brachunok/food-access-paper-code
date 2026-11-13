# Data

The repository does **not** distribute raw Google review text, reviewer IDs,
review URLs, photo URLs, or other platform-derived personal metadata. Those
fields are not required to run the store-scoring code and should not be added
to Git history.

## Expected prepared input

The scoring pipeline accepts one row per review with these fields:

| Field | Type | Description |
|---|---|---|
| `reviewId` | string | De-identified row identifier |
| `placeId` | string | Stable store identifier |
| `stars` | integer | Google rating from 1 through 5 |
| `pred_dim` | string | Highest-probability access dimension |
| `pred_dim2` | string/blank | Optional second dimension |
| `pred_dim3` | string/blank | Optional third dimension |

The example file contains invented records. The project authors are reviewing
the platform terms, ethical constraints, and journal requirements before
deciding whether a de-identified derived Raleigh dataset can be deposited.

## Local-only layout

If authorized source data are available, place them under `data/raw/`. That
directory is ignored by Git. Generated intermediate data belong in
`data/derived/`, which is also ignored except for its placeholder.

