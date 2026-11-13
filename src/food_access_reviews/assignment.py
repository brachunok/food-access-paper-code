"""Assign BERTopic topics to food-access dimensions."""

from __future__ import annotations

import numpy as np
import pandas as pd

DIMENSION_ORDER = (
    "Availability",
    "Accessibility",
    "Accommodation",
    "Affordability",
    "Acceptability",
)


def similarities_to_assignments(
    similarities: pd.DataFrame,
    *,
    temperature: float = 0.11,
    margin_threshold: float = 0.17,
    maximum_labels: int = 3,
) -> pd.DataFrame:
    """Convert topic–definition cosine similarities to dimension assignments.

    Similarities are mean-centred within topic and temperature-scaled before
    softmax. The highest-probability dimension is always retained. The second
    and third dimensions are retained when the difference between their
    probability and the top probability is no greater than ``margin_threshold``.
    """
    missing = [column for column in DIMENSION_ORDER if column not in similarities]
    if missing:
        raise ValueError(f"Missing similarity columns: {missing}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 1 <= maximum_labels <= len(DIMENSION_ORDER):
        raise ValueError("maximum_labels must be between 1 and 5")

    values = similarities.loc[:, DIMENSION_ORDER].to_numpy(dtype=float)
    centred = values - values.mean(axis=1, keepdims=True)
    logits = centred / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

    probability_columns = [f"{dimension}_p" for dimension in DIMENSION_ORDER]
    output = similarities.copy()
    output[probability_columns] = probabilities

    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    top_probability = sorted_probabilities[:, 0]

    output["pred_dim"] = [DIMENSION_ORDER[index] for index in order[:, 0]]
    output["pred_prob"] = top_probability

    if maximum_labels >= 2:
        output["prob_margin"] = top_probability - sorted_probabilities[:, 1]

    for rank in range(2, maximum_labels + 1):
        candidate_probability = sorted_probabilities[:, rank - 1]
        keep = (top_probability - candidate_probability) <= margin_threshold
        labels = np.full(len(output), None, dtype=object)
        label_probabilities = np.full(len(output), np.nan)
        labels[keep] = [DIMENSION_ORDER[index] for index in order[keep, rank - 1]]
        label_probabilities[keep] = candidate_probability[keep]
        output[f"pred_dim{rank}"] = labels
        output[f"pred_prob{rank}"] = label_probabilities

    return output

