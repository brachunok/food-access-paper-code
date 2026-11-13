import numpy as np
import pandas as pd

from food_access_reviews.assignment import (
    DIMENSION_ORDER,
    similarities_to_assignments,
)


def test_probabilities_sum_to_one_and_top_label_is_retained():
    similarities = pd.DataFrame(
        [[0.9, 0.1, 0.0, -0.1, -0.2]], columns=DIMENSION_ORDER, index=[7]
    )
    result = similarities_to_assignments(similarities)
    probability_columns = [f"{dimension}_p" for dimension in DIMENSION_ORDER]

    assert np.isclose(result.loc[7, probability_columns].sum(), 1.0)
    assert result.loc[7, "pred_dim"] == "Availability"
    assert pd.isna(result.loc[7, "pred_dim2"])


def test_close_runner_up_is_retained():
    similarities = pd.DataFrame(
        [[0.50, 0.49, 0.0, -0.1, -0.2]], columns=DIMENSION_ORDER
    )
    result = similarities_to_assignments(similarities)
    assert result.loc[0, "pred_dim"] == "Availability"
    assert result.loc[0, "pred_dim2"] == "Accessibility"

