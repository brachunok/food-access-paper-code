import numpy as np
import pandas as pd
import pytest

from food_access_reviews.scoring import compute_store_scores, star_to_signal


def test_star_signal_matches_manuscript_rule():
    stars = pd.Series([1, 2, 3, 4, 5])
    assert star_to_signal(stars).tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_invalid_star_raises():
    with pytest.raises(ValueError):
        star_to_signal(pd.Series([0, 5]))


def test_multilabel_reviews_contribute_to_each_dimension():
    reviews = pd.DataFrame(
        {
            "reviewId": ["a", "b", "c"],
            "placeId": ["store-1", "store-1", "store-1"],
            "stars": [1, 5, 3],
            "pred_dim": ["Acceptability", "Acceptability", "Availability"],
            "pred_dim2": ["Accommodation", None, "Acceptability"],
        }
    )
    wide, grouped = compute_store_scores(reviews)

    assert wide.loc["store-1", "Acceptability_n"] == 3
    assert np.isclose(wide.loc["store-1", "Acceptability_mean"], 0.0)
    assert wide.loc["store-1", "Accommodation_n"] == 1
    assert wide.loc["store-1", "Accommodation_mean"] == -1.0
    assert len(grouped) == 3

