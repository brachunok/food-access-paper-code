"""Analysis tools for the perceived food-access study."""

from .assignment import similarities_to_assignments
from .scoring import DIMENSIONS, compute_store_scores, star_to_signal

__all__ = [
    "DIMENSIONS",
    "compute_store_scores",
    "similarities_to_assignments",
    "star_to_signal",
]

__version__ = "0.1.0"

