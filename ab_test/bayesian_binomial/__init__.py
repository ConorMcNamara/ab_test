"""Evaluating the Performance of AB Tests for Binomial Distributions in Python"""

__version__ = "0.1.0"

from ab_test.bayesian_binomial import credible_intervals, utils

__all__: list[str] = [
    "credible_intervals",
    "utils",
]


def __dir__() -> list[str]:
    return __all__
