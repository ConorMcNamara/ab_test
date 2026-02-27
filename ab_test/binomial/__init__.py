"""Evaluating the Performance of AB Tests for Binomial Distributions in Python"""

__version__ = "0.1.0"

from ab_test.binomial import confidence_intervals, power_calculations, stats_tests, utils

__all__: list[str] = [
    "confidence_intervals",
    "power_calculations",
    "stats_tests",
    "utils",
]


def __dir__() -> list[str]:
    return __all__
