"""Evaluating the Performance of AB Tests in Python."""

__version__ = "0.1.0"

from ab_test import bayesian_binomial, frequentist_binomial

__all__: list[str] = [
    "bayesian_binomial",
    "frequentist_binomial",
]


def __dir__() -> list[str]:
    return __all__
