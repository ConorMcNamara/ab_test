"""Evaluating the Performance of AB Tests in Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abtest-analysis")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from ab_test import bayesian_binomial, corrections, diagnostics, frequentist_binomial

__all__: list[str] = [
    "bayesian_binomial",
    "corrections",
    "diagnostics",
    "frequentist_binomial",
]


def __dir__() -> list[str]:
    return __all__
