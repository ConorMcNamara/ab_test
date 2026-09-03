"""Evaluating the Performance of AB Tests for Binomial Distributions in Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abtest-analysis")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from ab_test.bayesian_binomial import (
    contingency,
    credible_intervals,
    diff_in_diff,
    power_calculations,
    stats_tests,
    stratified,
    utils,
)

__all__: list[str] = [
    "contingency",
    "credible_intervals",
    "diff_in_diff",
    "power_calculations",
    "stats_tests",
    "stratified",
    "utils",
]


def __dir__() -> list[str]:
    return __all__
