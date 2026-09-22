"""Evaluating the Performance of AB Tests for Normal Distributions in Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abtest-analysis")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from ab_test.frequentist_normal import (
    confidence_intervals,
    normal_table,
    stats_tests,
    utils,
)

__all__: list[str] = [
    "confidence_intervals",
    "normal_table",
    "stats_tests",
    "utils",
]


def __dir__() -> list[str]:
    return __all__