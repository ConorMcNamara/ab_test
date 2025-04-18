"""Evaluating the Performance of AB Tests in Python"""

__version__ = "0.1.0"

from typing import List

from ab_test import binomial

__all__: List[str] = ["binomial"]


def __dir__() -> List[str]:
    return __all__
