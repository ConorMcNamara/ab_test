"""Multiple hypothesis testing corrections.

Adjusts p-values from simultaneous tests to control familywise error
rate (FWER) or false discovery rate (FDR).  All functions accept a list
of raw p-values and return the corresponding adjusted p-values in the
same order.

These corrections are framework-agnostic: extract p-values from
:class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`,
:class:`~ab_test.frequentist_binomial.stratified.StratifiedContingencyTable`,
:class:`~ab_test.frequentist_binomial.cupac.CupacExperiment`, or any
other source and pass them here.
"""

from __future__ import annotations

__all__ = [
    "adjust_pvalues",
    "bonferroni",
    "sidak",
    "holm",
    "benjamini_hochberg",
]


def bonferroni(pvalues: list[float]) -> list[float]:
    """Bonferroni correction (FWER control).

    Multiplies each p-value by the number of tests.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values.

    Returns
    -------
    list of float
        Adjusted p-values, capped at 1.0.
    """
    m = len(pvalues)
    return [min(p * m, 1.0) for p in pvalues]


def sidak(pvalues: list[float]) -> list[float]:
    """Sidak correction (FWER control).

    Slightly less conservative than Bonferroni when tests are
    independent.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values.

    Returns
    -------
    list of float
        Adjusted p-values, capped at 1.0.
    """
    m = len(pvalues)
    return [min(1.0 - (1.0 - p) ** m, 1.0) for p in pvalues]


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down correction (FWER control).

    Uniformly more powerful than Bonferroni while still controlling
    the familywise error rate.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values.

    Returns
    -------
    list of float
        Adjusted p-values, capped at 1.0.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    cummax = 0.0
    for rank, idx in enumerate(order):
        adj = pvalues[idx] * (m - rank)
        cummax = max(cummax, adj)
        adjusted[idx] = min(cummax, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg correction (FDR control).

    Controls the expected proportion of false discoveries rather than
    the familywise error rate, making it more powerful when many tests
    are performed.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values.

    Returns
    -------
    list of float
        Adjusted p-values, capped at 1.0.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        adj = pvalues[idx] * m / rank
        cummin = min(cummin, adj)
        adjusted[idx] = min(cummin, 1.0)
    return adjusted


def adjust_pvalues(pvalues: list[float], method: str = "holm") -> list[float]:
    """Adjust p-values for multiple comparisons.

    Parameters
    ----------
    pvalues : list of float
        Raw p-values.
    method : str, default='holm'
        Correction method.  One of ``'bonferroni'``, ``'sidak'``,
        ``'holm'``, or ``'benjamini_hochberg'`` (alias ``'bh'``/``'fdr'``).

    Returns
    -------
    list of float
        Adjusted p-values in the same order as the input.

    Raises
    ------
    ValueError
        If *method* is not a recognised correction name.
    """
    method = method.casefold().replace("-", "_")
    methods = {
        "bonferroni": bonferroni,
        "sidak": sidak,
        "holm": holm,
        "benjamini_hochberg": benjamini_hochberg,
        "bh": benjamini_hochberg,
        "fdr": benjamini_hochberg,
    }
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}. Choose from: {', '.join(sorted(set(methods)))}")
    return methods[method](pvalues)
