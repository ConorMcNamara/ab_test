"""Our wrapper for analyzing experiment results."""
from __future__ import annotations

import functools
import math
from statistics import variance
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate

from ab_test._continuous_table import BaseContinuousTable
from ab_test._display import convert_to_tabulate_str
from ab_test.frequentist_binomial.contingency import _scale_bound


class NormalTable(BaseContinuousTable):
    _columns: ClassVar[list[str]] = ["cell_name", "means", "variances"]
    _pyspark_types: ClassVar[dict[str, str]] = {
        "cell_name": "StringType",
        "means": "DoubleType",
        "variances": "DoubleType",
        "trials": "IntegerType"
    }

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        return ["Total", np.sum(np.array(self.means) * np.array(self.trials)) / np.sum(self.trials), np.nan, np.sum(self.trials)]

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        return {"means": float(np.sum(np.array(self.means) * np.array(self.trials))) / np.sum(self.trials), "variances": np.nan, "trials": int(np.sum(self.trials))}

    def add(self, cell_name: str, means: float, variances: float, trials: int) -> "NormalTable":
        """Add cells to our contingency table.

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        means : float
            The mean / average in our cell_name
        variances: float
            The variances in our cell_name
        trials : int
            The number of trials in our cell_name

        Returns
        -------
        NormalTable, to be chained with other methods
        """
        cell_dict = {"means": means, "variances": variances, "trials": trials}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.means.append(means)
        self.variances.append(variances)
        self.trials.append(trials)
        return self

    def add_data(self, cell_name: str, data: np.generic | np.ndarray | list | tuple) -> "NormalTable":
        """

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        data: numpy array, list, or tuple
            The dataset of outcomes for our cell_name

        Returns
        -------
        NormalTable, to be chained with other methods
        """
        means = float(np.mean(data))
        variances = float(np.var(data, ddof=1))
        trials = int(len(data))
        cell_dict = {"means": means, "variances": variances, "trials": trials}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.means.append(means)
        self.variances.append(variances)
        self.trials.append(trials)
        return self

    def analyze(
            self,
            lift: str = "relative",
            test_method: str = "score",
            conf_int_method: str = "binary_search",
            alpha: float = 0.05,
            null_lift: float = 0.0,
            *,
            tau: float | None = None,
    ) -> str:
        """Analyzes the effect of our experiments through the ContingencyTable.

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue', 'cpa'}
            The kind of lift we are measuring for our campaign
        test_method : str
            The method we plan to use to assess whether our result is
            statistically significant.  One of ``'score'``, ``'likelihood'``,
            ``'t'``, ``'permutation'``, ``'bootstrap'``, or ``'msprt'``.
        conf_int_method : str
            The method we plan to use to craft confidence intervals of our lift
        alpha : float, default = 0.05
            The alpha level of our experiment, to be used to craft confidence intervals.
        null_lift : float
            Lift associated with null hypothesis. Defaults to 0.0.
        tau : float or None, optional
            Scale of the Gaussian mixing distribution for the mSPRT test.
            Only used when ``test_method="msprt"``. When ``None``, the scale
            is derived from the data. See :func:`~ab_test.frequentist_binomial.msprt.msprt_test`.

        Returns
        -------
        The results (lift as well as confidence intervals) of our experiment in string format, to be printed
        """
        if len(self.names) != 2:
            raise ValueError(f"analyze requires exactly 2 variants, got {len(self.names)}")
        lift = lift.casefold()