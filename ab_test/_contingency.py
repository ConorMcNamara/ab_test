"""Shared base class for the frequentist and Bayesian contingency tables.

Holds the storage, export (DataFrame/list/numpy), (de)serialization, string
rendering, and forest-plot machinery common to both
:class:`~ab_test.frequentist_binomial.contingency.ContingencyTable` and
:class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`.

Subclasses supply their column layout via the ``_columns`` / ``_pyspark_types``
class variables and the ``_total_row`` / ``_total_cell`` / ``_deserialize_extra``
hooks, and implement their own ``add`` and ``analyze`` methods.
"""

from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate

from ab_test._display import render_forest_plot


class BaseContingencyTable:
    """Common storage, export, and plotting for contingency tables."""

    #: Ordered column names, including the leading ``"cell_name"``.
    _columns: ClassVar[list[str]]
    #: Column name -> PySpark ``types`` class name, used to build a schema lazily.
    _pyspark_types: ClassVar[dict[str, str]]

    def __init__(self, name: str, metric_name: str, spend: float | None = None, msrp: float | None = None) -> None:
        """Create an empty contingency table.

        Parameters
        ----------
        name : str
            The name of our experiment associated with our Contingency Table
        metric_name : str
            The name of our metric
        spend : float
            The amount we spent for this campaign. Used to calculate the ROAS of our campaign
        msrp : float
            The average msrp of our product. Used to calculate the revenue return of our campaign
        """
        self.experiment_name: str = name
        self.names: list[str] = []
        self.metric_name: str = metric_name
        self.spend: float | None = spend
        self.msrp: float | None = msrp
        self.cells: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": {},
        }
        self.successes: list[int] = []
        self.trials: list[int] = []
        self.incremental_results: dict[str, Any] | None = None
        self.individual_results: dict[str, dict[str, float]] = {}

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        raise NotImplementedError

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        raise NotImplementedError

    def _deserialize_extra(self, serial: dict[str, Any]) -> None:
        """Restore any subclass-specific per-cell state during :meth:`deserialize`."""

    def to_df(
        self,
        method: str = "pandas",
        include_total: bool = False,
        spark_session: Any | None = None,
        ibis_backend: Any | None = None,
    ) -> pd.DataFrame | pl.DataFrame | Any:
        """Return the contingency table as a DataFrame.

        Parameters
        ----------
        method : {"pandas", "polars", "pyspark", "modin", "ibis", "narwhals"}
            Whether we want our DataFrame as a pandas, polars, pyspark, modin, ibis, or narwhals DataFrame
        include_total : bool, default=False
            Whether we want to include another section with the total amount
        spark_session : SparkSession, optional
            An active SparkSession, required when method="pyspark"
        ibis_backend : ibis backend, optional
            An Ibis backend connection. When provided, it is set as the active backend before
            creating the memtable. When omitted, the existing default backend is used.

        Returns
        -------
        The contingency table as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(self.to_list(include_total), columns=pd.Index(self._columns))
        elif method == "polars":
            return_df = pl.DataFrame(self.to_list(include_total), schema=list(self._columns), orient="row")
        elif method == "pyspark":
            from pyspark.sql import SparkSession
            from pyspark.sql import types as spark_types

            if spark_session is None:
                spark_session = SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No active SparkSession found. Please provide a spark_session argument.")
            schema = spark_types.StructType(
                [
                    spark_types.StructField(col, getattr(spark_types, self._pyspark_types[col])(), True)
                    for col in self._columns
                ]
            )
            return_df = spark_session.createDataFrame(self.to_list(include_total), schema=schema)
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            import modin.pandas as mpd  # type: ignore[import-not-found]

            return_df = mpd.DataFrame(self.to_list(include_total), columns=mpd.Index(self._columns))
        elif method == "ibis":
            import ibis  # type: ignore[import-not-found]

            if ibis_backend is not None:
                ibis.set_backend(ibis_backend)
            pandas_df = pd.DataFrame(self.to_list(include_total), columns=pd.Index(self._columns))
            return_df = ibis.memtable(pandas_df)
        elif method == "narwhals":
            import narwhals as nw

            pandas_df = pd.DataFrame(self.to_list(include_total), columns=pd.Index(self._columns))
            return_df = nw.from_native(pandas_df)
        else:
            raise ValueError(f"Method {method} not supported for creating DataFrames")
        return return_df

    def to_list(self, include_total: bool = False) -> list[Any]:
        """Return the contingency table as a list.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        The contingency table as a list
        """
        return_list = []
        for name in self.names:
            return_list.append([name] + list(self.cells["table"][name].values()))
        if include_total:
            return_list.append(self._total_row())
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray[Any, Any]:
        """Return the contingency table as a numpy array.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        The contingency table as a numpy array
        """
        return np.array(self.to_list(include_total))

    def serialize(self, include_total: bool = False) -> dict[str, Any]:
        """Return the contingency table as a JSON-compatible dict, with all information.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        The contingency table as a JSON-compatible dict
        """
        table = dict(self.cells["table"])
        if include_total:
            table["Total"] = self._total_cell()
        return {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": table,
        }

    def deserialize(self, serial: dict[str, Any]) -> Self:
        """Populate the contingency table from a serialized version.

        Used when we want to populate the table with results from a prior campaign.

        Parameters
        ----------
        serial : dict
            A serialized version of a contingency table

        Returns
        -------
        Itself, to be chained with other methods
        """
        self.experiment_name = serial["experiment_name"]
        self.metric_name = serial["metric_name"]
        self.spend = serial["spend"]
        self.msrp = serial["msrp"]
        self.cells = serial
        self.names = list(serial["table"].keys())
        self.successes = [v["successes"] for v in serial["table"].values()]
        self.trials = [v["trials"] for v in serial["table"].values()]
        self._deserialize_extra(serial)
        return self

    def plot(
        self,
        is_individual: bool = True,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Plot the point estimates as well as confidence/credible intervals.

        Parameters
        ----------
        is_individual : bool, default=True
            Whether we are looking at the individual performance of each cell or
            the comparative performance between variants.
        reverse_plot : bool, default=True
            Whether we are reversing the order of our plot or not.
        color : str, list, dict, or None, default=None
            If None, uses Plotly's default color scheme.
            If a string, one of the colorblind-friendly palette names: ``"ibm"``,
            ``"wong"``, ``"ito"``, ``"tol"``, ``"tol_bright"``, ``"tol_vibrant"``,
            ``"tol_muted"``, ``"tol_light"``.
            If a list, each item corresponds to a color for the relevant group.
            If a dict, keys are group names and values are colors.

        Notes
        -----
        This function is intended to be run after either .analyze() or
        .analyze_individually().
        """
        render_forest_plot(
            self.names,
            self.individual_results,
            self.incremental_results,
            is_individual=is_individual,
            reverse_plot=reverse_plot,
            color=color,
        )

    def __str__(self) -> str:
        """Return a grid-formatted string representation of the contingency table."""
        result: str = tabulate(
            self.to_list(include_total=True),
            headers=list(self._columns),
            tablefmt="grid",
        )
        return result
