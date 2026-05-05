"""Our wrapper for analyzing experiment results"""

import math
from typing import Any, overload

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from tabulate import tabulate

from ab_test.bayesian_binomial.credible_intervals import credible_interval, individual_credible_interval
from ab_test.bayesian_binomial.stats_tests import calculate_metrics

class BayesianContingencyTable:
    """A class for analyzing experiment results using Bayesian approaches"""

    def __init__(self, name: str, metric_name: str, spend: float | None = None, msrp: float | None = None) -> None:
        """BayesianContingencyTable is our class for creating and analyzing experiment results

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
        self.alphas: list[float] = []
        self.betas: list[float] = []
        self.incremental_results: dict[str, Any] | None = None
        self.individual_results: dict[str, dict[str, float]] = {}

    def add(self, cell_name: str, successes: int, trials: int, alpha: float, beta: float) -> "BayesianContingencyTable":
        """A method to add cells to our contingency table

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        successes : int
            The number of successes in our cell_name
        trials : int
            The number of trials in our cell_name

        Returns
        -------
        ContingencyTable, to be chained with other methods
        """
        cell_dict = {"successes": successes, "trials": trials, "alpha": alpha, "beta": beta}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.successes.append(successes)
        self.trials.append(trials)
        self.alphas.append(alpha)
        self.betas.append(beta)
        return self

    def to_df(self, method: str = "pandas", include_total: bool = False, spark_session: Any | None = None, ibis_backend: Any | None = None) -> pd.DataFrame | pl.DataFrame | Any:
        """Returns our BayesianContingencyTable as a DataFrame

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
        Our ContingencyTable as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(
                self.to_list(include_total), columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
        elif method == "polars":
            return_df = pl.DataFrame(
                self.to_list(include_total), schema=["cell_name", "successes", "trials", "alpha", "beta"], orient="row"
            )
        elif method == "pyspark":
            from pyspark.sql import SparkSession
            from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
            if spark_session is None:
                spark_session = SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No active SparkSession found. Please provide a spark_session argument.")
            schema = StructType([
                StructField("cell_name", StringType(), True),
                StructField("successes", IntegerType(), True),
                StructField("trials", IntegerType(), True),
                StructField("alpha", DoubleType(), True),
                StructField("beta", DoubleType(), True),
            ])
            return_df = spark_session.createDataFrame(self.to_list(include_total), schema=schema)
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            import modin.pandas as mpd
            return_df = mpd.DataFrame(
                self.to_list(include_total), columns=mpd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
        elif method == "ibis":
            import ibis
            if ibis_backend is not None:
                ibis.set_backend(ibis_backend)
            pandas_df = pd.DataFrame(
                self.to_list(include_total), columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
            return_df = ibis.memtable(pandas_df)
        elif method == "narwhals":
            import narwhals as nw
            pandas_df = pd.DataFrame(
                self.to_list(include_total), columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
            return_df = nw.from_native(pandas_df)
        else:
            raise ValueError(f"Method {method} not supported for creating DataFrames")
        return return_df


    def to_list(self, include_total: bool = False) -> list[Any]:
        """Returns our BayesianContingencyTable as a list

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a list
        """
        return_list = []
        for key, value in self.cells.items():
            if key == "table":
                for name in self.names:
                    loop_list = [name] + list(value[f"{name}"].values())
                    return_list.append(loop_list)
                if include_total:
                    total_list = ["Total", np.sum(self.successes), np.sum(self.trials), np.nan, np.nan]
                    return_list.append(total_list)
            else:
                pass
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray[Any, Any]:
        """Returns our ContingencyTable as a numpy array

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a numpy array
        """
        return np.array(self.to_list(include_total))

    def serialize(self, include_total: bool = False) -> dict[str, Any]:
        """Returns our ContingencyTable as a JSON, with all information

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a JSON
        """
        if include_total:
            total_dict = {"successes": self.successes, "trials": self.trials, "alpha": np.nan, "beta": np.nan}
            self.cells["table"]["Total"] = total_dict
        return self.cells

    def deserialize(self, serial: dict[str, Any]) -> "BayesianContingencyTable":
        """Takes in a serialized version of our ContingencyTable. Used when we want to populate our
        ContingencyTable with results from a prior campaign.

        Parameters
        ----------
        serial : dict
            A serialized version of our ContingencyTable

        Returns
        -------
        Itself, to be chained with other methods
        """
        self.experiment_name = serial["experiment_name"]
        self.spend = serial["spend"]
        self.msrp = serial["msrp"]
        self.cells = serial
        self.metric_name = serial["metric_name"]
        self.alphas = serial["alphas"]
        self.betas = serial["betas"]
        return self

    @overload
    @staticmethod
    def _convert_to_tabulate_str(value: float, lift: str) -> str | float:
        ...

    @overload
    @staticmethod
    def _convert_to_tabulate_str(value: list[Any], lift: str) -> list[Any]:
        ...

    @staticmethod
    def _convert_to_tabulate_str(value: float | list[Any], lift: str) -> str | list[Any] | float:
        """Converts our lift values to either percentages or dollar signs

        Parameters
        ----------
        value : float or list
            The value we are changing
        lift : str
            Depending on the lift type, whether we are adding percentages or dollar signs

        Returns
        -------
        Our new str_value, as either a percentage or dollar sign
        """
        str_value: str | list[str] | float
        if isinstance(value, float):
            if lift in ["revenue", "roas"]:
                str_value = f"${round(value, 2):,}"
            elif lift in ["absolute", "relative"]:
                str_value = f"{round(value * 100.0, 2)}%"
            elif lift in ["incremental"]:
                str_value = value
            else:
                raise ValueError(f"No support for {lift}")
        elif isinstance(value, list):
            if lift in ["revenue", "roas"]:
                str_value = [f"${round(val, 2):,}" for val in value]
            elif lift in ["absolute", "relative"]:
                str_value = [f"{round(val * 100.0, 2)}%" for val in value]
            elif lift == "incremental":
                str_value = value
            else:
                raise ValueError(f"No support for {lift}")
        else:
            raise TypeError(f"No support for converting {value} to string")
        return str_value
