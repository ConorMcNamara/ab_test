"""Our wrapper for analyzing experiment results"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate


class ContingencyTable:
    """A class for analyzing experiment results"""

    def __init__(self, name: str, spend: Optional[float] = None, msrp: Optional[float] = None) -> None:
        """ContingencyTable is our class for creating and analyzing experiment results

        Parameters
        ----------
        name : str
            The name of our experiment associated with our Contingency Table
        spend : float
            The amount we spent for this campaign. Used to calculate the ROAS of our campaign
        msrp : float
            The average msrp of our product. Used to calculate the revenue return of our campaign
        """
        self.experiment_name = name
        self.names = []
        self.spend = spend
        self.msrp = msrp
        self.cells = {"experiment_name": self.experiment_name, "spend": self.spend, "msrp": self.msrp, "table": {}}
        self.successes = 0
        self.trials = 0

    def add(self, cell_name: str, successes: float, trials: float) -> "ContingencyTable":
        """A method to add cells to our contingency table

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        successes : float
            The number of successes in our cell_name
        trials : float
            The number of trials in our cell_name

        Returns
        -------
        ContingencyTable, to be chained with other methods
        """
        cell_dict = {"successes": successes, "trials": trials}
        self.successes += successes
        self.trials += trials
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        return self

    def to_df(self, method: str = "pandas", include_total: bool = False) -> Union[pd.DataFrame, pl.DataFrame]:
        """Returns our ContingencyTable as a DataFrame

        Parameters
        ----------
        method : {"pandas", "polars"}
            Whether we want our DataFrame as a pandas or polars DataFrame
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(self.to_list(include_total), columns=["cell_name", "successes", "trials"])
        elif method == "polars":
            return_df = pl.DataFrame(
                self.to_list(include_total), schema=["cell_name", "successes", "trials"], orient="row"
            )
        elif method == "pyspark":
            raise NotImplementedError("Have not implemented Pyspark yet")
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            raise NotImplementedError("Have not implemented modin yet")
        else:
            raise ValueError(f"Methdod {method} not supported for creating DataFrames")
        return return_df

    def to_list(self, include_total: bool = False) -> list:
        """Returns our ContingencyTable as a list

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
                    total_list = ["Total", self.successes, self.trials]
                    return_list.append(total_list)
            else:
                pass
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray:
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

    def serialize(self, include_total: bool = False):
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
            total_dict = {"successes": self.successes, "trials": self.trials}
            self.cells["table"]["Total"] = total_dict
        return self.cells

    def deserialize(self, serial: dict) -> "ContingencyTable":
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
        return self

    def analyze(
        self,
        lift: str = "relative",
        test_method: str = "score",
        conf_int_method: str = "binary_search",
        alpha: float = 0.05,
    ) -> None:
        raise NotImplementedError("Haven't implemented yet")

    def __str__(self):
        return tabulate(self.to_list(include_total=True), headers=["cell_name", "successes", "trials"], tablefmt="grid")
