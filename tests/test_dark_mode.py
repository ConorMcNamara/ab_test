"""Every plotting function supports ``dark_mode`` and leaves its default look unchanged."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from ab_test.bayesian_binomial.cluster import (
    BayesianClusterRandomizedTrial,
    plot_cluster_bayes_power_curve,
    plot_cluster_bayes_sensitivity_curve,
)
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff
from ab_test.bayesian_binomial.power_calculations import plot_bayes_power_curve, plot_bayes_sensitivity_curve
from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable
from ab_test.diagnostics import time_trend_test
from ab_test.frequentist_binomial.cluster import (
    ClusterRandomizedTrial,
    plot_cluster_power_curve,
    plot_cluster_sensitivity_curve,
)
from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.cupac import CupacExperiment, plot_cupac_power_curve, plot_cupac_sensitivity_curve
from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff
from ab_test.frequentist_binomial.gst import GroupSequentialDesign, plot_gst_power_curve, plot_gst_sensitivity_curve
from ab_test.frequentist_binomial.msprt import plot_msprt_over_time
from ab_test.frequentist_binomial.power_calculations import plot_power_curve, plot_sensitivity_curve
from ab_test.frequentist_binomial.stratified import StratifiedContingencyTable

DARK_BACKGROUND = pio.templates["plotly_dark"].layout.paper_bgcolor


def _clusters(cls):
    trial = cls("Store test", "conversion")
    for i, (s, n) in enumerate([(45, 500), (52, 480), (48, 510)]):
        trial.add(f"c{i}", s, n, group="control")
    for i, (s, n) in enumerate([(62, 490), (58, 520), (66, 505)]):
        trial.add(f"t{i}", s, n, group="treatment")
    return trial


def _strata(cls, *prior):
    table = cls("Checkout", "conversion")
    for stratum, (sc, st) in {"desktop": (120, 140), "mobile": (80, 95)}.items():
        table.add("Control", sc, 1000, *prior, stratum=stratum).add("Treatment", st, 1000, *prior, stratum=stratum)
    return table


def _segments(cls, *prior):
    men = cls("Men", "converted").add("Control", 100, 1000, *prior).add("Treatment", 130, 1000, *prior)
    women = cls("Women", "converted").add("Control", 120, 1000, *prior).add("Treatment", 125, 1000, *prior)
    return cls is BayesianContingencyTable and BayesianDiffInDiff(men, women) or DiffInDiff(men, women)


def _cupac():
    rng = np.random.default_rng(0)
    x = rng.poisson(3, 1000)
    t = rng.integers(0, 2, 1000)
    y = rng.binomial(1, np.clip(0.05 + 0.02 * x + 0.02 * t, 0, 1))
    return CupacExperiment(pd.DataFrame({"y": y, "t": t, "x": x}), "y", "t", ["x"]).fit()


def _msprt_tables():
    tables = []
    for n in (500, 1000):
        tables.append(ContingencyTable("t", "c").add("Control", n // 10, n).add("Treatment", n // 8, n))
    return tables


BAYES_SIM = {"n_samples": 50, "mc_samples": 50}

# Each entry draws one figure; ``dark_mode`` is passed through as a keyword.
PLOTS = {
    "power_curve": lambda dm: plot_power_curve(0.1, 0.2, sample_sizes=[1000, 2000], dark_mode=dm),
    "sensitivity_curve": lambda dm: plot_sensitivity_curve(0.1, sample_sizes=[1000, 2000], dark_mode=dm),
    "cluster_power_curve": lambda dm: plot_cluster_power_curve(0.1, 0.2, 0.02, 50, sample_sizes=[2000], dark_mode=dm),
    "cluster_sensitivity_curve": lambda dm: plot_cluster_sensitivity_curve(
        0.1, 0.02, 50, sample_sizes=[2000], dark_mode=dm
    ),
    "cupac_power_curve": lambda dm: plot_cupac_power_curve(0.1, 0.2, 0.3, dark_mode=dm),
    "cupac_sensitivity_curve": lambda dm: plot_cupac_sensitivity_curve(0.1, 0.3, dark_mode=dm),
    "gst_power_curve": lambda dm: plot_gst_power_curve(0.1, 0.2, 3, sample_sizes=[2000], dark_mode=dm),
    "gst_sensitivity_curve": lambda dm: plot_gst_sensitivity_curve(0.1, 3, sample_sizes=[2000], dark_mode=dm),
    "gst_boundaries": lambda dm: GroupSequentialDesign(n_analyses=3).plot_boundaries(dark_mode=dm),
    "msprt_over_time": lambda dm: plot_msprt_over_time(_msprt_tables(), ["Day 1", "Day 2"], dark_mode=dm),
    "bayes_power_curve": lambda dm: plot_bayes_power_curve(
        [1, 1], [1, 1], 0.1, alt_lift=0.2, sample_sizes=[500], dark_mode=dm, **BAYES_SIM
    ),
    "bayes_sensitivity_curve": lambda dm: plot_bayes_sensitivity_curve(
        [1, 1], [1, 1], 0.1, sample_sizes=[500], dark_mode=dm, **BAYES_SIM
    ),
    "cluster_bayes_power_curve": lambda dm: plot_cluster_bayes_power_curve(
        0.02, 100, 0.1, alt_lift=0.2, n_points=2, dark_mode=dm, **BAYES_SIM
    ),
    "cluster_bayes_sensitivity_curve": lambda dm: plot_cluster_bayes_sensitivity_curve(
        0.02, 100, 0.1, cluster_counts=[5], dark_mode=dm, **BAYES_SIM
    ),
    "bayes_contingency_pdf": lambda dm: (
        BayesianContingencyTable("x", "y").add("A", 100, 1000, 1, 1).add("B", 130, 1000, 1, 1).plot_pdf(dark_mode=dm)
    ),
    "bayes_cluster_pdf": lambda dm: _clusters(BayesianClusterRandomizedTrial).plot_pdf(dark_mode=dm),
    "time_trend": lambda dm: time_trend_test([100, 98, 102], [1000] * 3, [140, 128, 118], [1000] * 3, dark_mode=dm)[
        "figure"
    ],
}

# These call fig.show() instead of returning the figure.
SHOWN = {
    "cluster_plot": lambda dm: _clusters(ClusterRandomizedTrial).plot(dark_mode=dm),
    "bayes_cluster_plot": lambda dm: _clusters(BayesianClusterRandomizedTrial).plot(n_samples=2_000, dark_mode=dm),
    "stratified_plot": lambda dm: _strata(StratifiedContingencyTable).plot(dark_mode=dm),
    "bayes_stratified_plot": lambda dm: _strata(BayesianStratifiedContingencyTable, 1, 1).plot(
        n_samples=2_000, dark_mode=dm
    ),
    "diff_in_diff_plot": lambda dm: _segments(ContingencyTable).plot(dark_mode=dm),
    "bayes_diff_in_diff_plot": lambda dm: _segments(BayesianContingencyTable, 1, 1).plot(n_samples=2_000, dark_mode=dm),
    "cupac_plot": lambda dm: _cupac().plot(dark_mode=dm),
}


@pytest.fixture
def draw(monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: shown.append(self))

    def _draw(name, dark_mode):
        if name in PLOTS:
            return PLOTS[name](dark_mode)
        SHOWN[name](dark_mode)
        return shown[-1]

    return _draw


ALL_PLOTS = [*PLOTS, *SHOWN]


@pytest.mark.parametrize("name", ALL_PLOTS)
def test_dark_mode_uses_dark_background(draw, name):
    fig = draw(name, dark_mode=True)
    assert fig.layout.template.layout.paper_bgcolor == DARK_BACKGROUND


@pytest.mark.parametrize("name", ALL_PLOTS)
def test_default_is_not_dark(draw, name):
    fig = draw(name, dark_mode=False)
    assert fig.layout.template.layout.paper_bgcolor != DARK_BACKGROUND
