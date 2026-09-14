"""CUPAC and MLRATE variance reduction for A/B tests.

Provides variance reduction for A/B tests by fitting a predictive model
on covariates and adjusting outcomes via the CUPED framework.  Two methods
are supported:

* **CUPAC** fits OLS on control-group covariates and predicts for all users.
* **MLRATE** accepts any scikit-learn-compatible estimator and uses K-fold
  cross-fitting so that flexible models (random forests, gradient boosting,
  etc.) produce valid inference without overfitting bias.

Both methods use HC2 robust standard errors for the final treatment-effect
estimate.

References
----------
Grover, A. et al. (2018). "CUPAC — Controlled-experiment Using Pre-experiment
    data with Adjusted Covariates."
Deng, A. et al. (2013). "Improving the Sensitivity of Online Controlled
    Experiments by Utilizing Pre-Experiment Data."
Lin, W. (2013). "Agnostic notes on regression adjustments to experimental
    data: Reexamining Freedman's critique."
Guo, Y. et al. (2021). "Machine Learning for Variance Reduction in Online
    Experiments."
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as ss

from ab_test._display import resolve_plot_color
from ab_test.frequentist_binomial.power_calculations import (
    abtest_power,
    minimum_detectable_lift,
    required_sample_size,
    score_power,
)

__all__ = [
    "CupacExperiment",
    "cupac_adjusted_power",
    "cupac_minimum_detectable_lift",
    "cupac_required_sample_size",
    "plot_cupac_power_curve",
    "plot_cupac_sensitivity_curve",
]


def _ols_fit(X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Fit OLS via the normal equations.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (should include an intercept column).
    y : ndarray of shape (n,)
        Response vector.

    Returns
    -------
    beta : ndarray of shape (p,)
        OLS coefficient vector.
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _hc2_standard_errors(
    X: np.ndarray[Any, Any], y: np.ndarray[Any, Any], beta: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """HC2 heteroskedasticity-robust standard errors.

    Uses QR decomposition to compute the hat-matrix diagonal efficiently.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Response vector.
    beta : ndarray of shape (p,)
        OLS coefficient vector.

    Returns
    -------
    se : ndarray of shape (p,)
        HC2 standard errors for each coefficient.
    """
    residuals = y - X @ beta
    Q, R = np.linalg.qr(X)
    h = np.sum(Q**2, axis=1)
    adjusted_resid_sq = residuals**2 / (1 - h)
    XtX_inv = np.linalg.inv(R.T @ R)
    meat = X.T @ (X * adjusted_resid_sq[:, np.newaxis])
    cov = XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(cov))


class CupacExperiment:
    """Analyze an A/B test with CUPAC or MLRATE variance reduction.

    CUPAC fits OLS on control-group pre-experiment covariates and adjusts
    outcomes via CUPED.  MLRATE generalises this by accepting any
    scikit-learn-compatible estimator and using K-fold cross-fitting so
    flexible models produce valid inference without overfitting bias.

    Both methods estimate the treatment effect with HC2 robust standard
    errors.

    Parameters
    ----------
    data : DataFrame
        Per-user data with outcome, treatment indicator, and covariate columns.
        Accepts pandas or polars DataFrames.
    outcome_col : str
        Name of the binary (0/1) outcome column.
    treatment_col : str
        Name of the column indicating group assignment.
    covariate_cols : list of str
        Names of pre-experiment covariate columns. All columns must be
        numeric. Nominal categorical features should be one-hot encoded
        before being passed in, since OLS interprets numeric values as
        continuous and will impose a spurious ordinal relationship.
    control_label : str or int
        Value in ``treatment_col`` identifying control-group users.
    treatment_label : str or int
        Value in ``treatment_col`` identifying treatment-group users.
    experiment_name : str
        Display name for the experiment.
    metric_name : str
        Display name for the outcome metric.
    method : str
        Adjustment method: ``"cupac"`` or ``"mlrate"``.
    estimator : object or None
        A scikit-learn-compatible estimator with ``fit`` and ``predict``
        methods.  Required when ``method="mlrate"``, ignored otherwise.
    n_folds : int
        Number of cross-fitting folds for MLRATE.  Ignored when
        ``method="cupac"``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        covariate_cols: list[str],
        control_label: str | int = 0,
        treatment_label: str | int = 1,
        experiment_name: str = "CUPAC Experiment",
        metric_name: str = "outcome",
        method: str = "cupac",
        estimator: Any = None,
        n_folds: int = 5,
    ) -> None:
        try:
            import polars as pl

            if isinstance(data, pl.DataFrame):  # type: ignore[unreachable]
                data = data.to_pandas()  # type: ignore[unreachable]
        except ImportError:
            pass

        self._validate_inputs(data, outcome_col, treatment_col, covariate_cols, control_label, treatment_label)

        self.data = data
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.covariate_cols = covariate_cols
        self.control_label = control_label
        self.treatment_label = treatment_label
        self.experiment_name = experiment_name
        self.metric_name = metric_name

        method = method.casefold()
        if method not in ("cupac", "mlrate"):
            raise NotImplementedError(f"Method {method!r} is not supported. Use 'cupac' or 'mlrate'.")
        if method == "mlrate":
            if estimator is None:
                raise ValueError("estimator is required when method='mlrate'")
            if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
                raise ValueError("estimator must have fit() and predict() methods")
        self.method = method
        self.estimator = estimator
        self.n_folds = n_folds

        self._results: dict[str, Any] | None = None

    @staticmethod
    def _validate_inputs(
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        covariate_cols: list[str],
        control_label: str | int,
        treatment_label: str | int,
    ) -> None:
        """Validate constructor inputs."""
        if not covariate_cols:
            raise ValueError("covariate_cols must not be empty. CUPAC and MLRATE require at least one covariate.")

        required_cols = [outcome_col, treatment_col, *covariate_cols]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        labels = set(data[treatment_col].unique())
        expected = {control_label, treatment_label}
        if labels != expected:
            raise ValueError(f"Expected treatment column to contain {expected}, got {labels}")

        outcome_vals = set(data[outcome_col].unique())
        if not outcome_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Outcome column must be binary (0/1), got values {outcome_vals}")

        for col in covariate_cols:
            if not np.issubdtype(data[col].dtype.type, np.number):
                raise ValueError(
                    f"Covariate column {col!r} must be numeric, got {data[col].dtype}. "
                    f"Nominal categorical features should be one-hot encoded before being passed in."
                )

    def _cross_fit_predictions(
        self,
        covariates: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """K-fold cross-fitted predictions for MLRATE.

        Each unit's prediction comes from a model trained on all other
        folds, ensuring the prediction is independent of the unit's own
        outcome.

        Parameters
        ----------
        covariates : ndarray of shape (n, p)
            Pre-experiment covariates.
        y : ndarray of shape (n,)
            Outcome vector.

        Returns
        -------
        y_hat : ndarray of shape (n,)
            Out-of-fold predictions.
        """
        n = len(y)
        indices = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        y_hat = np.empty(n, dtype=float)
        for fold_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[fold_idx] = False

            model = copy.deepcopy(self.estimator)
            model.fit(covariates[train_mask], y[train_mask])
            if hasattr(model, "predict_proba"):
                y_hat[fold_idx] = model.predict_proba(covariates[fold_idx])[:, 1]
            else:
                y_hat[fold_idx] = model.predict(covariates[fold_idx])

        return y_hat

    def fit(self) -> CupacExperiment:
        """Run the CUPAC or MLRATE analysis pipeline.

        Returns
        -------
        self
            For method chaining.
        """
        y = self.data[self.outcome_col].to_numpy(dtype=float)
        is_control = (self.data[self.treatment_col] == self.control_label).to_numpy()
        is_treatment = ~is_control
        covariates = self.data[self.covariate_cols].to_numpy(dtype=float)

        n_ctrl = int(is_control.sum())
        n_treat = int(is_treatment.sum())

        if self.method == "mlrate":
            y_hat = self._cross_fit_predictions(covariates, y)
        else:
            X_ctrl = np.column_stack([np.ones(n_ctrl), covariates[is_control]])
            y_ctrl = y[is_control]
            beta = _ols_fit(X_ctrl, y_ctrl)
            X_all = np.column_stack([np.ones(len(y)), covariates])
            y_hat = X_all @ beta

        # CUPED adjustment
        y_hat_var = np.var(y_hat, ddof=1)
        if y_hat_var > 1e-12:
            theta = np.cov(y, y_hat, ddof=1)[0, 1] / y_hat_var
            y_adj = y - theta * (y_hat - np.mean(y_hat))
        else:
            y_adj = y.copy()
            theta = 0.0

        # Treatment effect (adjusted and unadjusted)
        tau_hat = float(np.mean(y_adj[is_treatment]) - np.mean(y_adj[is_control]))
        tau_unadj = float(np.mean(y[is_treatment]) - np.mean(y[is_control]))

        # HC2 robust SEs via full regression
        treatment_indicator = is_treatment.astype(float)
        X_full = np.column_stack([np.ones(len(y_adj)), treatment_indicator, covariates])
        beta_full = _ols_fit(X_full, y_adj)
        se_full = _hc2_standard_errors(X_full, y_adj, beta_full)
        se_tau = float(se_full[1])

        # Unadjusted SE for comparison
        se_unadj = float(np.sqrt(np.var(y[is_control], ddof=1) / n_ctrl + np.var(y[is_treatment], ddof=1) / n_treat))

        # Inference
        z_stat = tau_hat / se_tau if se_tau > 0 else 0.0
        p_value = float(2 * ss.norm.sf(abs(z_stat)))

        # Variance reduction
        var_raw = np.var(y, ddof=1)
        var_adj = np.var(y_adj, ddof=1)
        r_squared = float(1 - var_adj / var_raw) if var_raw > 1e-12 else 0.0

        self._results = {
            "ate": tau_hat,
            "ate_unadjusted": tau_unadj,
            "se": se_tau,
            "se_unadjusted": se_unadj,
            "z_stat": z_stat,
            "p_value": p_value,
            "r_squared": r_squared,
            "theta": theta,
            "n_control": n_ctrl,
            "n_treatment": n_treat,
        }
        return self

    def _check_fitted(self) -> dict[str, Any]:
        """Return results dict, raising if fit() has not been called."""
        if self._results is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._results

    @property
    def ate(self) -> float:
        """Adjusted average treatment effect."""
        return self._check_fitted()["ate"]

    @property
    def se(self) -> float:
        """HC2 robust standard error of the treatment effect."""
        return self._check_fitted()["se"]

    @property
    def p_value(self) -> float:
        """Two-sided p-value."""
        return self._check_fitted()["p_value"]

    @property
    def variance_reduction(self) -> float:
        """R-squared: fraction of variance explained by the covariates."""
        return self._check_fitted()["r_squared"]

    def summary(self) -> dict[str, Any]:
        """Return results as a dict for programmatic access.

        Returns
        -------
        dict
            Keys: ``ate``, ``ate_unadjusted``, ``se``, ``se_unadjusted``,
            ``z_stat``, ``p_value``, ``r_squared``, ``theta``,
            ``n_control``, ``n_treatment``.
        """
        return dict(self._check_fitted())

    def analyze(self, alpha: float = 0.05) -> str:
        """Run the analysis and return a formatted results table.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        str
            Grid-formatted table of results.
        """
        if self._results is None:
            self.fit()
        results = self._check_fitted()

        z_crit = float(ss.norm.ppf(1 - alpha / 2))
        ci_lower = results["ate"] - z_crit * results["se"]
        ci_upper = results["ate"] + z_crit * results["se"]

        str_pvalue = f"{results['p_value']}" if results["p_value"] >= alpha else f"{results['p_value']}*"

        from tabulate import tabulate

        table = [
            ["Experiment", self.experiment_name],
            ["Metric", self.metric_name],
            ["Method", "MLRATE" if self.method == "mlrate" else "CUPAC"],
            ["N (control)", f"{results['n_control']:,}"],
            ["N (treatment)", f"{results['n_treatment']:,}"],
            ["Unadj. ATE", f"{results['ate_unadjusted']:.4%}"],
            ["Adjusted ATE", f"{results['ate']:.4%}"],
            ["Std. Error", f"{results['se']:.4%}"],
            ["p-value", str_pvalue],
            ["CI Lower **", f"{ci_lower:.4%}"],
            ["CI Upper **", f"{ci_upper:.4%}"],
            ["Variance Reduction", f"{results['r_squared']:.1%}"],
        ]
        return_string: str = tabulate(table, headers=["Metric", "Value"], tablefmt="grid")
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def plot(self, color: str | dict[str, Any] | list[Any] | None = None) -> None:
        """Plot unadjusted vs adjusted estimates with confidence intervals.

        Parameters
        ----------
        color : str, list, dict, or None, optional
            Color specification. Supports colorblind-friendly palette names
            (e.g. ``"ibm"``), a list of colors, or None for Plotly defaults.
        """
        if self._results is None:
            self.fit()
        results = self._check_fitted()

        z_crit = float(ss.norm.ppf(0.975))
        unadj_ci = z_crit * results["se_unadjusted"]
        adj_ci = z_crit * results["se"]

        plot_color = resolve_plot_color(color) or ["#636EFA", "#EF553B"]
        c_unadj = plot_color[0] if isinstance(plot_color, list) else list(plot_color.values())[0]
        c_adj = plot_color[1] if isinstance(plot_color, list) else list(plot_color.values())[1]

        fig = go.Figure()
        adj_label = "Adjusted (MLRATE)" if self.method == "mlrate" else "Adjusted (CUPAC)"
        for label, ate, ci_half, c in [
            ("Unadjusted", results["ate_unadjusted"], unadj_ci, c_unadj),
            (adj_label, results["ate"], adj_ci, c_adj),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[ate],
                    y=[label],
                    marker={"symbol": "diamond", "size": 12.5, "color": c},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_half],
                        "arrayminus": [ci_half],
                        "visible": True,
                        "color": c,
                    },
                    name=label,
                )
            )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"{self.experiment_name}: Unadjusted vs {'MLRATE' if self.method == 'mlrate' else 'CUPAC'}-Adjusted",
            xaxis_title="Treatment Effect",
            xaxis_tickformat=",.2%",
            template="plotly_white",
            yaxis={"autorange": "reversed"},
        )
        fig.show()


def cupac_adjusted_power(
    r_squared: float,
    power_func: Callable[..., float] = score_power,
) -> Callable[..., float]:
    """Return a power function that accounts for CUPAC variance reduction.

    Parameters
    ----------
    r_squared : float
        Fraction of outcome variance explained by covariates (0 to 1).
    power_func : callable
        Underlying power function with signature ``(n, p_null, p_alt, alpha)``.
        Defaults to :func:`~ab_test.frequentist_binomial.power_calculations.score_power`.

    Returns
    -------
    callable
        A power function with the same signature as ``power_func`` but with
        effective sample sizes inflated by ``1 / (1 - r_squared)``.
    """
    if not 0 <= r_squared < 1:
        raise ValueError(f"r_squared must be in [0, 1), got {r_squared}")

    def adjusted(
        n: np.ndarray[Any, Any] | list[Any],
        p_null: np.ndarray[Any, Any] | list[Any],
        p_alt: np.ndarray[Any, Any] | list[Any],
        alpha: float = 0.05,
    ) -> float:
        n_eff = [ni / (1 - r_squared) for ni in n]
        return power_func(n_eff, p_null, p_alt, alpha=alpha)

    return adjusted


def cupac_minimum_detectable_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    r_squared: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    drop: bool = False,
    lift: str = "relative",
) -> float:
    """Minimum detectable lift accounting for CUPAC variance reduction.

    Parameters
    ----------
    group_sizes : array_like
        Number of experimental units in each group.
    baseline : float
        Baseline success rate.
    r_squared : float
        Fraction of outcome variance explained by covariates (0 to 1).
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    drop : bool
        If True, return the minimum detectable drop. Defaults to False.
    lift : str
        ``"relative"`` or ``"absolute"``.

    Returns
    -------
    float
        Minimum detectable lift (or drop).
    """
    return minimum_detectable_lift(
        group_sizes,
        baseline,
        alpha=alpha,
        beta=beta,
        null_lift=null_lift,
        power=cupac_adjusted_power(r_squared),
        drop=drop,
        lift=lift,
    )


def cupac_required_sample_size(
    baseline: float,
    alt_lift: float,
    r_squared: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    null_lift: float = 0.0,
    lift: str = "relative",
) -> int:
    """Calculate the required sample size accounting for CUPAC variance reduction.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    r_squared : float
        Fraction of outcome variance explained by covariates (0 to 1).
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to 50/50.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : str
        ``"relative"`` or ``"absolute"``.

    Returns
    -------
    int
        Minimum total sample size.
    """
    return required_sample_size(
        baseline,
        alt_lift,
        alpha=alpha,
        beta=beta,
        group_proportions=group_proportions,
        null_lift=null_lift,
        power=cupac_adjusted_power(r_squared),
        lift=lift,
    )


def plot_cupac_power_curve(
    baseline: float,
    alt_lift: float,
    r_squared: float,
    alpha: float = 0.05,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot statistical power as a function of total sample size with CUPAC adjustment.

    Overlays the unadjusted power curve so that the variance-reduction
    benefit is visible.

    Parameters
    ----------
    baseline : float
        Baseline success rate associated with the first experiment group.
    alt_lift : float
        Lift associated with the alternative hypothesis.
    r_squared : float
        Fraction of outcome variance explained by covariates (0 to 1).
    alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
    null_lift : float, optional
        Lift associated with the null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}, optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to ``"relative"``.
    sample_sizes : array_like or None, optional
        Explicit total sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    group_proportions : array_like or None, optional
        Fraction of experimental units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 100.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with total sample size on the x-axis and
        power on the y-axis, showing both adjusted and unadjusted curves.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    adjusted_power = cupac_adjusted_power(r_squared)

    if sample_sizes is None:
        target_ss = cupac_required_sample_size(
            baseline,
            alt_lift,
            r_squared,
            alpha=alpha,
            beta=0.2,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        max_ss = int(target_ss * 2)
        sample_sizes = np.linspace(max(20, max_ss // n_points), max_ss, n_points, dtype=int)

    adjusted_powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=adjusted_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    unadjusted_powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=score_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=adjusted_powers,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"CUPAC-adjusted (R²={r_squared:.2f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=unadjusted_powers,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dot"},
            name="Unadjusted",
        )
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="80% power",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Power Curve (CUPAC-adjusted)",
        xaxis_title="Total sample size",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def plot_cupac_sensitivity_curve(
    baseline: float,
    r_squared: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot minimum detectable lift as a function of total sample size with CUPAC adjustment.

    Overlays the unadjusted sensitivity curve so that the
    variance-reduction benefit is visible.

    Parameters
    ----------
    baseline : float
        Baseline success rate associated with the first experiment group.
    r_squared : float
        Fraction of outcome variance explained by covariates (0 to 1).
    alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
    beta : float, optional
        Type-II error rate threshold (1 - power). Defaults to 0.2.
    null_lift : float, optional
        Lift associated with the null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}, optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to ``"relative"``.
    sample_sizes : array_like or None, optional
        Explicit total sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    group_proportions : array_like or None, optional
        Fraction of experimental units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 100.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with total sample size on the x-axis and
        minimum detectable lift on the y-axis, showing both adjusted and
        unadjusted curves.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    adjusted_power = cupac_adjusted_power(r_squared)

    if sample_sizes is None:
        target_ss = cupac_required_sample_size(
            baseline,
            alt_lift=0.05,
            r_squared=r_squared,
            alpha=alpha,
            beta=beta,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        min_ss = max(20, target_ss // 10)
        max_ss = target_ss * 5
        sample_sizes = np.linspace(min_ss, max_ss, n_points, dtype=int)

    adjusted_mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=adjusted_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    unadjusted_mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=score_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=adjusted_mdls,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"CUPAC-adjusted (R²={r_squared:.2f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=unadjusted_mdls,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dot"},
            name="Unadjusted",
        )
    )

    y_label = f"Minimum detectable {lift} lift"
    fig.update_layout(
        title="Sensitivity Curve (CUPAC-adjusted)",
        xaxis_title="Total sample size",
        yaxis_title=y_label,
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
