"""Group Sequential Testing for A/B tests.

Provides alpha spending functions, pre-computed z-scale stopping boundaries,
and power/sample-size calculations for group sequential designs. The boundary
computation uses recursive numerical integration following Jennison and
Turnbull (1999), and the alpha spending approach follows Lan and DeMets (1983).

References
----------
Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for
clinical trials. *Biometrika*, 70(3), 659-663.

Jennison, C., & Turnbull, B. W. (1999). *Group Sequential Methods with
Applications to Clinical Trials*. Chapman & Hall/CRC.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import scipy.stats as ss
from scipy.optimize import brentq

from ab_test.frequentist_binomial.power_calculations import (
    abtest_power,
    minimum_detectable_lift,
    required_sample_size,
    score_power,
)
from ab_test.frequentist_binomial.utils import mle_under_alternative, mle_under_null, validate_two_group

__all__ = [
    "obrien_fleming_spending",
    "pocock_spending",
    "power_spending",
    "GroupSequentialDesign",
    "gst_adjusted_power",
    "gst_minimum_detectable_lift",
    "gst_required_sample_size",
    "plot_gst_boundaries",
    "plot_gst_power_curve",
    "plot_gst_sensitivity_curve",
]

_N_GRID = 2001
_Z_MAX = 8.0
_SQRT_2PI = math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Alpha spending functions
# ---------------------------------------------------------------------------


def obrien_fleming_spending(t: float, alpha: float = 0.05) -> float:
    """Lan-DeMets O'Brien-Fleming alpha spending function.

    Spends very little alpha at early looks and concentrates spending near the
    final analysis. Use when early stopping should only occur for overwhelming
    effects.

    Parameters
    ----------
    t : float
        Information fraction in [0, 1].
    alpha : float
        Overall type-I error rate.

    Returns
    -------
    float
        Cumulative alpha spent up to information fraction ``t``.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    z = float(ss.norm.isf(alpha / 2))
    return float(2 * (1 - ss.norm.cdf(z / math.sqrt(t))))


def pocock_spending(t: float, alpha: float = 0.05) -> float:
    """Lan-DeMets Pocock alpha spending function.

    Spends alpha more uniformly across looks, yielding approximately equal
    boundaries at each analysis.

    Parameters
    ----------
    t : float
        Information fraction in [0, 1].
    alpha : float
        Overall type-I error rate.

    Returns
    -------
    float
        Cumulative alpha spent up to information fraction ``t``.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return alpha * math.log(1 + (math.e - 1) * t)


def power_spending(t: float, alpha: float = 0.05, *, rho: float = 1.0) -> float:
    """Power family alpha spending function.

    A flexible family parameterized by ``rho``. ``rho=1`` gives linear
    spending (Pocock-like); larger ``rho`` concentrates spending at later
    looks (O'Brien-Fleming-like).

    To use a custom ``rho`` with :class:`GroupSequentialDesign`, pass
    ``functools.partial(power_spending, rho=3)``.

    Parameters
    ----------
    t : float
        Information fraction in [0, 1].
    alpha : float
        Overall type-I error rate.
    rho : float
        Shape parameter. Must be positive.

    Returns
    -------
    float
        Cumulative alpha spent up to information fraction ``t``.
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return alpha * t**rho


# ---------------------------------------------------------------------------
# Private helpers: grid-based numerical integration
# ---------------------------------------------------------------------------


def _propagate_density(
    density: np.ndarray[Any, Any],
    grid: np.ndarray[Any, Any],
    dz: float,
    t_prev: float,
    t_curr: float,
    drift: float = 0.0,
) -> np.ndarray[Any, Any]:
    """Propagate the continuation density forward one analysis step.

    Uses the transition kernel for the joint distribution of sequential
    z-statistics under drift ``drift``.
    """
    sigma = math.sqrt((t_curr - t_prev) / t_curr)
    scale = math.sqrt(t_prev / t_curr)
    mu_shift = drift * (t_curr - t_prev) / math.sqrt(t_curr)
    means = grid * scale + mu_shift
    diff = grid[:, np.newaxis] - means[np.newaxis, :]
    kernel = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * _SQRT_2PI)
    return kernel @ (density * dz)


def _inside_mass(
    density: np.ndarray[Any, Any],
    grid: np.ndarray[Any, Any],
    boundary: float,
    sided: str,
) -> float:
    """Mass of density inside the continuation region."""
    if sided == "two":
        idx = np.abs(grid) < boundary
    else:
        idx = grid < boundary
    return float(np.trapezoid(density[idx], grid[idx]))


def _restrict_density(
    density: np.ndarray[Any, Any],
    grid: np.ndarray[Any, Any],
    boundary: float,
    sided: str,
) -> np.ndarray[Any, Any]:
    """Zero out density outside the continuation region."""
    result = density.copy()
    if sided == "two":
        result[np.abs(grid) >= boundary] = 0.0
    else:
        result[grid >= boundary] = 0.0
    return result


def _boundary_objective(
    c: float,
    new_density: np.ndarray[Any, Any],
    grid: np.ndarray[Any, Any],
    total_mass: float,
    target: float,
    sided: str,
) -> float:
    """Objective for root-finding: exit probability minus target spending."""
    return total_mass - _inside_mass(new_density, grid, c, sided) - target


def _compute_boundaries(
    n_analyses: int,
    alpha: float,
    spending_function: Callable[[float, float], float],
    info_fractions: np.ndarray[Any, Any],
    sided: str,
) -> np.ndarray[Any, Any]:
    """Compute z-scale boundaries via recursive integration and root-finding."""
    grid = np.linspace(-_Z_MAX, _Z_MAX, _N_GRID)
    dz = grid[1] - grid[0]

    boundaries = np.zeros(n_analyses)
    cum_spend = np.array([spending_function(t, alpha) for t in info_fractions])
    delta_spend = np.diff(np.concatenate(([0.0], cum_spend)))

    if sided == "two":
        boundaries[0] = float(ss.norm.isf(delta_spend[0] / 2))
    else:
        boundaries[0] = float(ss.norm.isf(delta_spend[0]))

    density = ss.norm.pdf(grid)
    density = _restrict_density(density, grid, boundaries[0], sided)

    for k in range(1, n_analyses):
        new_density = _propagate_density(density, grid, dz, info_fractions[k - 1], info_fractions[k])
        total_mass = float(np.trapezoid(new_density, grid))

        boundaries[k] = brentq(
            _boundary_objective,
            0.1,
            _Z_MAX - 0.1,
            args=(new_density, grid, total_mass, delta_spend[k], sided),
        )

        density = _restrict_density(new_density, grid, boundaries[k], sided)

    return boundaries


def _compute_exit_probabilities(
    boundaries: np.ndarray[Any, Any],
    info_fractions: np.ndarray[Any, Any],
    drift: float,
    sided: str,
) -> np.ndarray[Any, Any]:
    """Exit probabilities at each look under a given drift.

    The sum of exit probabilities is the overall rejection probability (power
    when drift > 0, type-I error when drift = 0).
    """
    grid = np.linspace(-_Z_MAX, _Z_MAX, _N_GRID)
    dz = grid[1] - grid[0]

    n_analyses = len(boundaries)
    exit_probs = np.zeros(n_analyses)

    density = ss.norm.pdf(grid, loc=drift * math.sqrt(info_fractions[0]))
    total_mass = float(np.trapezoid(density, grid))
    inside = _inside_mass(density, grid, boundaries[0], sided)
    exit_probs[0] = total_mass - inside
    density = _restrict_density(density, grid, boundaries[0], sided)

    for k in range(1, n_analyses):
        new_density = _propagate_density(density, grid, dz, info_fractions[k - 1], info_fractions[k], drift)
        total_mass = float(np.trapezoid(new_density, grid))
        inside = _inside_mass(new_density, grid, boundaries[k], sided)
        exit_probs[k] = total_mass - inside
        density = _restrict_density(new_density, grid, boundaries[k], sided)

    return exit_probs


# ---------------------------------------------------------------------------
# GroupSequentialDesign
# ---------------------------------------------------------------------------


class GroupSequentialDesign:
    """Pre-computed group sequential testing boundaries.

    Computes z-scale stopping boundaries for a design with ``n_analyses``
    planned analyses (interim + final) using an alpha spending function.
    The boundaries can then be used to test data at each analysis and to
    compute the power of the design.

    Parameters
    ----------
    n_analyses : int
        Number of planned analyses (interim + final), at least 1.
    alpha : float
        Overall two-sided type-I error rate. Defaults to 0.05.
    spending_function : callable
        Alpha spending function with signature ``(t, alpha) -> float``.
        Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions at each analysis, strictly increasing with the
        last element equal to 1.0. Defaults to equally spaced fractions.
    sided : {"two", "one"}
        Whether to use two-sided or one-sided boundaries. Defaults to
        ``"two"``.

    Examples
    --------
    >>> design = GroupSequentialDesign(n_analyses=3, alpha=0.05)
    >>> print(design.summary())  # doctest: +SKIP
    >>> design.test([5000, 5000], [480, 550], look=2)
    """

    def __init__(
        self,
        n_analyses: int,
        alpha: float = 0.05,
        spending_function: Callable[[float, float], float] = obrien_fleming_spending,
        info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
        sided: str = "two",
    ) -> None:
        if n_analyses < 1:
            raise ValueError("n_analyses must be at least 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if sided not in ("one", "two"):
            raise ValueError("sided must be 'one' or 'two'")

        if info_fractions is None:
            info_fractions_arr = np.array([k / n_analyses for k in range(1, n_analyses + 1)])
        else:
            info_fractions_arr = np.asarray(info_fractions, dtype=float)

        if len(info_fractions_arr) != n_analyses:
            raise ValueError(f"info_fractions must have length {n_analyses}, got {len(info_fractions_arr)}")
        if n_analyses > 1 and not np.all(np.diff(info_fractions_arr) > 0):
            raise ValueError("info_fractions must be strictly increasing")
        if abs(info_fractions_arr[-1] - 1.0) > 1e-10:
            raise ValueError("info_fractions must end at 1.0")

        self._n_analyses = n_analyses
        self._alpha = alpha
        self._spending_function = spending_function
        self._info_fractions = info_fractions_arr
        self._sided = sided

        self._boundaries = _compute_boundaries(n_analyses, alpha, spending_function, info_fractions_arr, sided)

        cum = np.array([spending_function(t, alpha) for t in info_fractions_arr])
        self._nominal_alpha = cum
        self._incremental_alpha = np.diff(np.concatenate(([0.0], cum)))

    @property
    def boundaries(self) -> np.ndarray[Any, Any]:
        """Z-scale critical values at each analysis."""
        return self._boundaries.copy()

    @property
    def nominal_alpha(self) -> np.ndarray[Any, Any]:
        """Cumulative alpha spent at each analysis."""
        return self._nominal_alpha.copy()

    @property
    def incremental_alpha(self) -> np.ndarray[Any, Any]:
        """Alpha spent at each individual analysis."""
        return self._incremental_alpha.copy()

    def test(
        self,
        trials: np.ndarray[Any, Any] | list[Any],
        successes: np.ndarray[Any, Any] | list[Any],
        look: int,
        null_lift: float = 0.0,
        lift: str = "relative",
    ) -> bool:
        """Test whether the boundary is crossed at the given look.

        Computes a score-test z-statistic from the data and compares it
        against the pre-computed boundary.

        Parameters
        ----------
        trials : array_like
            Number of trials in each group (cumulative up to this look).
        successes : array_like
            Number of successes in each group (cumulative up to this look).
        look : int
            Analysis number, 1-indexed (from 1 to ``n_analyses``).
        null_lift : float
            Lift under the null hypothesis. Defaults to 0.0.
        lift : {"relative", "absolute"}
            Interpretation of ``null_lift``.

        Returns
        -------
        bool
            True if the test statistic exceeds the boundary at this look.
        """
        if look < 1 or look > self._n_analyses:
            raise ValueError(f"look must be between 1 and {self._n_analyses}, got {look}")

        validate_two_group(trials, successes, null_lift, lift)
        trials_arr = np.asarray(trials, dtype=float)
        successes_arr = np.asarray(successes, dtype=float)

        p0 = mle_under_null(trials_arr, successes_arr, null_lift=null_lift, lift=lift)
        p1 = mle_under_alternative(trials_arr, successes_arr)

        d = (p1[1] - p1[0]) - (p0[1] - p0[0])
        sigma2 = p0[0] * (1 - p0[0]) / trials_arr[0] + p0[1] * (1 - p0[1]) / trials_arr[1]

        if sigma2 <= 1e-24:
            return False

        z_stat = d / math.sqrt(sigma2)
        boundary = self._boundaries[look - 1]

        if self._sided == "two":
            return bool(abs(z_stat) >= boundary)
        return bool(z_stat >= boundary)

    def power(
        self,
        n: np.ndarray[Any, Any] | list[Any],
        p_null: np.ndarray[Any, Any] | list[Any],
        p_alt: np.ndarray[Any, Any] | list[Any],
        alpha: float = 0.05,
    ) -> float:
        """Power of the group sequential design under a specified alternative.

        Computes the probability of rejecting at any analysis under the given
        alternative hypothesis.

        Parameters
        ----------
        n : array_like
            Sample sizes per group at the final analysis.
        p_null : array_like
            Success probabilities under the null.
        p_alt : array_like
            Success probabilities under the alternative.
        alpha : float
            Accepted for signature compatibility with :func:`score_power` but
            ignored — the boundaries already encode the type-I error rate.

        Returns
        -------
        float
            Overall power (probability of rejection at any look).
        """
        n_arr = np.asarray(n, dtype=float)
        p_null_arr = np.asarray(p_null, dtype=float)
        p_alt_arr = np.asarray(p_alt, dtype=float)

        d = (p_alt_arr[1] - p_alt_arr[0]) - (p_null_arr[1] - p_null_arr[0])
        sigma2 = p_null_arr[0] * (1 - p_null_arr[0]) / n_arr[0] + p_null_arr[1] * (1 - p_null_arr[1]) / n_arr[1]

        if sigma2 <= 1e-24:
            return 0.0

        theta = float(d / math.sqrt(sigma2))
        exit_probs = _compute_exit_probabilities(self._boundaries, self._info_fractions, theta, self._sided)
        return float(np.sum(exit_probs))

    def summary(self, alpha: float = 0.05) -> str:
        """Tabulated display of boundaries and alpha spending at each look.

        Parameters
        ----------
        alpha : float
            Ignored. Present for API consistency.

        Returns
        -------
        str
            Formatted table.
        """
        from tabulate import tabulate

        rows = []
        for k in range(self._n_analyses):
            rows.append(
                [
                    k + 1,
                    f"{self._info_fractions[k]:.4f}",
                    f"{self._boundaries[k]:.4f}",
                    f"{self._nominal_alpha[k]:.6f}",
                    f"{self._incremental_alpha[k]:.6f}",
                ]
            )
        headers = ["Look", "Info Fraction", "Z Boundary", "Cum. Alpha", "Incr. Alpha"]
        result: str = tabulate(rows, headers=headers, tablefmt="grid")
        result += f"\n\nDesign: {self._n_analyses} analyses, alpha={self._alpha}, sided={self._sided}"
        return result

    def plot_boundaries(self) -> go.Figure:
        """Plot z-boundaries across analyses.

        Returns
        -------
        go.Figure
            An interactive Plotly figure.
        """
        looks = list(range(1, self._n_analyses + 1))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=looks,
                y=list(self._boundaries),
                mode="lines+markers",
                marker={"size": 8, "symbol": "diamond"},
                line={"color": "#636EFA", "width": 2},
                name="Z Boundary",
            )
        )
        if self._sided == "two":
            fig.add_trace(
                go.Scatter(
                    x=looks,
                    y=list(-self._boundaries),
                    mode="lines+markers",
                    marker={"size": 8, "symbol": "diamond"},
                    line={"color": "#636EFA", "width": 2, "dash": "dash"},
                    name="Z Boundary (lower)",
                )
            )

        fig.update_layout(
            title="Group Sequential Boundaries",
            xaxis_title="Analysis",
            yaxis_title="Z Boundary",
            template="plotly_white",
            xaxis={"dtick": 1},
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        return fig


# ---------------------------------------------------------------------------
# Power / sample-size wrappers
# ---------------------------------------------------------------------------


def gst_adjusted_power(
    n_analyses: int,
    spending_function: Callable[[float, float], float] = obrien_fleming_spending,
    info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
    sided: str = "two",
    power_func: Callable[..., float] = score_power,
) -> Callable[..., float]:
    """Return a power callable adjusted for group sequential monitoring.

    The returned function has the same signature as :func:`score_power` and
    can be passed to :func:`~ab_test.frequentist_binomial.power_calculations.abtest_power`,
    :func:`~ab_test.frequentist_binomial.power_calculations.required_sample_size`, etc.

    Parameters
    ----------
    n_analyses : int
        Number of planned analyses.
    spending_function : callable
        Alpha spending function. Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions. Defaults to equally spaced.
    sided : {"two", "one"}
        Sidedness. Defaults to ``"two"``.
    power_func : callable
        Ignored. Present for API parity with
        :func:`~ab_test.frequentist_binomial.cupac.cupac_adjusted_power`.

    Returns
    -------
    callable
        Power function with signature ``(n, p_null, p_alt, alpha) -> float``.
    """
    cache: dict[float, GroupSequentialDesign] = {}

    def adjusted(
        n: np.ndarray[Any, Any] | list[Any],
        p_null: np.ndarray[Any, Any] | list[Any],
        p_alt: np.ndarray[Any, Any] | list[Any],
        alpha: float = 0.05,
    ) -> float:
        if alpha not in cache:
            cache[alpha] = GroupSequentialDesign(
                n_analyses,
                alpha=alpha,
                spending_function=spending_function,
                info_fractions=info_fractions,
                sided=sided,
            )
        return cache[alpha].power(n, p_null, p_alt, alpha)

    return adjusted


def gst_required_sample_size(
    baseline: float,
    alt_lift: float,
    n_analyses: int,
    spending_function: Callable[[float, float], float] = obrien_fleming_spending,
    info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
    sided: str = "two",
    alpha: float = 0.05,
    beta: float = 0.2,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    null_lift: float = 0.0,
    lift: str = "relative",
) -> int:
    """Calculate the required sample size for a group sequential design.

    Accounts for the stricter boundaries of the group sequential design,
    which inflate the required sample size relative to a fixed-horizon test.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    n_analyses : int
        Number of planned analyses.
    spending_function : callable
        Alpha spending function. Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions. Defaults to equally spaced.
    sided : {"two", "one"}
        Sidedness. Defaults to ``"two"``.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to ``[0.5, 0.5]``.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Interpretation of lifts. Defaults to ``"relative"``.

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
        power=gst_adjusted_power(n_analyses, spending_function, info_fractions, sided),
        lift=lift,
    )


def gst_minimum_detectable_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    n_analyses: int,
    spending_function: Callable[[float, float], float] = obrien_fleming_spending,
    info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
    sided: str = "two",
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    drop: bool = False,
    lift: str = "relative",
) -> float:
    """Minimum detectable lift for a group sequential design.

    Parameters
    ----------
    group_sizes : array_like
        Number of units in each group.
    baseline : float
        Baseline success rate.
    n_analyses : int
        Number of planned analyses.
    spending_function : callable
        Alpha spending function. Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions. Defaults to equally spaced.
    sided : {"two", "one"}
        Sidedness. Defaults to ``"two"``.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    drop : bool
        If True, compute the minimum detectable drop. Defaults to False.
    lift : {"relative", "absolute"}
        Interpretation of lifts. Defaults to ``"relative"``.

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
        power=gst_adjusted_power(n_analyses, spending_function, info_fractions, sided),
        drop=drop,
        lift=lift,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_gst_boundaries(design: GroupSequentialDesign) -> go.Figure:
    """Plot stopping boundaries for a group sequential design.

    Parameters
    ----------
    design : GroupSequentialDesign
        A pre-computed design.

    Returns
    -------
    go.Figure
        An interactive Plotly figure.
    """
    return design.plot_boundaries()


def plot_gst_power_curve(
    baseline: float,
    alt_lift: float,
    n_analyses: int,
    spending_function: Callable[[float, float], float] = obrien_fleming_spending,
    info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
    sided: str = "two",
    alpha: float = 0.05,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot power vs sample size for GST and fixed-horizon designs.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    n_analyses : int
        Number of planned analyses.
    spending_function : callable
        Alpha spending function. Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions. Defaults to equally spaced.
    sided : {"two", "one"}
        Sidedness. Defaults to ``"two"``.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Interpretation of lifts. Defaults to ``"relative"``.
    sample_sizes : array_like or None
        Explicit sample sizes. Auto-ranged when ``None``.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int
        Number of sample-size points when ``sample_sizes`` is ``None``.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with two traces.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    gst_power_func = gst_adjusted_power(n_analyses, spending_function, info_fractions, sided)

    if sample_sizes is None:
        target_ss = gst_required_sample_size(
            baseline,
            alt_lift,
            n_analyses,
            spending_function=spending_function,
            info_fractions=info_fractions,
            sided=sided,
            alpha=alpha,
            beta=0.2,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        max_ss = int(target_ss * 2)
        sample_sizes = np.linspace(max(20, max_ss // n_points), max_ss, n_points, dtype=int)

    gst_powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=gst_power_func,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    fixed_powers = [
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
            y=gst_powers,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"GST ({n_analyses} analyses)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=fixed_powers,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dash"},
            name="Fixed-horizon",
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
        title="Power Curve: GST vs Fixed-Horizon",
        xaxis_title="Total sample size",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def plot_gst_sensitivity_curve(
    baseline: float,
    n_analyses: int,
    spending_function: Callable[[float, float], float] = obrien_fleming_spending,
    info_fractions: np.ndarray[Any, Any] | list[float] | None = None,
    sided: str = "two",
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot minimum detectable lift vs sample size for GST and fixed-horizon.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    n_analyses : int
        Number of planned analyses.
    spending_function : callable
        Alpha spending function. Defaults to :func:`obrien_fleming_spending`.
    info_fractions : array_like or None
        Information fractions. Defaults to equally spaced.
    sided : {"two", "one"}
        Sidedness. Defaults to ``"two"``.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Interpretation of lifts. Defaults to ``"relative"``.
    sample_sizes : array_like or None
        Explicit sample sizes. Auto-ranged when ``None``.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int
        Number of sample-size points when ``sample_sizes`` is ``None``.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with two traces.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    gst_power_func = gst_adjusted_power(n_analyses, spending_function, info_fractions, sided)

    if sample_sizes is None:
        target_ss = gst_required_sample_size(
            baseline,
            alt_lift=0.05,
            n_analyses=n_analyses,
            spending_function=spending_function,
            info_fractions=info_fractions,
            sided=sided,
            alpha=alpha,
            beta=beta,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        min_ss = max(20, target_ss // 10)
        max_ss = target_ss * 5
        sample_sizes = np.linspace(min_ss, max_ss, n_points, dtype=int)

    gst_mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=gst_power_func,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    fixed_mdls = [
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
            y=gst_mdls,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"GST ({n_analyses} analyses)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=fixed_mdls,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dash"},
            name="Fixed-horizon",
        )
    )

    y_label = f"Minimum detectable {lift} lift"
    fig.update_layout(
        title="Sensitivity Curve: GST vs Fixed-Horizon",
        xaxis_title="Total sample size",
        yaxis_title=y_label,
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
