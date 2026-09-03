"""Validation of MLRATE against Guo et al. (2021) theoretical guarantees.

Tests the three core claims from "Machine Learning for Variance Reduction
in Online Experiments":

1. Cross-fitting eliminates the overfitting bias that in-sample predictions
   introduce when using flexible models (Section 3).
2. The cross-fitted estimator is asymptotically normal, so Wald confidence
   intervals achieve nominal coverage (Theorem 1).
3. Variance reduction scales with the predictive R-squared of the
   cross-fitted model, and flexible models capture more signal than OLS
   in nonlinear data-generating processes (Section 2).

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import copy

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from ab_test.frequentist_binomial.cupac import CupacExperiment  # noqa: E402


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------


def _generate_experiment(
    n_per_group=2000,
    treatment_effect=0.0,
    n_covariates=5,
    nonlinear=False,
    seed=42,
):
    """Synthetic experiment with a covariate-driven outcome probability.

    When ``nonlinear=True`` the outcome depends on absolute values,
    interactions, and thresholds that OLS cannot capture.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group

    covariates = rng.normal(0, 1, (n, n_covariates))
    group = np.array(["control"] * n_per_group + ["treatment"] * n_per_group)

    if nonlinear:
        signal = (
            0.08 * covariates[:, 0]
            + 0.06 * np.abs(covariates[:, 1])
            + 0.05 * covariates[:, 0] * covariates[:, 2]
            + 0.04 * (covariates[:, 3] > 0).astype(float)
        )
    else:
        signal = 0.08 * covariates[:, 0] + 0.06 * covariates[:, 1]

    prob = 0.15 + signal + treatment_effect * (group == "treatment")
    prob = np.clip(prob, 0.01, 0.99)
    outcome = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "group": group,
            "converted": outcome,
            **{f"cov_{i}": covariates[:, i] for i in range(n_covariates)},
        }
    )
    cov_cols = [f"cov_{i}" for i in range(n_covariates)]
    return df, cov_cols


def _ate_insample(df, cov_cols, estimator):
    """CUPED-adjusted ATE using in-sample (not cross-fitted) predictions.

    Replicates the MLRATE pipeline but trains and predicts on the same
    data, demonstrating the bias that cross-fitting prevents.
    """
    y = df["converted"].to_numpy(dtype=float)
    covariates = df[cov_cols].to_numpy(dtype=float)
    is_treatment = (df["group"] == "treatment").to_numpy()

    model = copy.deepcopy(estimator)
    model.fit(covariates, y)
    y_hat = model.predict(covariates)

    y_hat_var = np.var(y_hat, ddof=1)
    if y_hat_var > 1e-12:
        theta = np.cov(y, y_hat, ddof=1)[0, 1] / y_hat_var
        y_adj = y - theta * (y_hat - np.mean(y_hat))
    else:
        y_adj = y.copy()

    return float(np.mean(y_adj[is_treatment]) - np.mean(y_adj[~is_treatment]))


# ---------------------------------------------------------------------------
# 1. Cross-fitting necessity (Guo et al. Section 3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCrossFittingNecessity:
    """In-sample predictions from flexible models bias the ATE; cross-fitting fixes it."""

    @staticmethod
    def test_flexible_model_insample_is_biased():
        """A fully-grown tree memorises y, making theta ≈ 1 and attenuating the ATE toward zero."""
        n_sims = 300
        true_effect = 0.03
        tree = DecisionTreeRegressor(max_depth=None, random_state=0)

        ates_insample = []
        ates_crossfit = []

        for seed in range(n_sims):
            df, cov_cols = _generate_experiment(
                n_per_group=1000,
                treatment_effect=true_effect,
                nonlinear=True,
                seed=seed,
            )

            ates_insample.append(_ate_insample(df, cov_cols, tree))

            exp = CupacExperiment(
                df,
                "converted",
                "group",
                cov_cols,
                "control",
                "treatment",
                method="mlrate",
                estimator=DecisionTreeRegressor(max_depth=None, random_state=0),
            ).fit()
            ates_crossfit.append(exp.ate)

        insample_bias = abs(np.mean(ates_insample) - true_effect)
        crossfit_bias = abs(np.mean(ates_crossfit) - true_effect)

        assert crossfit_bias < insample_bias, (
            f"Cross-fitted bias ({crossfit_bias:.4f}) should be smaller than in-sample bias ({insample_bias:.4f})"
        )
        assert np.mean(ates_crossfit) == pytest.approx(true_effect, abs=0.005)

    @staticmethod
    def test_linear_model_insample_approximately_unbiased():
        """OLS has low capacity, so in-sample and cross-fitted should both be unbiased."""
        n_sims = 200
        true_effect = 0.03

        ates_insample = []
        ates_crossfit = []

        for seed in range(n_sims):
            df, cov_cols = _generate_experiment(
                n_per_group=1000,
                treatment_effect=true_effect,
                seed=seed,
            )
            ates_insample.append(_ate_insample(df, cov_cols, LinearRegression()))

            exp = CupacExperiment(
                df,
                "converted",
                "group",
                cov_cols,
                "control",
                "treatment",
                method="mlrate",
                estimator=LinearRegression(),
            ).fit()
            ates_crossfit.append(exp.ate)

        assert np.mean(ates_insample) == pytest.approx(true_effect, abs=0.005)
        assert np.mean(ates_crossfit) == pytest.approx(true_effect, abs=0.005)


# ---------------------------------------------------------------------------
# 2. Confidence interval coverage (Guo et al. Theorem 1)
# ---------------------------------------------------------------------------


def _coverage_rate(estimator, nonlinear, n_sims=500, n_per_group=1500, true_effect=0.02, alpha=0.05):
    """Fraction of simulations where the 95% CI covers the true effect."""
    z = ss.norm.ppf(1 - alpha / 2)
    covers = 0

    for seed in range(n_sims):
        df, cov_cols = _generate_experiment(
            n_per_group=n_per_group,
            treatment_effect=true_effect,
            nonlinear=nonlinear,
            seed=seed,
        )
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            cov_cols,
            "control",
            "treatment",
            method="mlrate",
            estimator=estimator,
        ).fit()

        ci_lo = exp.ate - z * exp.se
        ci_hi = exp.ate + z * exp.se
        if ci_lo <= true_effect <= ci_hi:
            covers += 1

    return covers / n_sims


@pytest.mark.slow
class TestConfidenceIntervalCoverage:
    """Wald CIs from cross-fitted MLRATE should achieve ≥ 92% coverage at the 95% nominal level."""

    @staticmethod
    def test_coverage_linear_model():
        coverage = _coverage_rate(LinearRegression(), nonlinear=False)
        assert 0.92 <= coverage <= 0.99, f"Coverage {coverage:.3f} outside [0.92, 0.99]"

    @staticmethod
    def test_coverage_flexible_model():
        coverage = _coverage_rate(
            DecisionTreeRegressor(max_depth=5, random_state=0),
            nonlinear=True,
        )
        assert 0.92 <= coverage <= 0.99, f"Coverage {coverage:.3f} outside [0.92, 0.99]"


# ---------------------------------------------------------------------------
# 3. Variance reduction (Guo et al. Section 2)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVarianceReduction:
    """Variance reduction should scale with model predictiveness."""

    @staticmethod
    def test_more_covariates_yield_greater_reduction():
        """All covariates should reduce variance more than a single weak one."""
        df, all_cov_cols = _generate_experiment(n_per_group=3000, nonlinear=True, seed=42)
        tree = DecisionTreeRegressor(max_depth=5, random_state=0)

        exp_weak = CupacExperiment(
            df,
            "converted",
            "group",
            [all_cov_cols[-1]],
            "control",
            "treatment",
            method="mlrate",
            estimator=tree,
        ).fit()

        exp_full = CupacExperiment(
            df,
            "converted",
            "group",
            all_cov_cols,
            "control",
            "treatment",
            method="mlrate",
            estimator=DecisionTreeRegressor(max_depth=5, random_state=0),
        ).fit()

        assert exp_full.variance_reduction > exp_weak.variance_reduction

    @staticmethod
    def test_variance_reduction_nonnegative_across_models():
        """Cross-fitted VR should be non-negative on average for any model complexity."""
        n_sims = 100
        estimators = {
            "linear": LinearRegression(),
            "shallow_tree": DecisionTreeRegressor(max_depth=3, random_state=0),
            "deep_tree": DecisionTreeRegressor(max_depth=None, random_state=0),
        }
        mean_vr = {}

        for name, est in estimators.items():
            vrs = []
            for seed in range(n_sims):
                df, cov_cols = _generate_experiment(n_per_group=2000, nonlinear=True, seed=seed)
                exp = CupacExperiment(
                    df,
                    "converted",
                    "group",
                    cov_cols,
                    "control",
                    "treatment",
                    method="mlrate",
                    estimator=est,
                ).fit()
                vrs.append(exp.variance_reduction)
            mean_vr[name] = np.mean(vrs)

        for name, vr in mean_vr.items():
            assert vr >= 0, f"{name} has negative mean VR ({vr:.4f})"


# ---------------------------------------------------------------------------
# 4. Unbiasedness across model complexity (Guo et al. Theorem 1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestUnbiasednessAcrossModels:
    """Cross-fitting ensures unbiased ATE regardless of model complexity."""

    @staticmethod
    def test_ate_unbiased_deep_tree():
        """Fully-grown tree: high capacity, but cross-fitting keeps ATE unbiased."""
        true_effect = 0.025
        ates = []

        for seed in range(300):
            df, cov_cols = _generate_experiment(
                n_per_group=1000,
                treatment_effect=true_effect,
                nonlinear=True,
                seed=seed,
            )
            exp = CupacExperiment(
                df,
                "converted",
                "group",
                cov_cols,
                "control",
                "treatment",
                method="mlrate",
                estimator=DecisionTreeRegressor(max_depth=None, random_state=0),
            ).fit()
            ates.append(exp.ate)

        assert np.mean(ates) == pytest.approx(true_effect, abs=0.005), (
            f"Mean ATE {np.mean(ates):.4f} too far from true effect {true_effect}"
        )

    @staticmethod
    def test_ate_unbiased_shallow_tree():
        """Shallow tree: low capacity, but still unbiased."""
        true_effect = 0.025
        ates = []

        for seed in range(300):
            df, cov_cols = _generate_experiment(
                n_per_group=1000,
                treatment_effect=true_effect,
                nonlinear=True,
                seed=seed,
            )
            exp = CupacExperiment(
                df,
                "converted",
                "group",
                cov_cols,
                "control",
                "treatment",
                method="mlrate",
                estimator=DecisionTreeRegressor(max_depth=3, random_state=0),
            ).fit()
            ates.append(exp.ate)

        assert np.mean(ates) == pytest.approx(true_effect, abs=0.005)
