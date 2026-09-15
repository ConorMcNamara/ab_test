"""Tests for the CUPAC (Covariate-Adjusted Variance Reduction) module."""

import numpy as np
import pandas as pd
import pytest

from ab_test.frequentist_binomial.cupac import (
    CupacExperiment,
    _cr2_standard_errors,
    _hc2_standard_errors,
    _ols_fit,
    cupac_adjusted_power,
    cupac_minimum_detectable_lift,
    cupac_required_sample_size,
    plot_cupac_power_curve,
    plot_cupac_sensitivity_curve,
)
from ab_test.frequentist_binomial.power_calculations import abtest_power, minimum_detectable_lift, required_sample_size


def _make_experiment_data(
    n_control: int = 2000,
    n_treatment: int = 2000,
    baseline: float = 0.1,
    treatment_effect: float = 0.0,
    covariate_r_squared: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic experiment data with a correlated covariate."""
    rng = np.random.default_rng(seed)
    n = n_control + n_treatment

    latent = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    covariate = latent * np.sqrt(covariate_r_squared) + noise * np.sqrt(1 - covariate_r_squared)

    group = np.array(["control"] * n_control + ["treatment"] * n_treatment)
    prob = baseline + treatment_effect * (group == "treatment") + 0.02 * latent
    prob = np.clip(prob, 0.01, 0.99)
    outcome = rng.binomial(1, prob)

    return pd.DataFrame({"group": group, "converted": outcome, "pre_visits": covariate})


class TestOlsFit:
    @staticmethod
    def test_simple_regression():
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        y = np.array([2.0, 4.0, 6.0, 8.0])
        beta = _ols_fit(X, y)
        assert beta[0] == pytest.approx(0.0, abs=1e-10)
        assert beta[1] == pytest.approx(2.0, abs=1e-10)

    @staticmethod
    def test_multiple_covariates():
        rng = np.random.default_rng(0)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        true_beta = np.array([1.0, 2.0, -0.5])
        y = X @ true_beta
        beta = _ols_fit(X, y)
        np.testing.assert_allclose(beta, true_beta, atol=1e-10)

    @staticmethod
    def test_rank_deficient():
        X = np.array([[1, 2, 2], [1, 4, 4], [1, 6, 6]], dtype=float)
        y = np.array([1.0, 2.0, 3.0])
        beta = _ols_fit(X, y)
        y_pred = X @ beta
        np.testing.assert_allclose(y_pred, y, atol=1e-10)


class TestHC2StandardErrors:
    @staticmethod
    def test_matches_classical_se_homoskedastic():
        rng = np.random.default_rng(42)
        n = 1000
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ [1.0, 2.0] + rng.normal(0, 1, n)
        beta = _ols_fit(X, y)
        hc2_se = _hc2_standard_errors(X, y, beta)
        residuals = y - X @ beta
        classical_var = np.sum(residuals**2) / (n - 2) * np.linalg.inv(X.T @ X)
        classical_se = np.sqrt(np.diag(classical_var))
        np.testing.assert_allclose(hc2_se, classical_se, rtol=0.15)

    @staticmethod
    def test_larger_under_heteroskedasticity():
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        noise_scale = 1 + 2 * np.abs(x)
        y = X @ [1.0, 2.0] + rng.normal(0, 1, n) * noise_scale
        beta = _ols_fit(X, y)
        hc2_se = _hc2_standard_errors(X, y, beta)
        residuals = y - X @ beta
        classical_var = np.sum(residuals**2) / (n - 2) * np.linalg.inv(X.T @ X)
        classical_se = np.sqrt(np.diag(classical_var))
        assert hc2_se[1] > classical_se[1]


class TestCupacExperiment:
    @staticmethod
    def test_init_valid():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        assert exp.method == "cupac"

    @staticmethod
    def test_init_empty_covariate_cols():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="covariate_cols must not be empty"):
            CupacExperiment(df, "converted", "group", [], "control", "treatment")

    @staticmethod
    def test_init_missing_column():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="Columns not found"):
            CupacExperiment(df, "converted", "group", ["nonexistent"], "control", "treatment")

    @staticmethod
    def test_init_non_binary_outcome():
        df = _make_experiment_data()
        df["converted"] = df["converted"] * 5
        with pytest.raises(ValueError, match="binary"):
            CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")

    @staticmethod
    def test_init_wrong_treatment_labels():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="Expected treatment column"):
            CupacExperiment(df, "converted", "group", ["pre_visits"], "A", "B")

    @staticmethod
    def test_init_method_lin_valid():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin")
        assert exp.method == "lin"

    @staticmethod
    def test_fit_returns_self():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        result = exp.fit()
        assert result is exp

    @staticmethod
    def test_properties_before_fit():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        with pytest.raises(RuntimeError, match="fit"):
            _ = exp.ate
        with pytest.raises(RuntimeError, match="fit"):
            _ = exp.se
        with pytest.raises(RuntimeError, match="fit"):
            _ = exp.p_value
        with pytest.raises(RuntimeError, match="fit"):
            _ = exp.variance_reduction

    @staticmethod
    def test_analyze_returns_string():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        result = exp.analyze()
        assert isinstance(result, str)
        assert "Adjusted ATE" in result
        assert "Variance Reduction" in result

    @staticmethod
    def test_analyze_calls_fit():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        assert exp._results is None
        exp.analyze()
        assert exp._results is not None

    @staticmethod
    def test_summary_dict_keys():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        s = exp.summary()
        expected_keys = {
            "ate",
            "ate_unadjusted",
            "se",
            "se_unadjusted",
            "z_stat",
            "p_value",
            "r_squared",
            "theta",
            "n_control",
            "n_treatment",
            "df",
            "n_clusters",
            "ci_lower",
            "ci_upper",
        }
        assert set(s.keys()) == expected_keys

    @staticmethod
    def test_known_treatment_effect():
        df = _make_experiment_data(n_control=5000, n_treatment=5000, treatment_effect=0.03, seed=123)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        assert exp.ate == pytest.approx(0.03, abs=0.015)

    @staticmethod
    def test_variance_reduction_positive():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        assert exp.variance_reduction > 0

    @staticmethod
    def test_variance_reduction_near_zero_with_noise():
        rng = np.random.default_rng(42)
        n = 4000
        df = pd.DataFrame(
            {
                "group": ["control"] * 2000 + ["treatment"] * 2000,
                "converted": rng.binomial(1, 0.1, n),
                "noise_covariate": rng.normal(0, 1, n),
            }
        )
        exp = CupacExperiment(df, "converted", "group", ["noise_covariate"], "control", "treatment").fit()
        assert abs(exp.variance_reduction) < 0.05

    @staticmethod
    def test_se_smaller_than_unadjusted():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        s = exp.summary()
        assert s["se"] < s["se_unadjusted"]

    @staticmethod
    def test_polars_input():
        pytest.importorskip("polars")
        pytest.importorskip("pyarrow")
        import polars as pl

        df_pd = _make_experiment_data()
        df_pl = pl.from_pandas(df_pd)

        exp_pd = CupacExperiment(df_pd, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        exp_pl = CupacExperiment(df_pl, "converted", "group", ["pre_visits"], "control", "treatment").fit()

        assert exp_pd.ate == pytest.approx(exp_pl.ate)
        assert exp_pd.se == pytest.approx(exp_pl.se)

    @staticmethod
    def test_chaining():
        df = _make_experiment_data()
        result = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit().analyze()
        assert isinstance(result, str)


class TestCupacPowerCalculations:
    @staticmethod
    def test_adjusted_power_higher():
        power_unadj = abtest_power([1000, 1000], 0.10, 0.20)
        power_adj = abtest_power([1000, 1000], 0.10, 0.20, power=cupac_adjusted_power(0.3))
        assert power_adj > power_unadj

    @staticmethod
    def test_adjusted_power_zero_r2():
        power_unadj = abtest_power([1000, 1000], 0.10, 0.20)
        power_adj = abtest_power([1000, 1000], 0.10, 0.20, power=cupac_adjusted_power(0.0))
        assert power_adj == pytest.approx(power_unadj)

    @staticmethod
    def test_mdl_smaller():
        mdl_unadj = minimum_detectable_lift([1000, 1000], 0.10)
        mdl_adj = cupac_minimum_detectable_lift([1000, 1000], 0.10, r_squared=0.3)
        assert mdl_adj < mdl_unadj

    @staticmethod
    def test_sample_size_smaller():
        n_unadj = required_sample_size(0.10, 0.20)
        n_adj = cupac_required_sample_size(0.10, 0.20, r_squared=0.3)
        assert n_adj < n_unadj

    @staticmethod
    def test_invalid_r_squared():
        with pytest.raises(ValueError, match="r_squared"):
            cupac_adjusted_power(1.0)
        with pytest.raises(ValueError, match="r_squared"):
            cupac_adjusted_power(-0.1)


class TestCupacStatisticalProperties:
    @staticmethod
    def test_type_i_error_control():
        """Under the null, CUPAC rejects at approximately alpha."""
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 500
        rejections = 0

        for i in range(n_sims):
            n = 2000
            df = pd.DataFrame(
                {
                    "group": ["control"] * 1000 + ["treatment"] * 1000,
                    "converted": rng.binomial(1, 0.1, n),
                    "cov": rng.normal(0, 1, n),
                }
            )
            exp = CupacExperiment(df, "converted", "group", ["cov"], "control", "treatment").fit()
            if exp.p_value < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate < alpha + 0.03, f"Rejection rate {rejection_rate:.3f} exceeds alpha={alpha} + margin"

    @staticmethod
    def test_ate_unbiased():
        """Over many simulations, the mean ATE should be close to the true effect."""
        true_effect = 0.02
        ates = []

        for seed in range(200):
            df = _make_experiment_data(n_control=1000, n_treatment=1000, treatment_effect=true_effect, seed=seed)
            exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
            ates.append(exp.ate)

        mean_ate = np.mean(ates)
        assert mean_ate == pytest.approx(true_effect, abs=0.005)


# ---------------------------------------------------------------------------
# MLRATE tests — guarded by sklearn availability
# ---------------------------------------------------------------------------

try:
    from sklearn.linear_model import LinearRegression

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

needs_sklearn = pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")


@needs_sklearn
class TestCupacMlrateValidation:
    @staticmethod
    def test_mlrate_without_estimator_raises():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="estimator is required"):
            CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="mlrate")

    @staticmethod
    def test_mlrate_bad_estimator_raises():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="fit.*predict"):
            CupacExperiment(
                df,
                "converted",
                "group",
                ["pre_visits"],
                "control",
                "treatment",
                method="mlrate",
                estimator="not_a_model",
            )

    @staticmethod
    def test_mlrate_valid_init():
        df = _make_experiment_data()
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        )
        assert exp.method == "mlrate"


@needs_sklearn
class TestCupacMlrateAnalyze:
    @staticmethod
    def test_fit_returns_self():
        df = _make_experiment_data()
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        )
        result = exp.fit()
        assert result is exp

    @staticmethod
    def test_analyze_returns_string():
        df = _make_experiment_data()
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        )
        result = exp.analyze()
        assert isinstance(result, str)
        assert "MLRATE" in result

    @staticmethod
    def test_summary_dict_keys():
        df = _make_experiment_data()
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        ).fit()
        s = exp.summary()
        expected_keys = {
            "ate",
            "ate_unadjusted",
            "se",
            "se_unadjusted",
            "z_stat",
            "p_value",
            "r_squared",
            "theta",
            "n_control",
            "n_treatment",
            "df",
            "n_clusters",
            "ci_lower",
            "ci_upper",
        }
        assert set(s.keys()) == expected_keys

    @staticmethod
    def test_known_treatment_effect():
        df = _make_experiment_data(n_control=5000, n_treatment=5000, treatment_effect=0.03, seed=123)
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        ).fit()
        assert exp.ate == pytest.approx(0.03, abs=0.015)

    @staticmethod
    def test_variance_reduction_positive():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        ).fit()
        assert exp.variance_reduction > 0

    @staticmethod
    def test_se_smaller_than_unadjusted():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        ).fit()
        s = exp.summary()
        assert s["se"] < s["se_unadjusted"]


@needs_sklearn
class TestCupacMlratePredictProba:
    @staticmethod
    def test_classifier_uses_predict_proba():
        """A classifier with predict_proba should yield better variance reduction than discrete predict."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        df = _make_experiment_data(n_control=3000, n_treatment=3000, covariate_r_squared=0.3, seed=99)

        exp_classifier = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LogisticRegression(),
        ).fit()

        assert exp_classifier.variance_reduction > 0

    @staticmethod
    def test_regressor_still_works():
        """A regressor (no predict_proba) still produces valid results."""
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        ).fit()
        assert exp.variance_reduction > 0
        assert exp._results is not None


@needs_sklearn
class TestCupacMlrateCrossFitting:
    @staticmethod
    def test_custom_n_folds():
        df = _make_experiment_data()
        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
            n_folds=3,
        ).fit()
        assert exp._results is not None

    @staticmethod
    def test_cross_fitted_predictions_differ_from_in_sample():
        """Cross-fitted predictions should differ from in-sample predictions."""
        df = _make_experiment_data(covariate_r_squared=0.3)
        covariates = df[["pre_visits"]].to_numpy(dtype=float)
        y = df["converted"].to_numpy(dtype=float)

        model = LinearRegression()
        model.fit(covariates, y)
        y_hat_insample = model.predict(covariates)

        exp = CupacExperiment(
            df,
            "converted",
            "group",
            ["pre_visits"],
            "control",
            "treatment",
            method="mlrate",
            estimator=LinearRegression(),
        )
        y_hat_oof = exp._cross_fit_predictions(covariates, y)

        assert not np.allclose(y_hat_insample, y_hat_oof)


@needs_sklearn
class TestCupacMlrateStatisticalProperties:
    @staticmethod
    def test_type_i_error_control():
        """Under the null, MLRATE rejects at approximately alpha."""
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 500
        rejections = 0

        for i in range(n_sims):
            n = 2000
            df = pd.DataFrame(
                {
                    "group": ["control"] * 1000 + ["treatment"] * 1000,
                    "converted": rng.binomial(1, 0.1, n),
                    "cov": rng.normal(0, 1, n),
                }
            )
            exp = CupacExperiment(
                df,
                "converted",
                "group",
                ["cov"],
                "control",
                "treatment",
                method="mlrate",
                estimator=LinearRegression(),
            ).fit()
            if exp.p_value < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate < alpha + 0.03, f"Rejection rate {rejection_rate:.3f} exceeds alpha={alpha} + margin"

    @staticmethod
    def test_ate_unbiased():
        """Over many simulations, the mean ATE should be close to the true effect."""
        true_effect = 0.02
        ates = []

        for seed in range(200):
            df = _make_experiment_data(n_control=1000, n_treatment=1000, treatment_effect=true_effect, seed=seed)
            exp = CupacExperiment(
                df,
                "converted",
                "group",
                ["pre_visits"],
                "control",
                "treatment",
                method="mlrate",
                estimator=LinearRegression(),
            ).fit()
            ates.append(exp.ate)

        mean_ate = np.mean(ates)
        assert mean_ate == pytest.approx(true_effect, abs=0.005)


class TestCupacPlotCurves:
    @staticmethod
    def test_power_curve_returns_figure():
        fig = plot_cupac_power_curve(baseline=0.10, alt_lift=0.20, r_squared=0.3)
        assert fig is not None
        assert len(fig.data) == 2
        assert fig.data[0].name == "CUPAC-adjusted (R²=0.30)"
        assert fig.data[1].name == "Unadjusted"

    @staticmethod
    def test_power_curve_adjusted_above_unadjusted():
        fig = plot_cupac_power_curve(baseline=0.10, alt_lift=0.20, r_squared=0.3)
        adjusted_y = fig.data[0].y
        unadjusted_y = fig.data[1].y
        assert all(a >= u - 1e-9 for a, u in zip(adjusted_y, unadjusted_y))

    @staticmethod
    def test_power_curve_custom_sample_sizes():
        sizes = [500, 1000, 2000, 4000]
        fig = plot_cupac_power_curve(baseline=0.10, alt_lift=0.20, r_squared=0.3, sample_sizes=sizes)
        assert list(fig.data[0].x) == sizes

    @staticmethod
    def test_sensitivity_curve_returns_figure():
        fig = plot_cupac_sensitivity_curve(baseline=0.10, r_squared=0.3)
        assert fig is not None
        assert len(fig.data) == 2
        assert fig.data[0].name == "CUPAC-adjusted (R²=0.30)"
        assert fig.data[1].name == "Unadjusted"

    @staticmethod
    def test_sensitivity_curve_adjusted_below_unadjusted():
        fig = plot_cupac_sensitivity_curve(baseline=0.10, r_squared=0.3)
        adjusted_y = fig.data[0].y
        unadjusted_y = fig.data[1].y
        assert all(a <= u + 1e-9 for a, u in zip(adjusted_y, unadjusted_y))

    @staticmethod
    def test_sensitivity_curve_custom_sample_sizes():
        sizes = [500, 1000, 2000, 4000]
        fig = plot_cupac_sensitivity_curve(baseline=0.10, r_squared=0.3, sample_sizes=sizes)
        assert list(fig.data[0].x) == sizes


# ---------------------------------------------------------------------------
# Confidence interval tests
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    @staticmethod
    def test_before_fit_raises():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        with pytest.raises(RuntimeError, match="fit"):
            exp.confidence_interval()

    @staticmethod
    def test_default_alpha():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        lo, hi = exp.confidence_interval()
        assert lo < hi

    @staticmethod
    def test_custom_alpha():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        lo_99, hi_99 = exp.confidence_interval(alpha=0.01)
        lo_95, hi_95 = exp.confidence_interval(alpha=0.05)
        assert hi_99 - lo_99 > hi_95 - lo_95

    @staticmethod
    def test_contains_ate():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        lo, hi = exp.confidence_interval()
        assert lo <= exp.ate <= hi

    @staticmethod
    def test_properties_match_method():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        lo, hi = exp.confidence_interval(0.05)
        assert exp.ci_lower == pytest.approx(lo)
        assert exp.ci_upper == pytest.approx(hi)

    @staticmethod
    def test_summary_includes_ci():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        s = exp.summary()
        assert "ci_lower" in s
        assert "ci_upper" in s
        assert s["ci_lower"] < s["ci_upper"]

    @staticmethod
    def test_ci_lower_property_before_fit():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment")
        with pytest.raises(RuntimeError, match="fit"):
            _ = exp.ci_lower


# ---------------------------------------------------------------------------
# Lin's method tests
# ---------------------------------------------------------------------------


class TestLinMethod:
    @staticmethod
    def test_fit_returns_self():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin")
        result = exp.fit()
        assert result is exp

    @staticmethod
    def test_known_effect_recovery():
        df = _make_experiment_data(n_control=5000, n_treatment=5000, treatment_effect=0.03, seed=123)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
        assert exp.ate == pytest.approx(0.03, abs=0.015)

    @staticmethod
    def test_variance_reduction_positive():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
        assert exp.variance_reduction > 0

    @staticmethod
    def test_se_smaller_than_unadjusted():
        df = _make_experiment_data(covariate_r_squared=0.3)
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
        s = exp.summary()
        assert s["se"] < s["se_unadjusted"]

    @staticmethod
    def test_analyze_shows_lin():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin")
        result = exp.analyze()
        assert "Lin" in result

    @staticmethod
    def test_summary_keys():
        df = _make_experiment_data()
        exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
        s = exp.summary()
        assert "ci_lower" in s
        assert "ci_upper" in s
        assert s["df"] is None
        assert s["n_clusters"] is None


# ---------------------------------------------------------------------------
# Cluster-robust SE tests
# ---------------------------------------------------------------------------


def _make_clustered_experiment_data(
    n_clusters: int = 50,
    cluster_size: int = 40,
    baseline: float = 0.3,
    treatment_effect: float = 0.0,
    icc: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data with cluster-level random intercepts."""
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

    cluster_effects = rng.normal(0, np.sqrt(icc), n_clusters)
    individual_noise = rng.normal(0, np.sqrt(1 - icc), n)

    treatment = np.zeros(n)
    treatment_clusters = rng.choice(n_clusters, n_clusters // 2, replace=False)
    for c in treatment_clusters:
        treatment[cluster_ids == c] = 1.0

    latent = cluster_effects[cluster_ids] + individual_noise
    covariate = latent + rng.normal(0, 0.5, n)
    prob = baseline + treatment_effect * treatment + 0.15 * cluster_effects[cluster_ids]
    prob = np.clip(prob, 0.01, 0.99)
    outcome = rng.binomial(1, prob)

    group = np.where(treatment == 1.0, "treatment", "control")

    return pd.DataFrame(
        {
            "group": group,
            "converted": outcome,
            "pre_visits": covariate,
            "cluster_id": cluster_ids,
        }
    )


class TestClusterRobustSE:
    @staticmethod
    def test_invalid_cluster_col_raises():
        df = _make_experiment_data()
        with pytest.raises(ValueError, match="not found in data"):
            CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="nonexistent")

    @staticmethod
    def test_fit_succeeds():
        df = _make_clustered_experiment_data()
        exp = CupacExperiment(
            df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
        ).fit()
        assert exp._results is not None

    @staticmethod
    def test_cr2_se_larger_than_hc2():
        df = _make_clustered_experiment_data(n_clusters=20, cluster_size=100, icc=0.5, seed=42)
        exp_hc2 = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment").fit()
        exp_cr2 = CupacExperiment(
            df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
        ).fit()
        assert exp_cr2.se > exp_hc2.se

    @staticmethod
    def test_summary_includes_cluster_info():
        df = _make_clustered_experiment_data()
        exp = CupacExperiment(
            df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
        ).fit()
        s = exp.summary()
        assert s["n_clusters"] == 50
        assert s["df"] is not None
        assert s["df"] > 0

    @staticmethod
    def test_analyze_shows_cr2():
        df = _make_clustered_experiment_data()
        exp = CupacExperiment(
            df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
        )
        result = exp.analyze()
        assert "CR2" in result
        assert "clusters" in result.lower()

    @staticmethod
    def test_confidence_interval_uses_t():
        import scipy.stats as ss

        df = _make_clustered_experiment_data()
        exp = CupacExperiment(
            df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
        ).fit()
        results = exp._results
        lo, hi = exp.confidence_interval(alpha=0.05)
        t_crit = ss.t.ppf(0.975, df=results["df"])
        expected_lo = results["ate"] - t_crit * results["se"]
        expected_hi = results["ate"] + t_crit * results["se"]
        assert lo == pytest.approx(expected_lo)
        assert hi == pytest.approx(expected_hi)


class TestCR2StandardErrors:
    @staticmethod
    def test_returns_se_and_df():
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.binomial(1, 0.5, n), rng.normal(0, 1, n)])
        y = X @ [0.1, 0.02, 0.01] + rng.normal(0, 0.3, n)
        beta = _ols_fit(X, y)
        cluster_ids = np.repeat(np.arange(20), 10)
        se, df_val = _cr2_standard_errors(X, y, beta, cluster_ids)
        assert len(se) == 3
        assert all(s > 0 for s in se)
        assert df_val > 0

    @staticmethod
    def test_df_bounded_by_n_clusters():
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.binomial(1, 0.5, n)])
        y = X @ [0.1, 0.02] + rng.normal(0, 0.3, n)
        beta = _ols_fit(X, y)
        n_clusters = 20
        cluster_ids = np.repeat(np.arange(n_clusters), 10)
        _, df_val = _cr2_standard_errors(X, y, beta, cluster_ids)
        assert df_val <= n_clusters


# ---------------------------------------------------------------------------
# Slow statistical property tests (Monte Carlo)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLinStatisticalProperties:
    @staticmethod
    def test_type_i_error_control():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 500
        rejections = 0

        for i in range(n_sims):
            n = 2000
            df = pd.DataFrame(
                {
                    "group": ["control"] * 1000 + ["treatment"] * 1000,
                    "converted": rng.binomial(1, 0.1, n),
                    "cov": rng.normal(0, 1, n),
                }
            )
            exp = CupacExperiment(df, "converted", "group", ["cov"], "control", "treatment", method="lin").fit()
            if exp.p_value < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate < alpha + 0.03

    @staticmethod
    def test_ate_unbiased():
        true_effect = 0.02
        ates = []

        for seed in range(200):
            df = _make_experiment_data(n_control=1000, n_treatment=1000, treatment_effect=true_effect, seed=seed)
            exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
            ates.append(exp.ate)

        mean_ate = np.mean(ates)
        assert mean_ate == pytest.approx(true_effect, abs=0.005)

    @staticmethod
    def test_ci_coverage():
        true_effect = 0.02
        covered = 0
        n_sims = 300

        for seed in range(n_sims):
            df = _make_experiment_data(n_control=1000, n_treatment=1000, treatment_effect=true_effect, seed=seed)
            exp = CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin").fit()
            lo, hi = exp.confidence_interval(0.05)
            if lo <= true_effect <= hi:
                covered += 1

        coverage = covered / n_sims
        assert coverage >= 0.92


@pytest.mark.slow
class TestClusterRobustStatisticalProperties:
    @staticmethod
    def test_type_i_error_control():
        alpha = 0.05
        n_sims = 300
        rejections = 0

        for seed in range(n_sims):
            df = _make_clustered_experiment_data(
                n_clusters=30, cluster_size=40, treatment_effect=0.0, icc=0.3, seed=seed
            )
            exp = CupacExperiment(
                df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
            ).fit()
            if exp.p_value < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate < alpha + 0.04

    @staticmethod
    def test_ate_unbiased():
        true_effect = 0.03
        ates = []

        for seed in range(200):
            df = _make_clustered_experiment_data(
                n_clusters=50, cluster_size=40, treatment_effect=true_effect, icc=0.3, seed=seed
            )
            exp = CupacExperiment(
                df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
            ).fit()
            ates.append(exp.ate)

        mean_ate = np.mean(ates)
        assert mean_ate == pytest.approx(true_effect, abs=0.01)

    @staticmethod
    def test_ci_coverage():
        true_effect = 0.03
        covered = 0
        n_sims = 300

        for seed in range(n_sims):
            df = _make_clustered_experiment_data(
                n_clusters=30, cluster_size=40, treatment_effect=true_effect, icc=0.3, seed=seed
            )
            exp = CupacExperiment(
                df, "converted", "group", ["pre_visits"], "control", "treatment", cluster_col="cluster_id"
            ).fit()
            lo, hi = exp.confidence_interval(0.05)
            if lo <= true_effect <= hi:
                covered += 1

        coverage = covered / n_sims
        assert coverage >= 0.92
