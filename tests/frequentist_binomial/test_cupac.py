"""Tests for the CUPAC (Covariate-Adjusted Variance Reduction) module."""

import numpy as np
import pandas as pd
import pytest

from ab_test.frequentist_binomial.cupac import (
    CupacExperiment,
    _hc2_standard_errors,
    _ols_fit,
    cupac_adjusted_power,
    cupac_minimum_detectable_lift,
    cupac_required_sample_size,
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
    def test_init_method_lin_raises():
        df = _make_experiment_data()
        with pytest.raises(NotImplementedError, match="not supported"):
            CupacExperiment(df, "converted", "group", ["pre_visits"], "control", "treatment", method="lin")

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

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402


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
