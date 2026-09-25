import numpy as np
import pytest

from ab_test.bayesian_binomial.power_calculations import (
    bayes_minimum_detectable_lift,
    bayes_minimum_detectable_lift_loss,
    bayes_minimum_sample_size,
    bayes_minimum_sample_size_loss,
    bayes_power_lift,
    bayes_power_loss,
)


@pytest.mark.slow
class TestBayesPowerLift:
    @staticmethod
    def test_approx_80_power_via_lift():
        # baseline=10%, 20% relative lift → 12% treatment rate, n=3_000 per arm
        # gives ~80% Bayesian power at the 95% confidence threshold
        np.random.seed(0)
        power = bayes_power_lift(
            group_sizes=[3_000, 3_000],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            lift="relative",
            n_samples=20_000,
            mc_samples=1_000,
            confidence_level=0.95,
        )
        assert power == pytest.approx(0.80, abs=0.05)

    @staticmethod
    def test_approx_80_power_via_alt_rate():
        # Equivalent to the above but supplying the treatment rate directly
        np.random.seed(0)
        power = bayes_power_lift(
            group_sizes=[3_000, 3_000],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_rate=0.12,
            n_samples=20_000,
            mc_samples=1_000,
            confidence_level=0.95,
        )
        assert power == pytest.approx(0.80, abs=0.05)


@pytest.mark.slow
class TestBayesMinimumSampleSize:
    @staticmethod
    def test_returns_plausible_n_via_lift():
        # baseline=10%, 20% relative lift → true minimum is ~3_000 per group for
        # 80% power; allow a generous range to absorb Monte Carlo variance
        np.random.seed(0)
        n = bayes_minimum_sample_size(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            lift="relative",
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 2_000 <= n <= 4_500

    @staticmethod
    def test_returns_plausible_n_via_alt_rate():
        # alt_rate=0.12 is equivalent to baseline=0.10 + 20% relative lift
        np.random.seed(0)
        n = bayes_minimum_sample_size(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_rate=0.12,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 2_000 <= n <= 4_500

    @staticmethod
    def test_larger_lift_requires_fewer_samples():
        np.random.seed(0)
        n_small_lift = bayes_minimum_sample_size(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        np.random.seed(0)
        n_large_lift = bayes_minimum_sample_size(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.40,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert n_large_lift < n_small_lift


@pytest.mark.slow
class TestBayesPowerLoss:
    @staticmethod
    def test_approx_80_power_via_lift():
        # baseline=10%, 20% relative lift, loss_threshold=0.001 → n=1_600 per arm
        # gives ~80% power under the expected-loss decision rule
        np.random.seed(0)
        power = bayes_power_loss(
            group_sizes=[1_600, 1_600],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            lift="relative",
            n_samples=20_000,
            mc_samples=1_000,
            loss_threshold=0.001,
        )
        assert power == pytest.approx(0.80, abs=0.05)

    @staticmethod
    def test_approx_80_power_via_alt_rate():
        # Equivalent to the above but supplying the treatment rate directly
        np.random.seed(0)
        power = bayes_power_loss(
            group_sizes=[1_600, 1_600],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_rate=0.12,
            n_samples=20_000,
            mc_samples=1_000,
            loss_threshold=0.001,
        )
        assert power == pytest.approx(0.80, abs=0.05)

    @staticmethod
    def test_tighter_threshold_requires_more_samples():
        # A stricter loss threshold should yield lower power at the same group size
        np.random.seed(0)
        power_loose = bayes_power_loss(
            group_sizes=[1_600, 1_600],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            n_samples=10_000,
            mc_samples=500,
            loss_threshold=0.002,
        )
        np.random.seed(0)
        power_strict = bayes_power_loss(
            group_sizes=[1_600, 1_600],
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            n_samples=10_000,
            mc_samples=500,
            loss_threshold=0.0005,
        )
        assert power_loose > power_strict


@pytest.mark.slow
class TestBayesMinimumSampleSizeLoss:
    @staticmethod
    def test_returns_plausible_n_via_lift():
        # baseline=10%, 20% relative lift, loss_threshold=0.001 → true minimum is
        # ~1_600 per group for 80% power; allow a generous range for MC variance
        np.random.seed(0)
        n = bayes_minimum_sample_size_loss(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            lift="relative",
            target_power=0.80,
            loss_threshold=0.001,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 1_000 <= n <= 2_500

    @staticmethod
    def test_returns_plausible_n_via_alt_rate():
        # alt_rate=0.12 is equivalent to baseline=0.10 + 20% relative lift
        np.random.seed(0)
        n = bayes_minimum_sample_size_loss(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_rate=0.12,
            target_power=0.80,
            loss_threshold=0.001,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 1_000 <= n <= 2_500

    @staticmethod
    def test_larger_lift_requires_fewer_samples():
        np.random.seed(0)
        n_small_lift = bayes_minimum_sample_size_loss(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.20,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        np.random.seed(0)
        n_large_lift = bayes_minimum_sample_size_loss(
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            alt_lift=0.40,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert n_large_lift < n_small_lift


@pytest.mark.slow
class TestBayesMinimumDetectableLift:
    @staticmethod
    def test_returns_plausible_lift():
        # baseline=10%, n=3_000 per group → MDL should be ~20% relative lift
        # for 80% power at 95% confidence (mirror of TestBayesPowerLift reference)
        np.random.seed(0)
        mdl = bayes_minimum_detectable_lift(
            group_size=3_000,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            lift="relative",
            target_power=0.80,
            confidence_level=0.95,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 0.12 <= mdl <= 0.30

    @staticmethod
    def test_larger_group_requires_smaller_lift():
        np.random.seed(0)
        mdl_small = bayes_minimum_detectable_lift(
            group_size=1_000,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        np.random.seed(0)
        mdl_large = bayes_minimum_detectable_lift(
            group_size=5_000,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert mdl_large < mdl_small


@pytest.mark.slow
class TestBayesMinimumDetectableLiftLoss:
    @staticmethod
    def test_returns_plausible_lift():
        # baseline=10%, n=1_600 per group → MDL should be ~20% relative lift
        # for 80% power at loss_threshold=0.001 (mirror of TestBayesPowerLoss reference)
        np.random.seed(0)
        mdl = bayes_minimum_detectable_lift_loss(
            group_size=1_600,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            lift="relative",
            target_power=0.80,
            loss_threshold=0.001,
            n_samples=5_000,
            mc_samples=300,
        )
        assert 0.12 <= mdl <= 0.30

    @staticmethod
    def test_larger_group_requires_smaller_lift():
        np.random.seed(0)
        mdl_small = bayes_minimum_detectable_lift_loss(
            group_size=500,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        np.random.seed(0)
        mdl_large = bayes_minimum_detectable_lift_loss(
            group_size=3_000,
            alphas=[1.0, 1.0],
            betas=[1.0, 1.0],
            baseline=0.10,
            target_power=0.80,
            n_samples=5_000,
            mc_samples=300,
        )
        assert mdl_large < mdl_small


class TestScaledLiftPowerLift:
    @staticmethod
    def test_incremental_matches_absolute():
        abs_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.04,
            lift="absolute",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        inc_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=40,
            lift="incremental",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        assert inc_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_roas_matches_absolute():
        abs_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.04,
            lift="absolute",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        roas_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.008,
            lift="roas",
            spend=5000.0,
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        assert roas_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_revenue_matches_absolute():
        abs_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.04,
            lift="absolute",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        rev_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=2000,
            lift="revenue",
            msrp=50.0,
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        assert rev_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_cpa_matches_absolute():
        abs_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.04,
            lift="absolute",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        cpa_pwr = bayes_power_lift(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=125,
            lift="cpa",
            spend=5000.0,
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        assert cpa_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_roas_requires_spend():
        with pytest.raises(ValueError, match="spend must be set"):
            bayes_power_lift(
                [1000, 1000],
                [1, 1],
                [1, 1],
                0.10,
                alt_lift=0.01,
                lift="roas",
                n_samples=100,
                mc_samples=10,
            )

    @staticmethod
    def test_cpa_requires_spend():
        with pytest.raises(ValueError, match="spend must be set"):
            bayes_power_lift(
                [1000, 1000],
                [1, 1],
                [1, 1],
                0.10,
                alt_lift=100,
                lift="cpa",
                n_samples=100,
                mc_samples=10,
            )

    @staticmethod
    def test_revenue_requires_msrp():
        with pytest.raises(ValueError, match="msrp must be set"):
            bayes_power_lift(
                [1000, 1000],
                [1, 1],
                [1, 1],
                0.10,
                alt_lift=2000,
                lift="revenue",
                n_samples=100,
                mc_samples=10,
            )


class TestScaledLiftPowerLoss:
    @staticmethod
    def test_incremental_matches_absolute():
        abs_pwr = bayes_power_loss(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=0.04,
            lift="absolute",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        inc_pwr = bayes_power_loss(
            [1000, 1000],
            [1, 1],
            [1, 1],
            0.10,
            alt_lift=40,
            lift="incremental",
            n_samples=5000,
            mc_samples=500,
            seed=42,
        )
        assert inc_pwr == pytest.approx(abs_pwr)


class TestScaledLiftMinSampleSizeRejects:
    @pytest.mark.parametrize("lift_type", ["incremental", "roas", "revenue", "cpa"])
    def test_minimum_sample_size_rejects(self, lift_type):
        with pytest.raises(ValueError, match="not supported"):
            bayes_minimum_sample_size(
                [1, 1],
                [1, 1],
                0.10,
                alt_lift=0.04,
                lift=lift_type,
            )

    @pytest.mark.parametrize("lift_type", ["incremental", "roas", "revenue", "cpa"])
    def test_minimum_sample_size_loss_rejects(self, lift_type):
        with pytest.raises(ValueError, match="not supported"):
            bayes_minimum_sample_size_loss(
                [1, 1],
                [1, 1],
                0.10,
                alt_lift=0.04,
                lift=lift_type,
            )


if __name__ == "__main__":
    pytest.main()
