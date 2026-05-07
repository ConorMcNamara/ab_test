import numpy as np
import pytest

from ab_test.bayesian_binomial.power_calculations import bayes_minimum_sample_size, bayes_power_lift, bayes_power_loss


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

    @staticmethod
    def test_raises_when_no_alternative_provided():
        with pytest.raises(ValueError, match="alt_lift or alt_rate"):
            bayes_power_lift(
                group_sizes=[3_000, 3_000],
                alphas=[1.0, 1.0],
                betas=[1.0, 1.0],
                baseline=0.10,
            )


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

    @staticmethod
    def test_raises_when_no_alternative_provided():
        with pytest.raises(ValueError, match="alt_lift or alt_rate"):
            bayes_minimum_sample_size(
                alphas=[1.0, 1.0],
                betas=[1.0, 1.0],
                baseline=0.10,
            )


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

    @staticmethod
    def test_raises_when_no_alternative_provided():
        with pytest.raises(ValueError, match="alt_lift or alt_rate"):
            bayes_power_loss(
                group_sizes=[1_600, 1_600],
                alphas=[1.0, 1.0],
                betas=[1.0, 1.0],
                baseline=0.10,
            )


if __name__ == "__main__":
    pytest.main()
