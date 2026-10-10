import numpy as np
import pytest

from ab_test.bayesian_binomial.power_calculations import (
    _search_min_sample_size,
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


class TestSeeding:
    """Reproducible power and searches, common random numbers, and bounded memory."""

    COMMON = {"alphas": [1.0, 1.0], "betas": [1.0, 1.0], "baseline": 0.10, "alt_lift": 0.20}
    FAST = {"n_samples": 2_000, "mc_samples": 200}

    @pytest.mark.parametrize("power_fn", [bayes_power_lift, bayes_power_loss])
    def test_power_reproducible_for_a_seed(self, power_fn):
        first = power_fn([3000, 3000], **self.COMMON, **self.FAST, seed=11)
        assert power_fn([3000, 3000], **self.COMMON, **self.FAST, seed=11) == first
        assert power_fn([3000, 3000], **self.COMMON, **self.FAST, seed=12) != first

    def test_generator_seed(self):
        first = bayes_power_lift([3000, 3000], **self.COMMON, **self.FAST, seed=np.random.default_rng(3))
        assert bayes_power_lift([3000, 3000], **self.COMMON, **self.FAST, seed=np.random.default_rng(3)) == first

    def test_does_not_depend_on_n_jobs(self):
        kwargs = {**self.COMMON, "n_samples": 6_000, "mc_samples": 500, "seed": 4}
        assert bayes_power_lift([3000, 3000], **kwargs) == bayes_power_lift([3000, 3000], **kwargs, n_jobs=2)

    @pytest.mark.parametrize("search_fn", [bayes_minimum_sample_size, bayes_minimum_sample_size_loss])
    def test_sample_size_search_reproducible(self, search_fn):
        # Reviewer: four identical unseeded calls returned 2947, 2949, 2900 and 3009.
        results = {search_fn(**self.COMMON, **self.FAST, seed=7) for _ in range(4)}
        assert len(results) == 1

    @pytest.mark.parametrize("search_fn", [bayes_minimum_detectable_lift, bayes_minimum_detectable_lift_loss])
    def test_lift_search_reproducible(self, search_fn):
        kwargs = {"group_size": 3000, "alphas": [1.0, 1.0], "betas": [1.0, 1.0], "baseline": 0.10, "tol": 0.001}
        assert search_fn(**kwargs, **self.FAST, seed=7) == search_fn(**kwargs, **self.FAST, seed=7)

    @pytest.mark.parametrize("seed", [7, 8])
    def test_power_is_smooth_in_n(self, seed):
        # Every n replays the same random numbers, so power changes smoothly with n. Re-drawing
        # them for each n made it fall by up to 0.008-0.0096 between sizes 10 apart; now at most
        # about 0.001. It is not exactly monotone: the posterior draws share a stream but not
        # an exact coupling.
        sizes = range(2950, 3101, 10)
        powers = [bayes_power_lift([n, n], **self.COMMON, n_samples=5_000, mc_samples=300, seed=seed) for n in sizes]
        assert min(np.diff(powers)) > -0.003

    def test_memory_stays_bounded(self):
        import tracemalloc

        # Holding every posterior draw took about 85 MB here (and about 1.6 GB at the defaults).
        tracemalloc.start()
        try:
            bayes_power_lift([3000, 3000], **self.COMMON, n_samples=5_000, mc_samples=1_000, seed=1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 40e6


if __name__ == "__main__":
    pytest.main()


class TestMinimumDetectableLiftRateBound:
    @staticmethod
    @pytest.mark.parametrize("group_size, baseline", [(5, 0.6), (3, 0.3)])
    def test_unreachable_power_raises_clear_error(group_size, baseline):
        # Used to raise numpy's "p < 0, p > 1 or p is NaN" once the lift pushed the rate past 1.
        np.random.seed(0)
        with pytest.raises(ValueError, match="keeps the treatment rate below 1"):
            bayes_minimum_detectable_lift(
                group_size, [1, 1], [1, 1], baseline, lift="relative", n_samples=300, mc_samples=100
            )

    @staticmethod
    def test_reachable_lift_near_rate_limit():
        # Used to crash: doubling from 0.32 to 0.64 overshot a rate of 1 before power was reached.
        np.random.seed(0)
        mdl = bayes_minimum_detectable_lift(10, [1, 1], [1, 1], 0.5, lift="absolute", n_samples=500, mc_samples=200)
        assert 0 < mdl < 0.5

    @staticmethod
    def test_loss_search_is_bounded_too():
        np.random.seed(0)
        with pytest.raises(ValueError, match="keeps the treatment rate below 1"):
            bayes_minimum_detectable_lift_loss(3, [1, 1], [1, 1], 0.3, lift="relative", n_samples=300, mc_samples=100)


class TestSearchMinSampleSize:
    @staticmethod
    @pytest.mark.parametrize("threshold", [1, 37, 100, 101, 2500])
    def test_returns_exact_threshold(threshold):
        # Used to return at least 101: n <= 100 was assumed underpowered without being checked.
        n = _search_min_sample_size(lambda n: float(n >= threshold), 0.8, 10_000, "unreachable")
        assert n == threshold

    @staticmethod
    def test_max_n_itself_is_evaluated():
        # Used to raise: doubling jumped from 800 to 1600 without trying max_n = 1000.
        assert _search_min_sample_size(lambda n: float(n >= 900), 0.8, 1000, "unreachable") == 900

    @staticmethod
    def test_unreachable_raises():
        with pytest.raises(ValueError, match="unreachable"):
            _search_min_sample_size(lambda n: 0.0, 0.8, 1000, "unreachable")
