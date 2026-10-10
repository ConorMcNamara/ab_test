"""Tests for frequentist equivalence testing (TOST)."""

from __future__ import annotations

import pytest

from ab_test.frequentist_binomial.equivalence import tost_test


class TestTostBasic:
    @staticmethod
    def test_returns_dict() -> None:
        result = tost_test([1000, 1000], [100, 102], delta=0.05)
        assert isinstance(result, dict)
        assert set(result) == {"p_value", "p_lower", "p_upper", "equivalent"}

    @staticmethod
    def test_equivalent_when_rates_are_close() -> None:
        result = tost_test([1000, 1000], [100, 102], delta=0.05)
        assert result["equivalent"] is True
        assert result["p_value"] < 0.05

    @staticmethod
    def test_not_equivalent_when_rates_differ() -> None:
        result = tost_test([1000, 1000], [100, 200], delta=0.02)
        assert result["equivalent"] is False
        assert result["p_value"] > 0.05

    @staticmethod
    def test_tight_margin_not_equivalent() -> None:
        result = tost_test([1000, 1000], [100, 105], delta=0.001)
        assert result["equivalent"] is False

    @staticmethod
    def test_wide_margin_equivalent() -> None:
        result = tost_test([1000, 1000], [100, 110], delta=0.10)
        assert result["equivalent"] is True

    @staticmethod
    def test_identical_rates() -> None:
        result = tost_test([1000, 1000], [100, 100], delta=0.04)
        assert result["equivalent"] is True
        assert result["p_value"] < 0.01


class TestTostPvalueOrdering:
    @staticmethod
    def test_p_value_is_max_of_onesided() -> None:
        result = tost_test([1000, 1000], [100, 103], delta=0.05)
        assert result["p_value"] == max(result["p_lower"], result["p_upper"])

    @staticmethod
    def test_p_values_between_zero_and_one() -> None:
        result = tost_test([1000, 1000], [100, 110], delta=0.05)
        for key in ("p_value", "p_lower", "p_upper"):
            assert 0 <= result[key] <= 1

    @staticmethod
    def test_larger_sample_more_power() -> None:
        small = tost_test([200, 200], [20, 21], delta=0.03)
        large = tost_test([2000, 2000], [200, 210], delta=0.03)
        assert large["p_value"] < small["p_value"]


class TestTostMethods:
    @staticmethod
    def test_score_method() -> None:
        result = tost_test([1000, 1000], [100, 102], delta=0.05, method="score")
        assert result["equivalent"] is True

    @staticmethod
    def test_likelihood_method() -> None:
        result = tost_test([1000, 1000], [100, 102], delta=0.05, method="likelihood")
        assert result["equivalent"] is True

    @staticmethod
    def test_z_method() -> None:
        result = tost_test([1000, 1000], [100, 102], delta=0.05, method="z")
        assert result["equivalent"] is True

    @staticmethod
    def test_methods_agree_roughly() -> None:
        args = ([1000, 1000], [100, 105])
        kwargs = {"delta": 0.05}
        p_score = tost_test(*args, method="score", **kwargs)["p_value"]
        p_lr = tost_test(*args, method="likelihood", **kwargs)["p_value"]
        p_z = tost_test(*args, method="z", **kwargs)["p_value"]
        assert p_score == pytest.approx(p_lr, abs=0.02)
        assert p_score == pytest.approx(p_z, abs=0.02)

    @staticmethod
    def test_unsupported_method_raises() -> None:
        with pytest.raises(ValueError, match="does not support"):
            tost_test([1000, 1000], [100, 102], delta=0.05, method="fisher")


class TestTostRelativeLift:
    @staticmethod
    def test_relative_lift_equivalent() -> None:
        result = tost_test([10000, 10000], [1000, 1010], delta=0.10, lift="relative")
        assert result["equivalent"] is True

    @staticmethod
    def test_relative_lift_not_equivalent() -> None:
        result = tost_test([1000, 1000], [100, 200], delta=0.05, lift="relative")
        assert result["equivalent"] is False


class TestTostValidation:
    @staticmethod
    def test_negative_delta_raises() -> None:
        with pytest.raises(ValueError, match="positive"):
            tost_test([1000, 1000], [100, 102], delta=-0.05)

    @staticmethod
    def test_zero_delta_raises() -> None:
        with pytest.raises(ValueError, match="positive"):
            tost_test([1000, 1000], [100, 102], delta=0.0)


def test_z_method_with_relative_lift_raises_clear_error():
    # The docstring listed "z" without noting it only supports absolute lift; it failed deep inside z_test.
    with pytest.raises(ValueError, match="supports only lift='absolute'"):
        tost_test([1000, 1000], [100, 102], delta=0.1, lift="relative", method="z")


class TestRelativeMarginOfOneOrMore:
    @staticmethod
    @pytest.mark.parametrize("method", ["score", "likelihood"])
    @pytest.mark.parametrize("delta", [1.0, 1.5])
    def test_lower_null_is_impossible(method, delta):
        # The lower null (lift <= -delta <= -1) needs a treatment rate of 0 or less.
        # The constrained MLE was invalid here: p_lower came out as 0.5 (score) or NaN.
        result = tost_test([1000, 1000], [100, 110], delta=delta, lift="relative", method=method)
        assert result["p_lower"] == 0.0
        assert result["p_value"] == result["p_upper"]
        assert result["equivalent"]

    @staticmethod
    def test_continuous_with_margin_just_below_one():
        below = tost_test([1000, 1000], [100, 110], delta=0.99, lift="relative")
        assert below["p_lower"] < 1e-12
        assert below["equivalent"] == tost_test([1000, 1000], [100, 110], delta=1.0, lift="relative")["equivalent"]

    @staticmethod
    def test_no_treatment_successes_at_margin_of_one():
        # A treatment rate of exactly 0 is the lower null at delta = 1, and the data agree with it.
        result = tost_test([1000, 1000], [100, 0], delta=1.0, lift="relative")
        assert result["p_lower"] == 1.0
        assert not result["equivalent"]
