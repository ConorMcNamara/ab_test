Power & Sample Size
===================

This module provides power analysis for frequentist binomial A/B tests. It
answers three common pre-experiment questions: how much power does a given
sample size provide, what is the minimum detectable lift for a given sample
size, and how many observations are needed to detect a given lift with adequate
power.

All functions use binary search and delegate to
:func:`~ab_test.frequentist_binomial.power_calculations.score_power` by
default. It uses the score test's contrast with its variance under the null
(which sets the critical value) and under the alternative (which sets its
spread), as in Fleiss, Tytun & Ury (1980) and Farrington & Manning (1990), so
it stays accurate with unequal group sizes. A custom power function can be passed to any of
these functions -- for example,
:func:`~ab_test.frequentist_binomial.cupac.cupac_adjusted_power` to account
for CUPAC variance reduction.

Both relative and absolute lift are supported.

With three or more groups, the functions plan for the comparisons
:meth:`~ab_test.frequentist_binomial.contingency.ContingencyTable.analyze`
will make. ``comparisons="control"`` (the default) compares each variant with
the first group; ``comparisons="all"`` compares every pair. For ``m``
comparisons, power is computed for the least-powered one (the control against
the smallest variant, or the two smallest groups) at the Bonferroni level
``alpha / m``. ``analyze()`` adjusts with Holm by default, which rejects at
least as often, so the power is a slight underestimate and the sample size a
slight overestimate.

.. code-block:: python

   # Total sample size for an A/B/C test, split evenly, comparing B and C to A
   required_sample_size(0.10, 0.30, group_proportions=[1 / 3] * 3)

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.power_calculations import (
       abtest_power,
       minimum_detectable_lift,
       required_sample_size,
   )

   # Power for a given sample size and expected lift
   power = abtest_power([5000, 5000], baseline=0.10, alt_lift=0.20)

   # Minimum detectable lift at 80% power
   mdl = minimum_detectable_lift([5000, 5000], baseline=0.10)

   # Required total sample size for 80% power
   n = required_sample_size(baseline=0.10, alt_lift=0.20)

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.power_calculations
   :members:
