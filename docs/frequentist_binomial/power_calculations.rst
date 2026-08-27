Power & Sample Size
===================

This module provides power analysis for frequentist binomial A/B tests. It
answers three common pre-experiment questions: how much power does a given
sample size provide, what is the minimum detectable lift for a given sample
size, and how many observations are needed to detect a given lift with adequate
power.

All functions use binary search and delegate to
:func:`~ab_test.frequentist_binomial.power_calculations.score_power` by
default, which computes power via the noncentrality parameter of a noncentral
chi-squared distribution. A custom power function can be passed to any of
these functions -- for example,
:func:`~ab_test.frequentist_binomial.cupac.cupac_adjusted_power` to account
for CUPAC variance reduction.

Both relative and absolute lift are supported. When more than two groups are
provided, the two smallest groups are used, as they govern overall power.

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
