Confidence Intervals
====================

This module computes confidence intervals on the lift between two binomial
proportions. The primary method uses test inversion via binary search --
inverting a significance test (score, likelihood ratio, or z-test) to find the
narrowest interval that is consistent with the data at the given significance
level. This approach produces tighter, better-calibrated intervals than
closed-form alternatives.

For individual proportions (rather than the lift between two groups), several
closed-form methods are available: Wilson, Agresti-Coull, Jeffreys,
Clopper-Pearson, Wald, and the Delta method. The Wilson interval is generally
recommended for individual cells due to its good coverage properties across
a wide range of sample sizes and proportions. The Clopper-Pearson interval is
exact but tends to be conservative (wider than necessary).

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.confidence_intervals import (
       confidence_interval,
       individual_confidence_interval,
   )

   # CI on the relative lift between two groups
   lb, ub = confidence_interval([1000, 1000], [100, 130], lift="relative")
   print(f"95% CI: [{lb:.4f}, {ub:.4f}]")

   # CI on an individual proportion
   lb, ub = individual_confidence_interval(s=130, n=1000, method="wilson")

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.confidence_intervals
   :members:
