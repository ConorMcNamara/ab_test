Difference-in-Differences (Bayesian)
=====================================

When an experiment includes distinct segments (demographics, device type,
region), you may want to test whether the treatment effect itself varies across
those segments.  The
:class:`~ab_test.bayesian_binomial.diff_in_diff.BayesianDiffInDiff` class
answers this question using posterior sampling to compare per-segment treatment
effects.

It computes:

1. **Per-segment effects** with credible intervals and P(Treatment > Control)
2. **Between-segment heterogeneity (tau)** — the posterior distribution of the
   standard deviation of treatment effects across segments
3. **All pairwise DiD comparisons** with posterior probabilities P(lift_i > lift_j)

This is the Bayesian counterpart to
:class:`~ab_test.frequentist_binomial.diff_in_diff.DiffInDiff`, which uses
Cochran's Q and Wald confidence intervals.  The Bayesian version replaces
p-values with posterior probabilities and provides the full posterior
distribution of heterogeneity rather than a single test statistic.

Usage
-----

.. code-block:: python

   from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
   from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff

   men = BayesianContingencyTable("Men", "converted")
   men.add("Control", successes=100, trials=1000, alpha=1, beta=1)
   men.add("Treatment", successes=130, trials=1000, alpha=1, beta=1)

   women = BayesianContingencyTable("Women", "converted")
   women.add("Control", successes=120, trials=1000, alpha=1, beta=1)
   women.add("Treatment", successes=125, trials=1000, alpha=1, beta=1)

   test = BayesianDiffInDiff(men, women)
   print(test.analyze(lift="absolute", confidence_level=0.95))
   test.plot(lift="absolute", confidence_level=0.95)

Segments must be **independent** — the same user should not appear in multiple
segment tables.  If your segments overlap, the posterior estimates will be
overconfident.

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.diff_in_diff
   :members:
