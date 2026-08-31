Difference-in-Differences
=========================

When an experiment includes distinct segments (demographics, device type,
region), you may want to test whether the treatment effect itself varies across
those segments.  The
:class:`~ab_test.frequentist_binomial.diff_in_diff.DiffInDiff` class answers
this question by comparing per-segment treatment effects.

It computes:

1. **Per-segment effects** with Wald confidence intervals
2. **Cochran's Q omnibus test** for effect heterogeneity — does the treatment
   effect differ *at all* across segments?
3. **All pairwise DiD comparisons** with multiplicity correction via
   :func:`~ab_test.corrections.adjust_pvalues`

This is complementary to
:class:`~ab_test.frequentist_binomial.stratified.StratifiedContingencyTable`,
which *pools* strata to produce a single treatment effect under the assumption
of homogeneity.  ``DiffInDiff`` explicitly tests whether that assumption holds.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.contingency import ContingencyTable
   from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff

   men = ContingencyTable("Men", "converted")
   men.add("Control", successes=100, trials=1000)
   men.add("Treatment", successes=130, trials=1000)

   women = ContingencyTable("Women", "converted")
   women.add("Control", successes=120, trials=1000)
   women.add("Treatment", successes=125, trials=1000)

   test = DiffInDiff(men, women)
   print(test.analyze(lift="absolute", alpha=0.05, correction="holm"))
   test.plot(lift="absolute", alpha=0.05)

Segments must be **independent** — the same user should not appear in multiple
segment tables.  If your segments overlap, the variance estimates (and therefore
the p-values) will be anti-conservative.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.diff_in_diff
   :members:
