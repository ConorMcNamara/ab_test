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

With ``lift="incremental"``, ``"roas"`` or ``"revenue"``, per-segment effects
are reported in those units, but Cochran's Q and the pairwise comparisons use
the risk difference: scaling each segment by its own size, spend or price would
make equal rate effects look different. ``lift="cpa"`` is not supported; use
``"roas"``.

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

Two Arms per Segment Only
-------------------------

Each segment must be a two-cell table: control and treatment. Other classes
in this library compare three or more variants (for example
:class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`), but this
one deliberately does not: a three-cell segment raises a ``ValueError``. The
reason is that the question it answers changes with more variants:

* **There is no single effect per segment.** With variants B and C, each
  segment has one effect per variant (B vs control and C vs control), so
  "does the treatment effect differ across segments?" becomes one
  heterogeneity question per variant, plus the question of whether B and C
  differ from each other differently in different segments.
* **There is no single omnibus test.** With several variants, the
  multi-variant classes add a test that every variant shares one rate (for
  example a chi-squared test). The natural counterpart here is a test of a
  variant-by-segment interaction, whose degrees of freedom and pairwise
  follow-ups multiply with both the number of variants and the number of
  segments, and whose answer depends on which contrasts matter for the
  decision.
* **Multiplicity compounds.** Correcting every pairwise segment comparison for
  every variant quickly leaves little power for any of them.

For an A/B/n test with segments, run one analysis per variant, building each
segment's table from the control and that variant, and adjust the per-variant
p-values together with :func:`~ab_test.corrections.adjust_pvalues`:

.. code-block:: python

   from ab_test.corrections import adjust_pvalues

   results = {}
   for variant in ["B", "C"]:
       segments = [
           ContingencyTable(name, "converted")
           .add("Control", *counts[name]["Control"])
           .add(variant, *counts[name][variant])
           for name in segment_names
       ]
       did = DiffInDiff(*segments)
       did.analyze()
       results[variant] = did.heterogeneity_results["Q_pvalue"]

   adjusted = adjust_pvalues(list(results.values()), method="holm")

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.diff_in_diff
   :members:
