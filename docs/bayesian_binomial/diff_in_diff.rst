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
   standard deviation of the true treatment effects across segments, from a
   normal random-effects model with a half-Cauchy prior on tau. Within-segment
   sampling noise is not counted, so the interval reaches zero when segments
   agree. With few segments the data say little about tau, so expect a wide
   interval.
3. **All pairwise DiD comparisons** with posterior probabilities P(lift_i > lift_j)

With ``lift="incremental"``, ``"roas"`` or ``"revenue"``, per-segment effects
are reported in those units, but tau and the pairwise comparisons use the risk
difference: scaling each segment by its own size, spend or price would make
equal rate effects look different. ``lift="cpa"`` is not supported; use
``"roas"``.

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
* **There is no single summary.** With several variants, the multi-variant
  classes report the probability each variant is best. Here the natural
  counterpart is a variant-by-segment interaction, whose comparisons multiply
  with both the number of variants and the number of segments, and which
  contrasts matter depends on the decision being made.
* **Comparisons multiply.** Every pairwise segment comparison for every
  variant is another posterior probability to read, and acting on whichever
  crosses a threshold first inflates false findings.

For an A/B/n test with segments, run one analysis per variant, building each
segment's table from the control and that variant:

.. code-block:: python

   for variant in ["B", "C"]:
       segments = [
           BayesianContingencyTable(name, "converted")
           .add("Control", *counts[name]["Control"], 1, 1)
           .add(variant, *counts[name][variant], 1, 1)
           for name in segment_names
       ]
       did = BayesianDiffInDiff(*segments)
       print(did.analyze())

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.diff_in_diff
   :members:
