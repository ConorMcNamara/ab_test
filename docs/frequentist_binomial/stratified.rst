Stratified Testing
==================

When experimental units fall into known subgroups (device type, geography,
acquisition channel), stratified analysis pools within-stratum estimates to
produce a single treatment effect and p-value. This controls for confounding
due to the stratification variable and can reduce variance when stratum-specific
success rates differ, translating to higher statistical power without requiring
more traffic.

The :class:`~ab_test.frequentist_binomial.stratified.StratifiedContingencyTable`
class collects per-stratum 2×2 tables via :meth:`add` and produces a pooled
analysis using the Cochran-Mantel-Haenszel (CMH) framework. The CMH test
provides the overall p-value, Mantel-Haenszel pooling gives the pooled effect
estimate and confidence interval (the MH risk ratio with the Greenland-Robins
variance for relative lift, the MH risk difference with Sato's variance
otherwise, so strata with zero successes need no correction), and the
Breslow-Day test checks whether the treatment effect is consistent across
strata.

Stratified analysis is complementary to
:class:`~ab_test.frequentist_binomial.cupac.CupacExperiment`: CUPAC requires
per-user covariate data, while stratification works with aggregate counts
grouped by a categorical variable.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.stratified import (
       StratifiedContingencyTable,
       stratified_power,
   )

   # Build the stratified table
   st = StratifiedContingencyTable("Landing Page", "Conversion Rate")
   st.add("Control", 50, 500, stratum="mobile")
   st.add("Treatment", 70, 500, stratum="mobile")
   st.add("Control", 80, 400, stratum="desktop")
   st.add("Treatment", 100, 400, stratum="desktop")

   # Pooled analysis with CMH p-value and Breslow-Day homogeneity check
   print(st.analyze(lift="relative", alpha=0.05))

   # Per-stratum breakdown
   print(st.analyze_by_stratum(lift="relative"))

   # Power calculation for a stratified design
   power = stratified_power(
       strata_sizes=[(500, 500), (400, 400)],
       baseline_rates=[0.10, 0.20],
       alt_lift=0.15,
       alpha=0.05,
       lift="relative",
   )
   print(f"Power: {power:.1%}")

Three or More Variants
----------------------

Add any number of groups; the first one added is the control. With three or
more, ``analyze()`` reports:

* an **omnibus test** that every group shares one rate across strata: the
  generalized Cochran-Mantel-Haenszel test of general association
  (:func:`~ab_test.frequentist_binomial.stratified.generalized_cmh_test`,
  ``k - 1`` degrees of freedom). With two groups it equals the CMH test;
* **pairwise comparisons**, each computed exactly as a two-group analysis of
  that pair: Mantel-Haenszel pooled effect, CMH p-value, and the Breslow-Day
  homogeneity p-value for that pair. ``comparisons="control"`` (the default)
  compares each group with the control, and ``comparisons="all"`` compares
  every pair;
* **adjusted p-values**, with Holm's method by default (``correction`` takes any
  method of :func:`~ab_test.corrections.adjust_pvalues`), alongside the raw
  CMH p-values;
* **simultaneous confidence intervals**: Bonferroni intervals at level
  ``1 - alpha / m`` for ``m`` comparisons.

Scaled lifts (incremental, ROAS, revenue, CPA) are all expressed over the
largest group's total trials, so identical rate differences give identical
values; the output says which scale. The results are stored in
``comparison_results``.

.. code-block:: python

   st = StratifiedContingencyTable("Landing Page", "Conversion Rate")
   for stratum, rates in {"mobile": (50, 62, 70), "desktop": (80, 90, 104)}.items():
       for cell, successes in zip(("A", "B", "C"), rates):
           st.add(cell, successes, 500, stratum=stratum)
   print(st.analyze(comparisons="all"))
   st.comparison_results["omnibus"]["p_value"]

The Breslow-Day p-values are per comparison and not adjusted. Per-stratum
output (:meth:`analyze_by_stratum` and :meth:`plot`) shows one comparison
across strata, so it stays two-group only and raises with three or more
groups: build a two-group table for the pair you want to see by stratum.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.stratified
   :members:
   :exclude-members: _mh_odds_ratio
