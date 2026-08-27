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
provides the overall p-value, inverse-variance weighting gives the pooled
effect estimate and confidence interval, and the Breslow-Day test checks
whether the treatment effect is consistent across strata.

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

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.stratified
   :members:
   :exclude-members: _mh_odds_ratio
