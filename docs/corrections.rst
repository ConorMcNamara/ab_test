Multiple Testing Corrections
============================

When an experiment evaluates several metrics (conversion, revenue, clicks) or
compares more than two variants, each test carries its own false-positive risk.
Without correction, the probability of at least one spurious significant result
grows quickly with the number of tests.

The :mod:`ab_test.corrections` module adjusts raw p-values to control either the
familywise error rate (FWER) or the false discovery rate (FDR).  All functions
accept a list of raw p-values — extracted from any analysis class — and return
adjusted p-values in the same order.

Available methods:

- **Bonferroni** — simplest FWER control; multiplies each p-value by *m*.
- **Sidak** — slightly less conservative than Bonferroni when tests are
  independent.
- **Holm** (default) — step-down procedure; uniformly more powerful than
  Bonferroni while still controlling FWER.
- **Benjamini-Hochberg** — controls FDR instead of FWER, giving substantially
  more power when many tests are performed.

Usage
-----

.. code-block:: python

   from ab_test.corrections import adjust_pvalues
   from ab_test.frequentist_binomial.contingency import ContingencyTable

   # Run two metrics through separate analyses
   ct_conv = ContingencyTable("Landing Page", "Conversion")
   ct_conv.add("Control", 100, 1000).add("Treatment", 120, 1000)
   ct_conv.analyze()

   ct_clicks = ContingencyTable("Landing Page", "Clicks")
   ct_clicks.add("Control", 200, 1000).add("Treatment", 250, 1000)
   ct_clicks.analyze()

   # Collect p-values and adjust for multiplicity
   raw = [
       ct_conv.incremental_results["p_value"],
       ct_clicks.incremental_results["p_value"],
   ]
   adjusted = adjust_pvalues(raw, method="holm")
   print(f"Adjusted p-values: {adjusted}")

   # Or use BH for FDR control with many metrics
   adjusted_fdr = adjust_pvalues(raw, method="bh")

API Reference
-------------

.. automodule:: ab_test.corrections
   :members:
