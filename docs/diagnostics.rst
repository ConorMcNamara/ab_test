Pre-Analysis Diagnostics
========================

Run these checks before trusting experiment results. A significant sample
ratio mismatch, for example, can indicate a broken randomisation layer, a data
pipeline bug, or differential attrition — any of which can invalidate the
entire analysis regardless of the p-value it produces.

Sample Ratio Mismatch (SRM)
---------------------------

A chi-squared goodness-of-fit test that compares the observed traffic split
against the intended split. If you planned a 50/50 split but observed 48/52,
was that just sampling noise or a systematic problem?

Common causes of SRM:

- **Broken randomiser** — the hashing or bucketing logic is not uniform.
- **Differential bot filtering** — bots are removed from one group but not the
  other.
- **Triggered-analysis mismatch** — the event that triggers assignment differs
  from the event used to count users.
- **Lossy joins** — a pipeline join drops rows asymmetrically.

Usage
-----

.. code-block:: python

   from ab_test.diagnostics import srm_test

   # Check a two-group 50/50 experiment
   stat, pvalue = srm_test([4800, 5200])
   print(f"SRM chi2={stat:.2f}, p={pvalue:.4f}")

   # Check a three-group experiment with a 50/25/25 split
   stat, pvalue = srm_test(
       [5000, 2600, 2400],
       expected_proportions=[0.50, 0.25, 0.25],
   )

API Reference
-------------

.. automodule:: ab_test.diagnostics
   :members:
