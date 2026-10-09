Contingency Table
=================

The :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable` class is
the primary entry point for analyzing frequentist binomial A/B tests. It ties
together statistical tests, confidence intervals, and lift calculations into a
chainable API. Users add cells (a control and one or more variants) with their
successes and trials, then call :meth:`~ab_test.frequentist_binomial.contingency.ContingencyTable.analyze`
to get a formatted results table. The class supports relative, absolute,
incremental, ROAS, and revenue lift types. Individual cell analysis is also
available via
:meth:`~ab_test.frequentist_binomial.contingency.ContingencyTable.analyze_individually`,
which computes per-cell success rates and confidence intervals. Results can be
exported to pandas, polars, PySpark, or NumPy formats, and serialized to JSON
for storage.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.contingency import ContingencyTable

   ct = ContingencyTable("My Experiment", "Conversion Rate")
   ct.add("Control", successes=100, trials=1000)
   ct.add("Treatment", successes=130, trials=1000)
   print(ct.analyze(lift="relative", test_method="score"))

   # Individual cell analysis
   print(ct.analyze_individually(conf_int_method="wilson"))

Three or More Variants
----------------------

With three or more cells (an A/B/n test), ``analyze()`` reports:

* an **omnibus test** that every variant shares one success rate: the k x 2
  power-divergence test matching ``test_method`` (the G-test for
  ``"likelihood"``), or Pearson's chi-squared otherwise;
* **pairwise comparisons**, each run with ``test_method`` and
  ``conf_int_method`` exactly as a two-variant analysis would.
  ``comparisons="control"`` (the default) compares each variant against the
  first cell added; ``comparisons="all"`` compares every pair;
* **adjusted p-values**, using Holm's method by default (``correction`` takes
  any method of :func:`~ab_test.corrections.adjust_pvalues`), alongside the raw
  p-values;
* **simultaneous confidence intervals**: Bonferroni intervals at level
  ``1 - alpha / m`` for ``m`` comparisons, so all of them hold together with
  probability ``1 - alpha``.

.. code-block:: python

   ct = ContingencyTable("Checkout", "conversion")
   ct.add("A", 100, 1000).add("B", 120, 1000).add("C", 140, 1000)
   print(ct.analyze())                                     # B vs A, C vs A
   print(ct.analyze(comparisons="all", correction="bh"))   # every pair, FDR control

   ct.incremental_results["omnibus"]["p_value"]
   ct.incremental_results["comparisons"]["C vs A"]["p_value"]   # adjusted
   ct.plot(is_individual=False)                            # one row per comparison

With two cells the output and ``incremental_results`` are unchanged, and
``comparisons`` and ``correction`` are ignored.

API Reference
-------------

.. autoclass:: ab_test.frequentist_binomial.contingency.ContingencyTable
   :members:
   :show-inheritance:
