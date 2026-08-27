Contingency Table
=================

The :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable` class is
the primary entry point for analyzing frequentist binomial A/B tests. It ties
together statistical tests, confidence intervals, and lift calculations into a
chainable API. Users add two cells (control and treatment) with their successes
and trials, then call :meth:`~ab_test.frequentist_binomial.contingency.ContingencyTable.analyze`
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

API Reference
-------------

.. autoclass:: ab_test.frequentist_binomial.contingency.ContingencyTable
   :members:
   :show-inheritance:
