Equivalence Testing (TOST)
==========================

A standard significance test asks whether two variants *differ*. Failing to
reject that null is not evidence that they are the same — the test may simply
be underpowered. Equivalence testing flips the question: it asks whether the
difference is *small enough to ignore*, within a margin ``delta`` you choose
up front.

The Two One-Sided Tests (TOST) procedure runs two one-sided tests:

1. Non-inferiority: H0: diff <= -delta vs. H1: diff > -delta
2. Non-superiority: H0: diff >= delta vs. H1: diff < delta

The variants are declared equivalent when both are rejected, i.e. when the
larger of the two one-sided p-values is at most ``alpha``.

Typical uses include confirming that a refactor, infrastructure migration or
cost-saving change did not move a metric.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.equivalence import tost_test

   result = tost_test(
       trials=[10_000, 10_000],
       successes=[1000, 1010],
       delta=0.02,  # +/- 2 percentage points
   )
   result["equivalent"]  # True
   result["p_value"]     # max of the two one-sided p-values

``lift`` sets the scale of ``delta`` (``"absolute"`` by default, or
``"relative"``), and ``method`` selects the underlying test: ``"score"``
(default), ``"likelihood"`` or ``"z"``.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.equivalence
   :members:
