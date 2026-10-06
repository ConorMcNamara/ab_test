Equivalence Testing (ROPE)
==========================

The Bayesian counterpart to TOST. Instead of two one-sided tests, it draws
from the Beta posterior of each variant and computes the posterior probability
that the lift falls inside a Region of Practical Equivalence (ROPE),
``[-delta, delta]``. The variants are declared equivalent when that
probability is at least ``threshold`` (0.95 by default).

The result also reports the probabilities that the lift lies above or below
the ROPE, which tell you which way the evidence points when equivalence is
not established.

Usage
-----

.. code-block:: python

   from ab_test.bayesian_binomial.equivalence import bayes_equivalence_test

   result = bayes_equivalence_test(
       successes=[1000, 1010],
       trials=[10_000, 10_000],
       alphas=[1, 1],  # Beta(1, 1) priors
       betas=[1, 1],
       delta=0.02,
       seed=0,
   )
   result["equivalent"]       # True
   result["prob_equivalent"]  # posterior P(-delta <= lift <= delta)

``lift`` sets the scale of ``delta`` and also accepts ``"incremental"``,
``"roas"``, ``"revenue"`` and ``"cpa"`` (pass ``spend`` and ``msrp`` as
needed). Results come from posterior sampling, so pass ``seed`` for
reproducibility.

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.equivalence
   :members:
