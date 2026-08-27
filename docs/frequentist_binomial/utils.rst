Utilities
=========

Helper functions used internally by the statistical tests and power
calculations. These include input validation for two-group experiments,
maximum likelihood estimation under null and alternative hypotheses,
conversion from composite hypotheses (e.g. "the treatment lifts by 10%") to
simple hypotheses (specific per-group success rates) for power calculations,
observed lift computation, and Wilson significance.

The :func:`~ab_test.frequentist_binomial.utils.wilson_significance` function
transforms a p-value into a more interpretable scale: ``log10(alpha / pval)``.
Positive values indicate statistical significance, and each unit increase
corresponds to a 10x decrease in p-value.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.utils import observed_lift, wilson_significance

   # Compute relative lift
   lift = observed_lift([1000, 1000], [100, 130], lift="relative")

   # Wilson significance: positive means significant at alpha
   w = wilson_significance(pval=0.03, alpha=0.05)

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.utils
   :members:
