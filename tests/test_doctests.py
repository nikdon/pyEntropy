import doctest

from pyentrp import _core, _dispersion, _fuzzy, _multiscale, _ordinal, _regularity, _sample, _validation


def load_tests(loader, tests, ignore):
    """Run the examples in the docstrings as part of the test suite."""
    for module in (_core, _dispersion, _fuzzy, _multiscale, _ordinal, _regularity, _sample, _validation):
        tests.addTests(doctest.DocTestSuite(module))
    return tests
