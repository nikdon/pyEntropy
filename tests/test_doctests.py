import doctest
import importlib
import pkgutil

import pyentrp


def load_tests(loader, tests, ignore):
    """Run the examples in the docstrings of every pyentrp module as part of the test suite."""
    for info in pkgutil.iter_modules(pyentrp.__path__):
        module = importlib.import_module(f"pyentrp.{info.name}")
        if module.__doc__ is not None or any(doctest.DocTestFinder().find(module)):
            tests.addTests(doctest.DocTestSuite(module))
    return tests
