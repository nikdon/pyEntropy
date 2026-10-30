import unittest

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import SHANNON_ENTROPY, TIME_SERIES, TIME_SERIES_STRING


class TestEntropy(unittest.TestCase):
    def test_shannon_entropy_string(self):
        """Calculate Shannon entropy for a string-valued time series."""
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES_STRING), SHANNON_ENTROPY, rtol=1e-5)

    def test_shannon_entropy_numerical(self):
        """Calculate Shannon entropy for a numerical time series."""
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES), SHANNON_ENTROPY, rtol=1e-5)


class TestShannonEdgeCases(unittest.TestCase):
    def test_returns_python_float(self):
        """Return a built-in float for both numeric and string input."""
        self.assertIs(type(ent.shannon_entropy(TIME_SERIES)), float)
        self.assertIs(type(ent.shannon_entropy(TIME_SERIES_STRING)), float)

    def test_single_symbol_is_positive_zero(self):
        """Return +0.0 (not -0.0) when only one symbol occurs."""
        for data in ([5, 5, 5], "aaa"):
            value = ent.shannon_entropy(data)
            self.assertEqual(value, 0.0)
            self.assertEqual(np.copysign(1.0, value), 1.0)

    def test_empty_input_is_rejected(self):
        """Raise ValueError for empty input instead of returning -0.0."""
        self.assertRaises(ValueError, ent.shannon_entropy, [])
        self.assertRaises(ValueError, ent.shannon_entropy, "")
