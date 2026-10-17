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
