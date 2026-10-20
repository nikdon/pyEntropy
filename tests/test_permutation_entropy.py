import unittest

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import PERM_ENTROPY_BANDT, RANDOM_TIME_SERIES, TS_SAMPLE_ENTROPY


class TestEntropy(unittest.TestCase):
    def test_permutation_entropy(self):
        """Calculate permutation entropy with and without normalization."""
        np.testing.assert_allclose(
            ent.permutation_entropy(PERM_ENTROPY_BANDT, order=2, delay=1),
            0.918,
            rtol=1e-3,
        )

        np.testing.assert_allclose(
            ent.permutation_entropy(PERM_ENTROPY_BANDT, order=3, delay=1),
            1.522,
            rtol=1e-3,
        )

        # Assert that a fully random vector has an entropy of 0.99999...
        np.testing.assert_allclose(
            ent.permutation_entropy(RANDOM_TIME_SERIES, order=3, delay=1, normalize=True),
            0.999,
            rtol=1e-3,
        )

    def test_weighted_permutation_entropy(self):
        """Calculate weighted permutation entropy for supported parameters."""
        np.testing.assert_allclose(
            ent.weighted_permutation_entropy(PERM_ENTROPY_BANDT, order=2, delay=1),
            0.913,
            rtol=1e-3,
        )

        np.testing.assert_allclose(
            ent.weighted_permutation_entropy(PERM_ENTROPY_BANDT, order=3, delay=1),
            1.414,
            rtol=1e-3,
        )

        # Assert that a fully random vector has an entropy of 0.99999...
        np.testing.assert_allclose(
            ent.weighted_permutation_entropy(RANDOM_TIME_SERIES, order=3, delay=1, normalize=True),
            0.999,
            rtol=1e-3,
        )

    def test_multiscale_permutation_entropy(self):
        """Calculate multiscale permutation entropy for deterministic input."""
        np.testing.assert_array_equal(
            np.round(ent.multiscale_permutation_entropy(TS_SAMPLE_ENTROPY, 3, 5, 2), 4),
            np.array([2.4699, 2.5649]),
        )
