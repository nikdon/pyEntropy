import math
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


class TestPermutationValidation(unittest.TestCase):
    def test_invalid_values(self):
        """Reject bad orders, delays, short series and non-finite data with ValueError."""
        for func in (ent.permutation_entropy, ent.weighted_permutation_entropy):
            self.assertRaises(ValueError, func, PERM_ENTROPY_BANDT, order=0)
            self.assertRaises(ValueError, func, PERM_ENTROPY_BANDT, delay=0)
            self.assertRaises(ValueError, func, PERM_ENTROPY_BANDT, delay=-1)
            self.assertRaises(ValueError, func, [1.0, 2.0], order=3)
            self.assertRaises(ValueError, func, [1.0, np.nan, 2.0, 3.0, 1.0], order=3)
            self.assertRaises(ValueError, func, np.ones((5, 5)), order=3)
        self.assertRaises(ValueError, ent.multiscale_permutation_entropy, TS_SAMPLE_ENTROPY, 3, 1, 0)
        self.assertRaises(ValueError, ent.multiscale_permutation_entropy, [1.0, np.inf, 2.0, 3.0], 2, 1, 1)

    def test_invalid_types(self):
        """Reject non-integer order, delay and scale with TypeError."""
        for func in (ent.permutation_entropy, ent.weighted_permutation_entropy):
            self.assertRaises(TypeError, func, PERM_ENTROPY_BANDT, order=3.0)
            self.assertRaises(TypeError, func, PERM_ENTROPY_BANDT, delay=True)
        self.assertRaises(TypeError, ent.multiscale_permutation_entropy, TS_SAMPLE_ENTROPY, 3, 1, 2.0)

    def test_shortest_valid_series(self):
        """Give +0.0 (not -0.0) when only one pattern is observed."""
        for value in (
            ent.permutation_entropy([3.0, 1.0, 2.0], order=3),
            ent.weighted_permutation_entropy([1.0, 2.0, 3.0, 4.0, 5.0], order=3),
        ):
            self.assertEqual(value, 0.0)
            self.assertEqual(math.copysign(1.0, value), 1.0)

    def test_order_one_normalized_is_zero(self):
        """Return 0 for order 1 with normalize=True instead of dividing by log2(1!) = 0."""
        for func in (ent.permutation_entropy, ent.weighted_permutation_entropy):
            value = func(PERM_ENTROPY_BANDT, order=1, normalize=True)
            self.assertEqual(value, 0.0)
            self.assertEqual(math.copysign(1.0, value), 1.0)


class TestMultiscalePermutationShortScales(unittest.TestCase):
    def test_too_short_scales_give_nan(self):
        """Return NaN with a warning for scales whose coarse series is too short, and keep the others."""
        series = RANDOM_TIME_SERIES[:20]
        with self.assertWarns(RuntimeWarning):
            res = ent.multiscale_permutation_entropy(series, 3, 2, 6)
        # scale k leaves 20 // k samples; m=3, delay=2 needs at least 5
        self.assertEqual(res.shape, (6,))
        for k in range(1, 5):
            self.assertEqual(res[k - 1], ent.permutation_entropy(ent.util_granulate_time_series(series, k), 3, 2))
        self.assertTrue(np.all(np.isnan(res[4:])))


class TestPermutationOrderLimit(unittest.TestCase):
    def test_order_above_twenty_is_rejected(self):
        """Reject orders whose pattern ids would overflow int64 (21! > 2**63)."""
        x = RANDOM_TIME_SERIES[:60]
        self.assertTrue(np.isfinite(ent.permutation_entropy(x, order=20)))
        self.assertTrue(np.isfinite(ent.weighted_permutation_entropy(x, order=20)))
        self.assertRaises(ValueError, ent.permutation_entropy, x, order=21)
        self.assertRaises(ValueError, ent.weighted_permutation_entropy, x, order=21)
        self.assertRaises(ValueError, ent.multiscale_permutation_entropy, x, 21, 1, 1)


class TestLargeIntegers(unittest.TestCase):
    """Integers above 2**53 are not exactly representable in float64 and must be compared as integers."""

    # 2**53 + 1 rounds to 2**53 in float64, so a float conversion would make the series constant
    SERIES = (2**53, 2**53 + 1, 2**53, 2**53 + 1)
    # order 2 sees the patterns up, down, up
    EXPECTED = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))

    def test_permutation_entropy_keeps_large_integers_apart(self):
        """Count the two patterns of large integers that float64 would merge."""
        for data in (self.SERIES, np.array(self.SERIES, dtype=np.int64), np.array(self.SERIES, dtype=np.uint64)):
            self.assertAlmostEqual(ent.permutation_entropy(data, order=2), self.EXPECTED, places=12)

    def test_multiscale_permutation_entropy_scale_one_matches(self):
        """Use the integer series itself at scale 1."""
        res = ent.multiscale_permutation_entropy(self.SERIES, 2, 1, 1)
        self.assertAlmostEqual(res[0], self.EXPECTED, places=12)

    def test_weighted_permutation_entropy_uses_integer_patterns(self):
        """Build weighted patterns from the integers, not from their float64 roundings."""
        b = 2**53
        series = [b + 1, b, b + 8, b + 16]
        # integers: windows (b + 1, b, b + 8) and (b, b + 8, b + 16) are two different patterns.
        # float64 rounds b + 1 to b, which ties the first window and merges it with the second.
        # The weights are computed in float: 3 * sum(d**2) - sum(d)**2 = 128 and 384.
        expected = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
        self.assertAlmostEqual(ent.weighted_permutation_entropy(series, order=3), expected, places=12)
        self.assertEqual(ent.weighted_permutation_entropy(np.array(series, dtype=float), order=3), 0.0)
