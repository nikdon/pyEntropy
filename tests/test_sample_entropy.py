import unittest

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import TS_SAMPLE_ENTROPY
from tests._template_match_reference import _brute_match_counts


class TestEntropy(unittest.TestCase):
    def test_sample_entropy(self):
        """Calculate sample entropy for deterministic input."""
        ts = TS_SAMPLE_ENTROPY
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        np.testing.assert_allclose(sample_entropy, np.array([2.26881823, 2.11119024, 2.33537492, 1.79175947]))


class TestTemplateMatches(unittest.TestCase):
    def test_count_template_matches(self):
        """Count matching templates at each template length."""
        ts = [1, 2, 1, 2, 1]
        matches = ent._count_template_matches(ts, sample_length=2, tolerance=0.5)
        np.testing.assert_array_equal(matches, np.array([10.0, 2.0, 2.0]))

    def test_count_template_matches_unsigned_input(self):
        """Do not wrap differences around for unsigned integer input."""
        expected = np.array([10.0, 6.0, 6.0])
        as_float = ent._count_template_matches(np.array([3, 4, 3, 4, 3], dtype=float), 2, 1.5)
        as_uint = ent._count_template_matches(np.array([3, 4, 3, 4, 3], dtype=np.uint8), 2, 1.5)
        np.testing.assert_array_equal(as_float, expected)
        np.testing.assert_array_equal(as_uint, expected)

    def test_count_template_matches_agrees_with_brute_force(self):
        """Match a brute-force count of prefix and full-window pairs."""
        local_rng = np.random.default_rng(7)
        series = local_rng.standard_normal(60)
        for sample_length in (1, 2, 3):
            tol = 0.5
            counts = ent._count_template_matches(series, sample_length, tol)
            n_prefix, n_full = _brute_match_counts(series, sample_length, tol)
            self.assertEqual(counts[sample_length - 1], n_prefix)
            self.assertEqual(counts[sample_length], n_full)
