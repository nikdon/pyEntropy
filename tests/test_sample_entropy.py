import unittest

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pyentrp import _sample
from pyentrp import entropy as ent
from pyentrp._sample import band_blocks, count_matches_banded, count_matches_dense, count_template_matches
from tests._entropy_test_data import TS_SAMPLE_ENTROPY
from tests._template_match_reference import _brute_match_counts


class TestEntropy(unittest.TestCase):
    def test_sample_entropy(self):
        """Calculate sample entropy for deterministic input."""
        ts = TS_SAMPLE_ENTROPY
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, m=3, tolerance=0.2 * std_ts)
        np.testing.assert_allclose(sample_entropy, np.array([2.26881823, 2.11119024, 2.33537492, 1.79175947]))


class TestTemplateMatches(unittest.TestCase):
    def test_count_template_matches(self):
        """Count matching templates at each template length."""
        ts = [1, 2, 1, 2, 1]
        matches = count_template_matches(ts, sample_length=2, tolerance=0.5)
        np.testing.assert_array_equal(matches, np.array([10.0, 2.0, 2.0]))

    def test_count_template_matches_unsigned_input(self):
        """Do not wrap differences around for unsigned integer input."""
        expected = np.array([10.0, 6.0, 6.0])
        as_float = count_template_matches(np.array([3, 4, 3, 4, 3], dtype=float), 2, 1.5)
        as_uint = count_template_matches(np.array([3, 4, 3, 4, 3], dtype=np.uint8), 2, 1.5)
        np.testing.assert_array_equal(as_float, expected)
        np.testing.assert_array_equal(as_uint, expected)

    def test_count_template_matches_agrees_with_brute_force(self):
        """Match a brute-force count of prefix and full-window pairs."""
        local_rng = np.random.default_rng(7)
        series = local_rng.standard_normal(60)
        for sample_length in (1, 2, 3):
            tol = 0.5
            counts = count_template_matches(series, sample_length, tol)
            n_prefix, n_full = _brute_match_counts(series, sample_length, tol)
            self.assertEqual(counts[sample_length - 1], n_prefix)
            self.assertEqual(counts[sample_length], n_full)


def _both_strategies(x, sample_length, tolerance):
    """Return the counts of the dense and of the banded strategy, forced."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    dense, banded = np.zeros(sample_length + 1), np.zeros(sample_length + 1)
    dense[0] = banded[0] = n * (n - 1) / 2
    n_win = n - sample_length + 1
    if n_win > 1:
        count_matches_dense(x, sample_length, tolerance, dense)
        windows = sliding_window_view(x, sample_length)[:n_win]
        order = np.argsort(windows[:, 0], kind="stable")
        cols = [np.ascontiguousarray(windows[order, k]) for k in range(sample_length)]
        count_matches_banded(cols, tolerance, band_blocks(cols[0], tolerance), banded)
    return dense, banded


class TestMatchingStrategies(unittest.TestCase):
    def test_banded_and_dense_give_identical_counts(self):
        """Give the same counts with both strategies, including ties, offsets and constant series."""
        rng = np.random.default_rng(11)
        for trial in range(240):
            n = int(rng.integers(0, 260))
            x = (
                rng.standard_normal(n),
                rng.integers(0, 4, n).astype(float),  # many exact ties
                np.round(rng.standard_normal(n), 1) + 1e6,  # large offset: rounding near the tolerance
                np.cumsum(rng.standard_normal(n)),
                np.full(n, 0.3),
                np.sin(np.arange(n) / 3.0),
            )[trial % 6]
            sample_length = int(rng.integers(1, 6))
            sd = float(np.std(x)) if n else 1.0
            tolerance = float(rng.choice([0.05, 0.1, 0.2, 0.5, 3.0])) * (sd if sd > 0 else 1.0)
            dense, banded = _both_strategies(x, sample_length, tolerance)
            with self.subTest(trial=trial, n=n, sample_length=sample_length, tolerance=tolerance):
                np.testing.assert_array_equal(banded, dense)
                np.testing.assert_array_equal(count_template_matches(x, sample_length, tolerance), dense)

    def test_tolerance_exactly_on_a_distance(self):
        """Keep the strict test d < tolerance when distances equal the tolerance (including after an offset)."""
        for offset in (0.0, 1e6, -1e9):
            x = offset + np.array([0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.2, 0.3, 0.1, 0.2] * 20)
            for tolerance in (0.1, np.nextafter(0.1, 1.0), 0.2):
                dense, banded = _both_strategies(x, 3, tolerance)
                with self.subTest(offset=offset, tolerance=tolerance):
                    np.testing.assert_array_equal(banded, dense)

    def test_both_strategies_match_brute_force(self):
        """Agree with a brute-force count on the prefix and full-window pairs."""
        series = np.random.default_rng(3).standard_normal(150)
        for sample_length in (1, 2, 3, 4):
            for tolerance in (0.1, 0.3, 2.0):
                n_prefix, n_full = _brute_match_counts(series, sample_length, tolerance)
                for counts in _both_strategies(series, sample_length, tolerance):
                    self.assertEqual(counts[sample_length - 1], n_prefix)
                    self.assertEqual(counts[sample_length], n_full)

    def test_strategy_choice(self):
        """Use the banded search for typical tolerances on long series, and all pairs otherwise."""
        x = np.random.default_rng(0).standard_normal(2000)
        calls = []
        original = _sample.count_matches_banded
        try:
            _sample.count_matches_banded = lambda *a: calls.append("banded") or original(*a)
            count_template_matches(x, 3, 0.2 * np.std(x))
            count_template_matches(x, 3, 2.0 * np.std(x))
            count_template_matches(x[:300], 3, 0.2 * np.std(x))
        finally:
            _sample.count_matches_banded = original
        self.assertEqual(calls, ["banded"])


class TestSampleEntropyValidation(unittest.TestCase):
    def test_invalid_values(self):
        """Reject bad lengths, tolerances and non-finite data with ValueError."""
        ts = TS_SAMPLE_ENTROPY
        self.assertRaises(ValueError, ent.sample_entropy, ts, m=-1)
        self.assertRaises(ValueError, ent.sample_entropy, ts, m=1, tolerance=-1.0)
        self.assertRaises(ValueError, ent.sample_entropy, ts, m=1, tolerance=0.0)
        self.assertRaises(ValueError, ent.sample_entropy, ts, m=1, tolerance=np.nan)
        self.assertRaises(ValueError, ent.sample_entropy, [*ts, np.nan], m=1)
        self.assertRaises(ValueError, ent.sample_entropy, np.ones((5, 5)), m=1)

    def test_invalid_types(self):
        """Reject a non-integer sample_length with TypeError."""
        self.assertRaises(TypeError, ent.sample_entropy, TS_SAMPLE_ENTROPY, m=1.0)
        self.assertRaises(TypeError, ent.sample_entropy, TS_SAMPLE_ENTROPY, m=True)
