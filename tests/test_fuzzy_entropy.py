import math
import unittest

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import TIME_SERIES, TS_SAMPLE_ENTROPY


def _brute_fuzzy(series, m, tolerance, n):
    """Calculate reference FuzzyEn with centred templates and exp(-(d / r) ** n) similarity."""
    series = np.asarray(series, dtype=float)
    count = len(series) - m

    def phi(length):
        vectors = [series[i : i + length] - np.mean(series[i : i + length]) for i in range(count)]
        total = 0.0
        pairs = 0
        for i in range(count):
            for j in range(i + 1, count):
                d = np.max(np.abs(vectors[i] - vectors[j]))
                total += math.exp(-((d / tolerance) ** n))
                pairs += 1
        return total / pairs

    return -math.log(phi(m + 1) / phi(m))


class TestFuzzyEntropy(unittest.TestCase):
    def test_fuzzy_entropy_basic(self):
        """Return finite fuzzy entropy with explicit and default tolerance."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe = ent.fuzzy_entropy(signal, sample_length=2, tolerance=0.2 * np.std(signal), n=2)
        self.assertIsInstance(fe, float)
        self.assertTrue(np.isfinite(fe))
        self.assertGreater(fe, 0.0)

        fe_default = ent.fuzzy_entropy(signal, sample_length=2)
        np.testing.assert_allclose(fe, fe_default)

    def test_fuzzy_entropy_deterministic_and_sinusoidal(self):
        """Return expected fuzzy entropy for deterministic signals."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe = ent.fuzzy_entropy(signal, sample_length=2)
        np.testing.assert_allclose(fe, 0.60045352, rtol=1e-5)

        fe_ts = ent.fuzzy_entropy(TS_SAMPLE_ENTROPY, sample_length=2)
        self.assertTrue(np.isfinite(fe_ts))
        self.assertGreater(fe_ts, 0.0)

    def test_fuzzy_entropy_matches_brute_force(self):
        """Agree with an independent brute-force FuzzyEn for several parameter sets."""
        local_rng = np.random.default_rng(5)
        series = local_rng.standard_normal(70)
        for sample_length, n in ((1, 2), (2, 2), (3, 1), (2, 4)):
            with self.subTest(sample_length=sample_length, n=n):
                tol = 0.25 * np.std(series)
                expected = _brute_fuzzy(series, sample_length, tol, n)
                got = ent.fuzzy_entropy(series, sample_length=sample_length, tolerance=tol, n=n)
                np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_fuzzy_entropy_baseline_and_scale_invariance(self):
        """Ignore additive offsets and positive rescaling with the default tolerance."""
        local_rng = np.random.default_rng(21)
        series = local_rng.standard_normal(150)
        base = ent.fuzzy_entropy(series)
        np.testing.assert_allclose(ent.fuzzy_entropy(series + 5.0), base, rtol=1e-9)
        np.testing.assert_allclose(ent.fuzzy_entropy(series * 3.0), base, rtol=1e-9)

    def test_fuzzy_entropy_constant_signal(self):
        """Return zero fuzzy entropy for constant signals."""
        constant_ts = np.ones(50)
        self.assertEqual(ent.fuzzy_entropy(constant_ts), 0.0)
        self.assertEqual(ent.fuzzy_entropy(constant_ts, tolerance=0.2), 0.0)
        self.assertEqual(ent.fuzzy_entropy([5] * 20), 0.0)

    def test_fuzzy_entropy_smoothness_vs_tolerance(self):
        """Decrease fuzzy entropy as tolerance increases."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        tolerances = [0.1, 0.2, 0.3, 0.4, 0.5]
        fe_values = [ent.fuzzy_entropy(signal, sample_length=2, tolerance=tol) for tol in tolerances]
        for i in range(len(fe_values) - 1):
            self.assertGreater(fe_values[i], fe_values[i + 1])

    def test_fuzzy_entropy_varying_n(self):
        """Return finite fuzzy entropy across boundary weights."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe_n1 = ent.fuzzy_entropy(signal, sample_length=2, n=1)
        fe_n2 = ent.fuzzy_entropy(signal, sample_length=2, n=2)
        fe_n4 = ent.fuzzy_entropy(signal, sample_length=2, n=4)
        for val in (fe_n1, fe_n2, fe_n4):
            self.assertTrue(np.isfinite(val))
            self.assertGreater(val, 0.0)

    def test_fuzzy_entropy_long_series(self):
        """Handle fuzzy entropy calculations for a long time series."""
        rng_local = np.random.default_rng(999)
        long_ts = rng_local.standard_normal(1200)
        fe = ent.fuzzy_entropy(long_ts, sample_length=2)
        self.assertTrue(np.isfinite(fe))
        self.assertGreater(fe, 1.5)

    def test_fuzzy_entropy_vanished_similarity_warning(self):
        """Warn and return NaN when fuzzy similarities vanish."""
        extreme_ts = [1.0, 1000.0, 100000.0, 10000000.0, 1000000000.0]
        with self.assertWarns(RuntimeWarning):
            fe = ent.fuzzy_entropy(extreme_ts, sample_length=1, tolerance=1e-12, n=2)
        self.assertTrue(np.isnan(fe))

    def test_fuzzy_entropy_input_types(self):
        """Return equivalent fuzzy entropy for supported input and parameter types."""
        list_data = [1, 3, 2, 5, 4, 7, 6, 8, 2, 4]
        np_float = np.array(list_data, dtype=float)
        np_int = np.array(list_data, dtype=int)
        res_list = ent.fuzzy_entropy(list_data, sample_length=2)
        res_float = ent.fuzzy_entropy(np_float, sample_length=2)
        res_int = ent.fuzzy_entropy(np_int, sample_length=2)
        self.assertAlmostEqual(res_list, res_float)
        self.assertAlmostEqual(res_list, res_int)
        self.assertAlmostEqual(res_list, ent.fuzzy_entropy(list_data, sample_length=np.int64(2)))

    def test_fuzzy_entropy_invalid_values(self):
        """Reject out-of-range parameters, bad shapes and non-finite data with ValueError."""
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, sample_length=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, sample_length=-1)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, n=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, n=-2)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, tolerance=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, tolerance=-0.1)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, tolerance=float("nan"))
        self.assertRaises(ValueError, ent.fuzzy_entropy, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent.fuzzy_entropy, [1, 2, 3], sample_length=2)
        self.assertRaises(ValueError, ent.fuzzy_entropy, [1.0, np.nan, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ent.fuzzy_entropy, [1.0, np.inf, 3.0, 4.0, 5.0])

    def test_fuzzy_entropy_invalid_types(self):
        """Reject non-integer sample_length with TypeError."""
        self.assertRaises(TypeError, ent.fuzzy_entropy, TIME_SERIES, sample_length=2.5)
        self.assertRaises(TypeError, ent.fuzzy_entropy, TIME_SERIES, sample_length=2.0)
        self.assertRaises(TypeError, ent.fuzzy_entropy, TIME_SERIES, sample_length=True)


class TestFuzzySign(unittest.TestCase):
    def test_constant_series_explicit_tolerance_is_positive_zero(self):
        """Return +0.0 (not -0.0) for a constant series with an explicit tolerance."""
        value = ent.fuzzy_entropy(np.ones(50), tolerance=0.2)
        self.assertEqual(np.copysign(1.0, value), 1.0)
