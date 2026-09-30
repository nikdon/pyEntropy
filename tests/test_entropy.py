import math
import unittest
import warnings
from collections import Counter

import numpy as np

from pyentrp import entropy as ent

rng = np.random.default_rng(1234567)

TIME_SERIES = [1, 1, 1, 2, 3, 4, 5]
TIME_SERIES_STRING = "1112345"
SHANNON_ENTROPY = 2.12809

# fmt: off
TS_SAMPLE_ENTROPY = [
    1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3, 9, 1, 2, 3, 4, 2, 9, 6, 7, 4, 9, 2, 9, 9, 6,
    5, 1, 3, 8, 1, 5, 3, 8, 4, 1, 2, 2, 1, 6, 5, 3, 6, 5, 4, 8, 9, 6, 7, 5, 3, 2, 5, 4, 2, 5, 1, 6, 5,
    3, 5, 6, 7, 8, 5, 2, 8, 6, 3, 8, 2, 7, 1, 7, 3, 5, 6, 2, 1, 3, 7, 3, 5, 3, 7, 6, 7, 7, 2, 3, 1, 7,
    8
]
# fmt: on

PERM_ENTROPY_BANDT = [4, 7, 9, 10, 6, 11, 3]

RANDOM_TIME_SERIES = rng.random(1000)


# ---------------------------------------------------------------------------
# Slow but correct reference implementations used as oracles
# ---------------------------------------------------------------------------


def _brute_match_counts(series, sample_length, tolerance):
    """Count matching pairs of full windows: (prefix of length L - 1, full length L)."""
    n = len(series)
    n_prefix = 0
    n_full = 0
    for i in range(n - sample_length):
        for j in range(i + 1, n - sample_length + 1):
            prefix_close = all(abs(series[i + k] - series[j + k]) < tolerance for k in range(sample_length - 1))
            if prefix_close:
                n_prefix += 1
                if abs(series[i + sample_length - 1] - series[j + sample_length - 1]) < tolerance:
                    n_full += 1
    return n_prefix, n_full


def _brute_rcmse(series, sample_length, scale, tolerance):
    """Reference RCMSE: pool counts over all shifted coarse-grained series at each scale."""
    out = []
    for k in range(1, scale + 1):
        n_prefix = n_full = 0
        for j in range(k):
            coarse = ent.util_granulate_time_series(series[j:], k)
            a, b = _brute_match_counts(coarse, sample_length, tolerance)
            n_prefix += a
            n_full += b
        out.append(-math.log(n_full / n_prefix))
    return np.array(out)


def _brute_fuzzy(series, m, tolerance, n):
    """Reference FuzzyEn with centred templates and exp(-(d / r) ** n) similarity."""
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


def _entropy_bits(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Existing functionality
# ---------------------------------------------------------------------------


class TestEntropy(unittest.TestCase):
    def test_shannon_entropy_string(self):
        """Calculate Shannon entropy for a string-valued time series."""
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES_STRING), SHANNON_ENTROPY, rtol=1e-5)

    def test_shannon_entropy_numerical(self):
        """Calculate Shannon entropy for a numerical time series."""
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES), SHANNON_ENTROPY, rtol=1e-5)

    def test_sample_entropy(self):
        """Calculate sample entropy for deterministic input."""
        ts = TS_SAMPLE_ENTROPY
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        np.testing.assert_allclose(sample_entropy, np.array([2.26881823, 2.11119024, 2.33537492, 1.79175947]))

    def test_multiscale_entropy(self):
        """Calculate multiscale entropy across the requested scales."""
        multi_scale_entropy = ent.multiscale_entropy(RANDOM_TIME_SERIES, 4, maxscale=4)
        np.testing.assert_allclose(
            multi_scale_entropy,
            np.array([3.178054, 3.178054, 2.890372, 3.401197]),
            rtol=1e-6,
        )

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

    def test_util_pattern_space(self):
        """Construct the expected embedded pattern space."""
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 0, 2)
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 10, 20)
        np.testing.assert_array_equal(
            ent.util_pattern_space(TIME_SERIES, 2, 3),
            np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]),
        )

    def test_composite_multiscale_entropy(self):
        """Calculate composite multiscale entropy across scales."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.composite_multiscale_entropy(signal, sample_length=3, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19283124, 0.94056984]))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestValidationHelpers(unittest.TestCase):
    def test_validate_series_returns_float_array(self):
        """Convert list and integer input to a 1D float array."""
        out = ent._validate_series([1, 2, 3])
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(out.shape, (3,))

    def test_validate_series_rejects_bad_shape_and_values(self):
        """Reject multi-dimensional and non-finite input."""
        self.assertRaises(ValueError, ent._validate_series, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent._validate_series, [1.0, np.nan])
        self.assertRaises(ValueError, ent._validate_series, [1.0, np.inf])
        self.assertRaises(ValueError, ent._validate_series, [1.0, -np.inf])

    def test_check_integer(self):
        """Accept Python and NumPy integers; reject bools and floats by type, small values by value."""
        ent._check_integer("x", 3, 1)
        ent._check_integer("x", np.int64(3), 1)
        ent._check_integer("x", np.int32(1), 1)
        self.assertRaises(TypeError, ent._check_integer, "x", True, 1)
        self.assertRaises(TypeError, ent._check_integer, "x", np.bool_(True), 1)
        self.assertRaises(TypeError, ent._check_integer, "x", 2.0, 1)
        self.assertRaises(TypeError, ent._check_integer, "x", "2", 1)
        self.assertRaises(ValueError, ent._check_integer, "x", 0, 1)
        self.assertRaises(ValueError, ent._check_integer, "x", np.int64(-4), 1)


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


# ---------------------------------------------------------------------------
# Refined Composite Multiscale Entropy
# ---------------------------------------------------------------------------


class TestRCMSE(unittest.TestCase):
    def test_rcmse_scale_1_equivalence(self):
        """Match sample entropy when RCMSE uses a single scale."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        se = ent.sample_entropy(signal, sample_length=3)[-1]
        res = ent.rcmse(signal, sample_length=3, scale=1)
        self.assertEqual(res.shape, (1,))
        np.testing.assert_allclose(res[0], se)

        se_ts = ent.sample_entropy(TS_SAMPLE_ENTROPY, sample_length=4)[-1]
        res_ts = ent.rcmse(TS_SAMPLE_ENTROPY, sample_length=4, scale=1)
        np.testing.assert_allclose(res_ts[0], se_ts)

    def test_rcmse_sinusoidal(self):
        """Return expected RCMSE values for a sinusoidal signal."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.rcmse(signal, sample_length=3, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19237189, 0.90078655]))

    def test_rcmse_random(self):
        """Return finite positive RCMSE values for random input."""
        res = ent.rcmse(RANDOM_TIME_SERIES, sample_length=3, scale=4)
        self.assertEqual(res.shape, (4,))
        self.assertTrue(np.all(np.isfinite(res)))
        self.assertTrue(np.all(res > 0))

    def test_rcmse_matches_brute_force(self):
        """Agree with an independent brute-force RCMSE at several scales."""
        local_rng = np.random.default_rng(3)
        series = local_rng.standard_normal(150)
        tol = 0.4 * np.std(series)  # wide enough that every pooled count is positive
        for sample_length in (1, 2, 3):
            expected = _brute_rcmse(series, sample_length, 3, tol)
            np.testing.assert_allclose(ent.rcmse(series, sample_length, 3, tol), expected, rtol=1e-10)

    def test_rcmse_default_tolerance_uses_original_series(self):
        """Keep the default tolerance fixed at 0.1 * std of the original series."""
        local_rng = np.random.default_rng(11)
        series = local_rng.standard_normal(200)
        np.testing.assert_allclose(
            ent.rcmse(series, 2, 3),
            ent.rcmse(series, 2, 3, tolerance=0.1 * np.std(series)),
        )

    def test_rcmse_short_series_resilience(self):
        """Pool matches across shifts for a short time series."""
        rng_local = np.random.default_rng(42)
        ts = rng_local.random(40)
        tol = 0.2 * np.std(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmse_res = ent.composite_multiscale_entropy(ts, sample_length=3, scale=4, tolerance=tol)
            rcmse_res = ent.rcmse(ts, sample_length=3, scale=4, tolerance=tol)

        # Standard CMSE produces NaN or Inf on this short series at scale 3 and 4
        self.assertTrue(not np.isfinite(cmse_res[2]) or np.isnan(cmse_res[2]))
        self.assertTrue(np.isnan(cmse_res[3]))
        # RCMSE pools template matches across shifts and yields finite values
        self.assertTrue(np.isfinite(rcmse_res[2]))
        self.assertTrue(np.isfinite(rcmse_res[3]))
        np.testing.assert_allclose(rcmse_res[2:], np.array([1.60943791, 1.38629436]))

    def test_rcmse_zero_matches_warning(self):
        """Warn and return NaN when RCMSE finds no matches."""
        with self.assertWarns(RuntimeWarning):
            res = ent.rcmse([1.0, 10.0, 100.0, 1000.0, 10000.0], sample_length=2, scale=1, tolerance=0.01)
        self.assertEqual(res.shape, (1,))
        self.assertTrue(np.isnan(res[0]))

    def test_rcmse_scale_larger_than_series_gives_nan(self):
        """Return NaN with a warning once the coarse-grained series is too short."""
        with self.assertWarns(RuntimeWarning):
            res = ent.rcmse(RANDOM_TIME_SERIES[:20], sample_length=3, scale=15)
        self.assertEqual(res.shape, (15,))
        self.assertTrue(np.isnan(res[-1]))

    def test_rcmse_input_types(self):
        """Accept lists, integer arrays and NumPy integer parameters."""
        expected = ent.rcmse(np.array(TS_SAMPLE_ENTROPY, dtype=float), 2, 2)
        np.testing.assert_allclose(ent.rcmse(TS_SAMPLE_ENTROPY, 2, 2), expected)
        np.testing.assert_allclose(ent.rcmse(np.array(TS_SAMPLE_ENTROPY), 2, 2), expected)
        np.testing.assert_allclose(ent.rcmse(TS_SAMPLE_ENTROPY, np.int64(2), np.int32(2)), expected)

    def test_rcmse_alias(self):
        """Expose the descriptive alias for rcmse."""
        self.assertIs(ent.refined_composite_multiscale_entropy, ent.rcmse)

    def test_rcmse_invalid_values(self):
        """Reject out-of-range parameters, bad shapes and non-finite data with ValueError."""
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=2, scale=0)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=2, scale=-1)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=0, scale=2)
        self.assertRaises(ValueError, ent.rcmse, [1, 2], sample_length=3, scale=1)
        self.assertRaises(ValueError, ent.rcmse, np.zeros((3, 3)), sample_length=2, scale=1)
        self.assertRaises(ValueError, ent.rcmse, [1.0, np.nan, 2.0, 3.0], sample_length=2, scale=1)
        self.assertRaises(ValueError, ent.rcmse, [1.0, np.inf, 2.0, 3.0], sample_length=2, scale=1)
        for bad_tolerance in (0, -0.1, float("nan")):
            with self.subTest(tolerance=bad_tolerance):
                self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=2, scale=1, tolerance=bad_tolerance)

    def test_rcmse_invalid_types(self):
        """Reject non-integer scale and sample_length with TypeError."""
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, sample_length=2, scale=True)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, sample_length=2, scale=2.0)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, sample_length=2.5, scale=1)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, sample_length=False, scale=1)


# ---------------------------------------------------------------------------
# Fuzzy Entropy
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dispersion Entropy
# ---------------------------------------------------------------------------


class TestDispersionEntropy(unittest.TestCase):
    def test_dispersion_entropy_basic(self):
        """Return finite raw and normalized dispersion entropy."""
        de_raw = ent.dispersion_entropy(TS_SAMPLE_ENTROPY, classes=3, order=3, delay=1)
        self.assertTrue(np.isfinite(de_raw))
        self.assertGreater(de_raw, 0.0)

        de_norm = ent.dispersion_entropy(TS_SAMPLE_ENTROPY, classes=3, order=3, delay=1, normalize=True)
        self.assertTrue(np.isfinite(de_norm))
        self.assertGreater(de_norm, 0.0)
        self.assertLessEqual(de_norm, 1.0)
        np.testing.assert_allclose(de_norm, de_raw / (3 * math.log2(3)))

    def test_dispersion_entropy_constant_signal(self):
        """Return exactly zero dispersion entropy for constant signals."""
        constant_ts = np.ones(50)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="ncdf", normalize=False), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="ncdf", normalize=True), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="linear", normalize=False), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="linear", normalize=True), 0.0)
        self.assertEqual(ent.dispersion_entropy([4] * 20), 0.0)

    def test_dispersion_entropy_single_pattern_is_positive_zero(self):
        """Return +0.0 (not -0.0) when only one embedded pattern exists."""
        for normalize in (False, True):
            de = ent.dispersion_entropy([1, 5, 10], classes=3, order=3, delay=1, normalize=normalize)
            self.assertEqual(de, 0.0)
            self.assertEqual(math.copysign(1.0, de), 1.0)

    def test_dispersion_entropy_random_noise_near_one(self):
        """Approach maximum normalized entropy for random noise."""
        local_rng = np.random.default_rng(12345)
        noise = local_rng.standard_normal(20000)
        de_noise = ent.dispersion_entropy(noise, classes=3, order=2, delay=1, normalize=True)
        self.assertGreater(de_noise, 0.98)
        self.assertLessEqual(de_noise, 1.0)

    def test_dispersion_entropy_deterministic_and_sinusoidal(self):
        """Distinguish deterministic and sinusoidal signals from noise."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        val_norm = ent.dispersion_entropy(x, classes=3, order=2, delay=1, normalize=True)
        self.assertEqual(round(val_norm, 3), 0.667)

        t = np.linspace(0, 50, 2000)
        sin_sig = np.sin(t)
        de_sin = ent.dispersion_entropy(sin_sig, classes=3, order=2, delay=1, normalize=True)
        local_rng = np.random.default_rng(42)
        noise = local_rng.standard_normal(2000)
        de_noise = ent.dispersion_entropy(noise, classes=3, order=2, delay=1, normalize=True)
        self.assertLess(de_sin, de_noise)

    def test_dispersion_entropy_known_values(self):
        """Match hand-computed entropies for the series 1..10 with 3 classes and order 2."""
        x = list(range(1, 11))
        # ncdf classes [1,1,1,1,2,2,3,3,3,3] -> pattern counts 3,1,1,1,3 -> (4/3) * log2(3) bits
        np.testing.assert_allclose(ent.dispersion_entropy(x, 3, 2, 1, mapping="ncdf"), 4.0 / 3.0 * math.log2(3))
        # linear classes [1,1,1,2,2,2,3,3,3,3] -> pattern counts 2,1,2,1,3
        np.testing.assert_allclose(ent.dispersion_entropy(x, 3, 2, 1, mapping="linear"), _entropy_bits([2, 1, 2, 1, 3]))

    def test_dispersion_entropy_mappings(self):
        """Support case-insensitive NCDF and linear mappings."""
        signal = np.cos(np.linspace(start=0, stop=30, num=200))
        de_ncdf = ent.dispersion_entropy(signal, classes=3, order=2, delay=1, mapping="ncdf", normalize=True)
        de_linear = ent.dispersion_entropy(signal, classes=3, order=2, delay=1, mapping="linear", normalize=True)
        de_ncdf_upper = ent.dispersion_entropy(signal, classes=3, order=2, delay=1, mapping="NCDF", normalize=True)
        de_linear_upper = ent.dispersion_entropy(signal, classes=3, order=2, delay=1, mapping="Linear", normalize=True)

        self.assertEqual(de_ncdf, de_ncdf_upper)
        self.assertEqual(de_linear, de_linear_upper)
        self.assertTrue(0.0 <= de_ncdf <= 1.0)
        self.assertTrue(0.0 <= de_linear <= 1.0)

    def test_dispersion_entropy_invariance(self):
        """Preserve dispersion entropy under affine transformations."""
        local_rng = np.random.default_rng(98765)
        x = local_rng.standard_normal(500)
        x_shifted = 3.5 * x + 10.0

        for m in ["ncdf", "linear"]:
            de_orig = ent.dispersion_entropy(x, classes=4, order=3, delay=2, mapping=m, normalize=True)
            de_shifted = ent.dispersion_entropy(x_shifted, classes=4, order=3, delay=2, mapping=m, normalize=True)
            self.assertAlmostEqual(de_orig, de_shifted, places=10)

    def test_dispersion_entropy_order_and_delay(self):
        """Keep normalized entropy bounded across orders and delays."""
        local_rng = np.random.default_rng(42)
        ts = local_rng.standard_normal(300)
        for order in [1, 2, 3, 4]:
            for delay in [1, 2, 3]:
                de = ent.dispersion_entropy(ts, classes=3, order=order, delay=delay, normalize=True)
                self.assertTrue(0.0 <= de <= 1.0)

    def test_dispersion_entropy_matches_manual_pattern_counts(self):
        """Agree with a plain Counter over embedded class patterns for several orders and delays."""
        local_rng = np.random.default_rng(77)
        x = local_rng.standard_normal(400)
        for mapping in ("ncdf", "linear"):
            z = ent._map_to_classes(x, 4, mapping)
            for order, delay in ((2, 1), (3, 2), (4, 3)):
                with self.subTest(mapping=mapping, order=order, delay=delay):
                    span = (order - 1) * delay
                    patterns = Counter(tuple(z[i + k * delay] for k in range(order)) for i in range(len(z) - span))
                    expected = _entropy_bits(list(patterns.values()))
                    got = ent.dispersion_entropy(x, classes=4, order=order, delay=delay, mapping=mapping)
                    np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_dispersion_entropy_order_one_is_shannon_of_classes(self):
        """Reduce to the Shannon entropy of the class labels when order is 1."""
        local_rng = np.random.default_rng(8)
        x = local_rng.standard_normal(300)
        z = ent._map_to_classes(x, 5, "linear")
        np.testing.assert_allclose(
            ent.dispersion_entropy(x, classes=5, order=1, mapping="linear"),
            ent.shannon_entropy(z),
        )

    def test_dispersion_entropy_large_classes_order(self):
        """Avoid integer overflow for large class and order values."""
        local_rng = np.random.default_rng(42)
        ts = local_rng.standard_normal(100)
        de = ent.dispersion_entropy(ts, classes=10, order=20, delay=1, normalize=True)
        self.assertTrue(np.isfinite(de))
        self.assertGreater(de, 0.0)
        self.assertLessEqual(de, 1.0)

    def test_dispersion_entropy_numpy_integer_parameters(self):
        """Give identical results for NumPy integer and Python integer parameters, even past 2**63."""
        local_rng = np.random.default_rng(42)
        ts = local_rng.standard_normal(100)
        for classes, order in ((3, 3), (10, 25)):  # 10 ** 25 does not fit in int64
            with self.subTest(classes=classes, order=order):
                expected = ent.dispersion_entropy(ts, classes=classes, order=order, delay=1, normalize=True)
                got = ent.dispersion_entropy(
                    ts, classes=np.int64(classes), order=np.int32(order), delay=np.int64(1), normalize=True
                )
                self.assertEqual(got, expected)
                self.assertTrue(0.0 < got <= 1.0)

    def test_dispersion_entropy_input_types(self):
        """Return equivalent dispersion entropy for supported input types."""
        list_data = [1, 3, 2, 5, 4, 7, 6, 8, 2, 4]
        np_float = np.array(list_data, dtype=float)
        np_int = np.array(list_data, dtype=int)
        res_list = ent.dispersion_entropy(list_data, classes=3, order=2, delay=1)
        res_float = ent.dispersion_entropy(np_float, classes=3, order=2, delay=1)
        res_int = ent.dispersion_entropy(np_int, classes=3, order=2, delay=1)
        self.assertAlmostEqual(res_list, res_float)
        self.assertAlmostEqual(res_list, res_int)

    def test_dispersion_entropy_accepts_numpy_bool_normalize(self):
        """Accept np.bool_ for normalize."""
        x = list(range(1, 11))
        self.assertEqual(
            ent.dispersion_entropy(x, 3, 2, 1, normalize=np.True_),
            ent.dispersion_entropy(x, 3, 2, 1, normalize=True),
        )

    def test_map_to_classes_known_values(self):
        """Map 1..10 to the expected classes under both mappings."""
        x = np.arange(1, 11, dtype=float)
        np.testing.assert_array_equal(ent._map_to_classes(x, 3, "ncdf"), [1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
        np.testing.assert_array_equal(ent._map_to_classes(x, 3, "linear"), [1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def test_map_to_classes_ties_go_to_higher_class(self):
        """Send values exactly on a class boundary to the higher class (round half up)."""
        # min-max scaling puts these values at y = 0, 1/3, 2/3, 1, so classes * y is an exact integer
        x = np.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_array_equal(ent._map_to_classes(x, 3, "linear"), [1, 2, 3, 3])
        # 4 classes: c*y = 0, 1, 2, 3, 4 for x = 0..4 -> classes 1, 2, 3, 4, 4
        np.testing.assert_array_equal(ent._map_to_classes(np.arange(5.0), 4, "linear"), [1, 2, 3, 4, 4])
        # 2 classes: the midpoint belongs to the upper class
        np.testing.assert_array_equal(ent._map_to_classes(np.array([0.0, 1.0, 2.0]), 2, "linear"), [1, 2, 2])

    def test_map_to_classes_ncdf_ties_go_to_higher_class(self):
        """Send a value equal to the series mean (z = 0, Phi = 0.5) to the higher class."""
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(ent._map_to_classes(x, 2, "ncdf"), [1, 2, 2])
        np.testing.assert_array_equal(ent._map_to_classes(x, 4, "ncdf"), [1, 3, 4])

    def test_map_to_classes_ncdf_matches_erf_reference(self):
        """Match floor(c * Phi(z)) + 1 computed directly with math.erf."""
        local_rng = np.random.default_rng(1)
        x = local_rng.standard_normal(500)
        z = (x - x.mean()) / x.std()
        phi = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
        for classes in (2, 3, 5, 8):
            with self.subTest(classes=classes):
                expected = np.minimum(np.floor(classes * phi).astype(int) + 1, classes)
                np.testing.assert_array_equal(ent._map_to_classes(x, classes, "ncdf"), expected)

    def test_map_to_classes_range_and_degenerate_input(self):
        """Use labels 1..classes and return None for degenerate input."""
        local_rng = np.random.default_rng(2)
        x = local_rng.standard_normal(1000)
        for mapping in ("ncdf", "linear"):
            z = ent._map_to_classes(x, 6, mapping)
            self.assertEqual(z.dtype, np.int64)
            self.assertEqual(z.min(), 1)
            self.assertEqual(z.max(), 6)
            self.assertIsNone(ent._map_to_classes(np.ones(10), 6, mapping))

    def test_dispersion_entropy_invalid_values(self):
        """Reject out-of-range parameters, bad shapes and non-finite data with ValueError."""
        self.assertRaises(ValueError, ent.dispersion_entropy, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent.dispersion_entropy, [1, 2], classes=3, order=3, delay=1)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, classes=1)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, order=0)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, delay=0)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, mapping="invalid")
        self.assertRaises(ValueError, ent.dispersion_entropy, [1.0, np.nan, 3.0, 4.0])
        self.assertRaises(ValueError, ent.dispersion_entropy, [1.0, np.inf, 3.0, 4.0])
        self.assertRaises(ValueError, ent.dispersion_entropy, [1.0, -np.inf, 3.0, 4.0], mapping="linear")

    def test_dispersion_entropy_invalid_types(self):
        """Reject wrongly typed parameters with TypeError."""
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, classes=True)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, classes=2.5)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, order=False)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, order=1.5)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, delay=True)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, delay=1.0)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, mapping=None)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, mapping=1)
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, normalize="yes")
        self.assertRaises(TypeError, ent.dispersion_entropy, TIME_SERIES, normalize=1)


if __name__ == "__main__":
    unittest.main()
