import math
import unittest
import warnings

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import RANDOM_TIME_SERIES, TIME_SERIES, TS_SAMPLE_ENTROPY
from tests._template_match_reference import _brute_match_counts


def _brute_rcmse(series, sample_length, scale, tolerance):
    """Pool counts over all shifted coarse-grained series at each scale for reference RCMSE."""
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


class TestEntropy(unittest.TestCase):
    def test_multiscale_entropy(self):
        """Calculate multiscale entropy across the requested scales."""
        multi_scale_entropy = ent.multiscale_entropy(RANDOM_TIME_SERIES, m=3, scale=4)
        np.testing.assert_allclose(
            multi_scale_entropy,
            np.array([3.178054, 3.178054, 2.890372, 3.401197]),
            rtol=1e-6,
        )

    def test_composite_multiscale_entropy(self):
        """Calculate composite multiscale entropy across scales."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.composite_multiscale_entropy(signal, m=2, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19283124, 0.94056984]))


class TestRCMSE(unittest.TestCase):
    def test_rcmse_scale_1_equivalence(self):
        """Match sample entropy when RCMSE uses a single scale."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        se = ent.sample_entropy(signal, m=2)[-1]
        res = ent.rcmse(signal, m=2, scale=1)
        self.assertEqual(res.shape, (1,))
        np.testing.assert_allclose(res[0], se)

        se_ts = ent.sample_entropy(TS_SAMPLE_ENTROPY, m=3)[-1]
        res_ts = ent.rcmse(TS_SAMPLE_ENTROPY, m=3, scale=1)
        np.testing.assert_allclose(res_ts[0], se_ts)

    def test_rcmse_sinusoidal(self):
        """Return expected RCMSE values for a sinusoidal signal."""
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.rcmse(signal, m=2, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19237189, 0.90078655]))

    def test_rcmse_random(self):
        """Return finite positive RCMSE values for random input."""
        res = ent.rcmse(RANDOM_TIME_SERIES, m=2, scale=4)
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
            np.testing.assert_allclose(
                ent.rcmse(series, m=sample_length - 1, scale=3, tolerance=tol), expected, rtol=1e-10
            )

    def test_rcmse_default_tolerance_uses_original_series(self):
        """Keep the default tolerance fixed at 0.1 * std of the original series."""
        local_rng = np.random.default_rng(11)
        series = local_rng.standard_normal(200)
        np.testing.assert_allclose(
            ent.rcmse(series, m=1, scale=3),
            ent.rcmse(series, m=1, scale=3, tolerance=0.1 * np.std(series)),
        )

    def test_rcmse_short_series_resilience(self):
        """Pool matches across shifts for a short time series."""
        rng_local = np.random.default_rng(42)
        ts = rng_local.random(40)
        tol = 0.2 * np.std(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmse_res = ent.composite_multiscale_entropy(ts, m=2, scale=4, tolerance=tol)
            rcmse_res = ent.rcmse(ts, m=2, scale=4, tolerance=tol)

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
            res = ent.rcmse([1.0, 10.0, 100.0, 1000.0, 10000.0], m=1, scale=1, tolerance=0.01)
        self.assertEqual(res.shape, (1,))
        self.assertTrue(np.isnan(res[0]))

    def test_rcmse_scale_larger_than_series_gives_nan(self):
        """Return NaN with a warning once the coarse-grained series is too short."""
        with self.assertWarns(RuntimeWarning):
            res = ent.rcmse(RANDOM_TIME_SERIES[:20], m=2, scale=15)
        self.assertEqual(res.shape, (15,))
        self.assertTrue(np.isnan(res[-1]))

    def test_rcmse_input_types(self):
        """Accept lists, integer arrays and NumPy integer parameters."""
        expected = ent.rcmse(np.array(TS_SAMPLE_ENTROPY, dtype=float), m=1, scale=2)
        np.testing.assert_allclose(ent.rcmse(TS_SAMPLE_ENTROPY, m=1, scale=2), expected)
        np.testing.assert_allclose(ent.rcmse(np.array(TS_SAMPLE_ENTROPY), m=1, scale=2), expected)
        np.testing.assert_allclose(ent.rcmse(TS_SAMPLE_ENTROPY, m=np.int64(1), scale=np.int32(2)), expected)

    def test_rcmse_alias(self):
        """Expose the descriptive alias for rcmse."""
        self.assertIs(ent.refined_composite_multiscale_entropy, ent.rcmse)

    def test_rcmse_invalid_values(self):
        """Reject out-of-range parameters, bad shapes and non-finite data with ValueError."""
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, m=1, scale=0)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, m=1, scale=-1)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, m=-1, scale=2)
        self.assertRaises(ValueError, ent.rcmse, [1, 2], m=2, scale=1)
        self.assertRaises(ValueError, ent.rcmse, np.zeros((3, 3)), m=1, scale=1)
        self.assertRaises(ValueError, ent.rcmse, [1.0, np.nan, 2.0, 3.0], m=1, scale=1)
        self.assertRaises(ValueError, ent.rcmse, [1.0, np.inf, 2.0, 3.0], m=1, scale=1)
        for bad_tolerance in (0, -0.1, float("nan")):
            with self.subTest(tolerance=bad_tolerance):
                self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, m=1, scale=1, tolerance=bad_tolerance)

    def test_rcmse_invalid_types(self):
        """Reject non-integer scale and sample_length with TypeError."""
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, m=1, scale=True)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, m=1, scale=2.0)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, m=1.5, scale=1)
        self.assertRaises(TypeError, ent.rcmse, TIME_SERIES, m=False, scale=1)


class TestMultiscaleValidation(unittest.TestCase):
    def test_multiscale_entropy_invalid_input(self):
        """Reject bad parameters and non-finite data in multiscale_entropy."""
        self.assertRaises(ValueError, ent.multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, scale=0)
        self.assertRaises(ValueError, ent.multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, tolerance=-1.0)
        self.assertRaises(ValueError, ent.multiscale_entropy, [*TS_SAMPLE_ENTROPY, np.nan], m=1, scale=2)
        self.assertRaises(TypeError, ent.multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, scale=2.0)
        self.assertRaises(TypeError, ent.multiscale_entropy, TS_SAMPLE_ENTROPY, m=1.0, scale=2)

    def test_composite_multiscale_entropy_invalid_input(self):
        """Reject bad parameters and non-finite data in composite_multiscale_entropy."""
        self.assertRaises(ValueError, ent.composite_multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, scale=0)
        self.assertRaises(ValueError, ent.composite_multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, scale=2, tolerance=-1.0)
        self.assertRaises(ValueError, ent.composite_multiscale_entropy, [*TS_SAMPLE_ENTROPY, np.inf], m=1, scale=2)
        self.assertRaises(TypeError, ent.composite_multiscale_entropy, TS_SAMPLE_ENTROPY, m=1, scale=2.0)


class TestConstantSeries(unittest.TestCase):
    """A constant series has std 0; the default tolerance must not turn that into NaN, inf or an error."""

    CONSTANT = np.full(60, 3.0)

    def _assert_positive_zeros(self, values):
        values = np.atleast_1d(values)
        np.testing.assert_array_equal(values, 0.0)
        self.assertTrue(np.all(np.copysign(1.0, values) == 1.0), values)

    def test_sample_entropy_default_matches_explicit_tolerance(self):
        """Give the same result as any positive tolerance, with a zero last entry."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            default = ent.sample_entropy(self.CONSTANT, m=2)
        for tol in (1e-9, 0.5, 100.0):
            np.testing.assert_array_equal(default, ent.sample_entropy(self.CONSTANT, m=2, tolerance=tol))
        self._assert_positive_zeros(default[1:])

    def test_multiscale_functions_return_zero(self):
        """Return +0.0 at every scale for MSE, CMSE and RCMSE, without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._assert_positive_zeros(ent.multiscale_entropy(self.CONSTANT, m=2, scale=3))
            self._assert_positive_zeros(ent.composite_multiscale_entropy(self.CONSTANT, m=2, scale=3))
            self._assert_positive_zeros(ent.rcmse(self.CONSTANT, m=2, scale=3))

    def test_float_constant_series(self):
        """Treat a constant series of non-representable floats the same way."""
        self._assert_positive_zeros(ent.rcmse(np.full(60, 0.1), m=1, scale=2))


class TestMultiscaleDefaults(unittest.TestCase):
    def test_default_maxscale_is_twenty(self):
        """Compute scales 1 to 20 by default, and treat None as 20."""
        x = RANDOM_TIME_SERIES[:600]
        default = ent.multiscale_entropy(x, m=1)
        self.assertEqual(default.shape, (20,))
        np.testing.assert_array_equal(default, ent.multiscale_entropy(x, m=1, scale=None))
        np.testing.assert_array_equal(default[:5], ent.multiscale_entropy(x, m=1, scale=5))
