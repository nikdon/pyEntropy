import unittest
import warnings

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


class TestEntropy(unittest.TestCase):
    def test_shannon_entropy_string(self):
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES_STRING), SHANNON_ENTROPY, rtol=1e-5)

    def test_shannon_entropy_numerical(self):
        np.testing.assert_allclose(ent.shannon_entropy(TIME_SERIES), SHANNON_ENTROPY, rtol=1e-5)

    def test_sample_entropy(self):
        ts = TS_SAMPLE_ENTROPY
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        np.testing.assert_allclose(sample_entropy, np.array([2.26881823, 2.11119024, 2.33537492, 1.79175947]))

    def test_multiscale_entropy(self):
        multi_scale_entropy = ent.multiscale_entropy(RANDOM_TIME_SERIES, 4, maxscale=4)
        np.testing.assert_allclose(
            multi_scale_entropy,
            np.array([3.178054, 3.178054, 2.890372, 3.401197]),
            rtol=1e-6,
        )

    def test_permutation_entropy(self):
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
        np.testing.assert_array_equal(
            np.round(ent.multiscale_permutation_entropy(TS_SAMPLE_ENTROPY, 3, 5, 2), 4),
            np.array([2.4699, 2.5649]),
        )

    def test_util_pattern_space(self):
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 0, 2)
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 10, 20)
        np.testing.assert_array_equal(
            ent.util_pattern_space(TIME_SERIES, 2, 3),
            np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]),
        )

    def test_composite_multiscale_entropy(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.composite_multiscale_entropy(signal, sample_length=3, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19283124, 0.94056984]))

    def test_count_template_matches(self):
        ts = [1, 2, 1, 2, 1]
        matches = ent._count_template_matches(ts, sample_length=2, tolerance=0.5)
        np.testing.assert_array_equal(matches, np.array([10.0, 2.0, 2.0]))

    def test_rcmse_scale_1_equivalence(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        se = ent.sample_entropy(signal, sample_length=3)[-1]
        rcmse = ent.rcmse(signal, sample_length=3, scale=1)
        self.assertEqual(rcmse.shape, (1,))
        np.testing.assert_allclose(rcmse[0], se)

        se_ts = ent.sample_entropy(TS_SAMPLE_ENTROPY, sample_length=4)[-1]
        rcmse_ts = ent.rcmse(TS_SAMPLE_ENTROPY, sample_length=4, scale=1)
        np.testing.assert_allclose(rcmse_ts[0], se_ts)

    def test_rcmse_sinusoidal(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.rcmse(signal, sample_length=3, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19237189, 0.90078655]))

    def test_rcmse_random(self):
        res = ent.rcmse(RANDOM_TIME_SERIES, sample_length=3, scale=4)
        self.assertEqual(res.shape, (4,))
        self.assertTrue(np.all(np.isfinite(res)))
        self.assertTrue(np.all(res > 0))

    def test_rcmse_short_series_resilience(self):
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
        with self.assertWarns(RuntimeWarning):
            res = ent.rcmse([1.0, 10.0, 100.0, 1000.0, 10000.0], sample_length=2, scale=1, tolerance=0.01)
        self.assertEqual(res.shape, (1,))
        self.assertTrue(np.isnan(res[0]))

    def test_rcmse_invalid_inputs(self):
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=2, scale=0)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=2, scale=-1)
        self.assertRaises(ValueError, ent.rcmse, TIME_SERIES, sample_length=0, scale=2)
        self.assertRaises(ValueError, ent.rcmse, [1, 2], sample_length=3, scale=1)
        self.assertRaises(ValueError, ent.rcmse, np.zeros((3, 3)), sample_length=2, scale=1)

    def test_fuzzy_entropy_basic(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe = ent.fuzzy_entropy(signal, sample_length=2, tolerance=0.2 * np.std(signal), n=2)
        self.assertIsInstance(fe, float)
        self.assertTrue(np.isfinite(fe))
        self.assertGreater(fe, 0.0)

        fe_default = ent.fuzzy_entropy(signal, sample_length=2)
        np.testing.assert_allclose(fe, fe_default)

    def test_fuzzy_entropy_deterministic_and_sinusoidal(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe = ent.fuzzy_entropy(signal, sample_length=2)
        np.testing.assert_allclose(fe, 0.60045352, rtol=1e-5)

        fe_ts = ent.fuzzy_entropy(TS_SAMPLE_ENTROPY, sample_length=2)
        self.assertTrue(np.isfinite(fe_ts))
        self.assertGreater(fe_ts, 0.0)

    def test_fuzzy_entropy_constant_signal(self):
        constant_ts = np.ones(50)
        self.assertEqual(ent.fuzzy_entropy(constant_ts), 0.0)
        self.assertEqual(ent.fuzzy_entropy(constant_ts, tolerance=0.2), 0.0)
        self.assertEqual(ent.fuzzy_entropy([5] * 20), 0.0)

    def test_fuzzy_entropy_smoothness_vs_tolerance(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        tolerances = [0.1, 0.2, 0.3, 0.4, 0.5]
        fe_values = [ent.fuzzy_entropy(signal, sample_length=2, tolerance=tol) for tol in tolerances]
        for i in range(len(fe_values) - 1):
            self.assertGreater(fe_values[i], fe_values[i + 1])

    def test_fuzzy_entropy_varying_n(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        fe_n1 = ent.fuzzy_entropy(signal, sample_length=2, n=1)
        fe_n2 = ent.fuzzy_entropy(signal, sample_length=2, n=2)
        fe_n4 = ent.fuzzy_entropy(signal, sample_length=2, n=4)
        for val in (fe_n1, fe_n2, fe_n4):
            self.assertTrue(np.isfinite(val))
            self.assertGreater(val, 0.0)

    def test_fuzzy_entropy_chunking_large_series(self):
        rng_local = np.random.default_rng(999)
        long_ts = rng_local.standard_normal(1200)
        fe = ent.fuzzy_entropy(long_ts, sample_length=2)
        self.assertTrue(np.isfinite(fe))
        self.assertGreater(fe, 1.5)

    def test_fuzzy_entropy_vanished_similarity_warning(self):
        extreme_ts = [1.0, 1000.0, 100000.0, 10000000.0, 1000000000.0]
        with self.assertWarns(RuntimeWarning):
            fe = ent.fuzzy_entropy(extreme_ts, sample_length=1, tolerance=1e-12, n=2)
        self.assertTrue(np.isnan(fe))

    def test_fuzzy_entropy_input_types(self):
        list_data = [1, 3, 2, 5, 4, 7, 6, 8, 2, 4]
        np_float = np.array(list_data, dtype=float)
        np_int = np.array(list_data, dtype=int)
        res_list = ent.fuzzy_entropy(list_data, sample_length=2)
        res_float = ent.fuzzy_entropy(np_float, sample_length=2)
        res_int = ent.fuzzy_entropy(np_int, sample_length=2)
        self.assertAlmostEqual(res_list, res_float)
        self.assertAlmostEqual(res_list, res_int)

    def test_fuzzy_entropy_invalid_inputs(self):
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, sample_length=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, sample_length=-1)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, sample_length=2.5)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, n=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, n=-2)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, tolerance=0)
        self.assertRaises(ValueError, ent.fuzzy_entropy, TIME_SERIES, tolerance=-0.1)
        self.assertRaises(ValueError, ent.fuzzy_entropy, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent.fuzzy_entropy, [1, 2, 3], sample_length=2)

    def test_dispersion_entropy_basic(self):
        de_raw = ent.dispersion_entropy(TS_SAMPLE_ENTROPY, classes=3, order=3, delay=1)
        self.assertTrue(np.isfinite(de_raw))
        self.assertGreater(de_raw, 0.0)

        de_norm = ent.dispersion_entropy(TS_SAMPLE_ENTROPY, classes=3, order=3, delay=1, normalize=True)
        self.assertTrue(np.isfinite(de_norm))
        self.assertGreater(de_norm, 0.0)
        self.assertLessEqual(de_norm, 1.0)

    def test_dispersion_entropy_constant_signal(self):
        constant_ts = np.ones(50)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="ncdf", normalize=False), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="ncdf", normalize=True), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="linear", normalize=False), 0.0)
        self.assertEqual(ent.dispersion_entropy(constant_ts, mapping="linear", normalize=True), 0.0)
        self.assertEqual(ent.dispersion_entropy([4] * 20), 0.0)
        # Minimal series yielding exactly one embedded pattern
        self.assertEqual(ent.dispersion_entropy([1, 5, 10], classes=3, order=3, delay=1), 0.0)

    def test_dispersion_entropy_random_noise_near_one(self):
        local_rng = np.random.default_rng(12345)
        noise = local_rng.standard_normal(20000)
        de_noise = ent.dispersion_entropy(noise, classes=3, order=2, delay=1, normalize=True)
        self.assertGreater(de_noise, 0.98)
        self.assertLessEqual(de_noise, 1.0)

    def test_dispersion_entropy_deterministic_and_sinusoidal(self):
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

    def test_dispersion_entropy_mappings(self):
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
        local_rng = np.random.default_rng(98765)
        x = local_rng.standard_normal(500)
        x_shifted = 3.5 * x + 10.0

        for m in ["ncdf", "linear"]:
            de_orig = ent.dispersion_entropy(x, classes=4, order=3, delay=2, mapping=m, normalize=True)
            de_shifted = ent.dispersion_entropy(x_shifted, classes=4, order=3, delay=2, mapping=m, normalize=True)
            self.assertAlmostEqual(de_orig, de_shifted, places=10)

    def test_dispersion_entropy_order_and_delay(self):
        local_rng = np.random.default_rng(42)
        ts = local_rng.standard_normal(300)
        for order in [1, 2, 3, 4]:
            for delay in [1, 2, 3]:
                de = ent.dispersion_entropy(ts, classes=3, order=order, delay=delay, normalize=True)
                self.assertTrue(0.0 <= de <= 1.0)

    def test_dispersion_entropy_large_classes_order(self):
        local_rng = np.random.default_rng(42)
        ts = local_rng.standard_normal(100)
        de = ent.dispersion_entropy(ts, classes=10, order=20, delay=1, normalize=True)
        self.assertTrue(np.isfinite(de))
        self.assertGreater(de, 0.0)
        self.assertLessEqual(de, 1.0)

    def test_dispersion_entropy_input_types(self):
        list_data = [1, 3, 2, 5, 4, 7, 6, 8, 2, 4]
        np_float = np.array(list_data, dtype=float)
        np_int = np.array(list_data, dtype=int)
        res_list = ent.dispersion_entropy(list_data, classes=3, order=2, delay=1)
        res_float = ent.dispersion_entropy(np_float, classes=3, order=2, delay=1)
        res_int = ent.dispersion_entropy(np_int, classes=3, order=2, delay=1)
        self.assertAlmostEqual(res_list, res_float)
        self.assertAlmostEqual(res_list, res_int)

    def test_dispersion_entropy_invalid_inputs(self):
        self.assertRaises(ValueError, ent.dispersion_entropy, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent.dispersion_entropy, [1, 2], classes=3, order=3, delay=1)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, classes=1)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, classes=True)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, classes=2.5)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, order=0)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, order=False)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, order=1.5)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, delay=0)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, delay=True)
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, mapping="invalid")
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, normalize="yes")
        self.assertRaises(ValueError, ent.dispersion_entropy, TIME_SERIES, normalize=1)


if __name__ == "__main__":
    unittest.main()
