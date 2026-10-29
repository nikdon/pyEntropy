import math
import unittest
from collections import Counter

import numpy as np

from pyentrp import entropy as ent
from pyentrp._dispersion import map_to_classes
from tests._entropy_test_data import RANDOM_TIME_SERIES, TIME_SERIES, TS_SAMPLE_ENTROPY


def _entropy_bits(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


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
            z = map_to_classes(x, 4, mapping)
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
        z = map_to_classes(x, 5, "linear")
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
        np.testing.assert_array_equal(map_to_classes(x, 3, "ncdf"), [1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
        np.testing.assert_array_equal(map_to_classes(x, 3, "linear"), [1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def test_map_to_classes_ties_go_to_higher_class(self):
        """Send values exactly on a class boundary to the higher class (round half up)."""
        # min-max scaling puts these values at y = 0, 1/3, 2/3, 1, so classes * y is an exact integer
        x = np.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_array_equal(map_to_classes(x, 3, "linear"), [1, 2, 3, 3])
        # 4 classes: c*y = 0, 1, 2, 3, 4 for x = 0..4 -> classes 1, 2, 3, 4, 4
        np.testing.assert_array_equal(map_to_classes(np.arange(5.0), 4, "linear"), [1, 2, 3, 4, 4])
        # 2 classes: the midpoint belongs to the upper class
        np.testing.assert_array_equal(map_to_classes(np.array([0.0, 1.0, 2.0]), 2, "linear"), [1, 2, 2])

    def test_map_to_classes_ncdf_ties_go_to_higher_class(self):
        """Send a value equal to the series mean (z = 0, Phi = 0.5) to the higher class."""
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(map_to_classes(x, 2, "ncdf"), [1, 2, 2])
        np.testing.assert_array_equal(map_to_classes(x, 4, "ncdf"), [1, 3, 4])

    def test_map_to_classes_ncdf_matches_erf_reference(self):
        """Match floor(c * Phi(z)) + 1 computed directly with math.erf."""
        local_rng = np.random.default_rng(1)
        x = local_rng.standard_normal(500)
        z = (x - x.mean()) / x.std()
        phi = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
        for classes in (2, 3, 5, 8):
            with self.subTest(classes=classes):
                expected = np.minimum(np.floor(classes * phi).astype(int) + 1, classes)
                np.testing.assert_array_equal(map_to_classes(x, classes, "ncdf"), expected)

    def test_map_to_classes_range_and_degenerate_input(self):
        """Use labels 1..classes and return None for degenerate input."""
        local_rng = np.random.default_rng(2)
        x = local_rng.standard_normal(1000)
        for mapping in ("ncdf", "linear"):
            z = map_to_classes(x, 6, mapping)
            self.assertEqual(z.dtype, np.int64)
            self.assertEqual(z.min(), 1)
            self.assertEqual(z.max(), 6)
            self.assertIsNone(map_to_classes(np.ones(10), 6, mapping))

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

    def test_dispersion_entropy_mapping_is_case_insensitive(self):
        """Accept mapping names in any letter case."""
        x = RANDOM_TIME_SERIES[:200]
        for mapping in ("ncdf", "linear"):
            expected = ent.dispersion_entropy(x, mapping=mapping)
            self.assertEqual(ent.dispersion_entropy(x, mapping=mapping.upper()), expected)
            self.assertEqual(ent.dispersion_entropy(x, mapping=mapping.capitalize()), expected)


if __name__ == "__main__":
    unittest.main()


class TestDispersionConstantSeries(unittest.TestCase):
    def test_constant_float_series_is_zero(self):
        """Return +0.0 for constant series whose std is a tiny nonzero number, e.g. np.full(61, 0.1)."""
        for value in (0.1, 0.3, 1 / 3):
            for mapping in ("ncdf", "linear"):
                res = ent.dispersion_entropy(np.full(61, value), mapping=mapping)
                self.assertEqual(res, 0.0)
                self.assertEqual(math.copysign(1.0, res), 1.0)
