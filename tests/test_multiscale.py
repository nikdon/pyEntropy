import itertools
import math
import unittest
import warnings
from collections import defaultdict
from statistics import NormalDist

import numpy as np

import pyentrp as pe
from pyentrp import entropy as ent
from pyentrp._multiscale import MEASURES
from tests._entropy_test_data import RANDOM_TIME_SERIES

X = np.random.default_rng(3).standard_normal(240)
R = 0.2 * float(np.std(X))
METHODS = (pe.multiscale, pe.composite_multiscale, pe.refined_composite_multiscale)

# every registered measure with parameters that suit X
MEASURE_CASES = (
    (pe.sample_entropy, {"m": 2, "tolerance": R}),
    (pe.fuzzy_entropy, {"m": 2, "tolerance": R}),
    (pe.permutation_entropy, {"order": 3, "delay": 1}),
    (pe.weighted_permutation_entropy, {"order": 3, "delay": 2, "normalize": True}),
    (pe.dispersion_entropy, {"classes": 4, "order": 2}),
    (pe.dispersion_entropy, {"classes": 3, "order": 3, "mapping": "linear"}),
)


def _single(measure, x, **params):
    """Return the measure as a single number (sample_entropy contributes its last entry)."""
    value = measure(x, **params)
    return float(np.asarray(value)[-1]) if measure is pe.sample_entropy else value


def _shifts(x, k):
    return [pe.util_granulate_time_series(x[j:], k) for j in range(k)]


def _entropy_bits(p):
    p = np.asarray([v for v in p if v > 0])
    return float(-np.sum(p * np.log2(p)))


def _averaged_pattern_entropy(series_list, patterns_of):
    """Average the pattern distributions of several series, then take the entropy (RCMPE, RCMDE)."""
    avg = defaultdict(float)
    for s in series_list:
        patterns = patterns_of(s)
        for pat in patterns:
            avg[pat] += 1 / len(patterns) / len(series_list)
    return _entropy_bits(avg.values())


def _ordinal_patterns(s, order, delay):
    n = len(s) - (order - 1) * delay
    return [tuple(np.argsort(s[i : i + order * delay : delay], kind="stable")) for i in range(n)]


class TestRegistry(unittest.TestCase):
    def test_measures_are_registered_under_public_names(self):
        """Register the objects users import from pyentrp and pyentrp.entropy."""
        for name in (
            "sample_entropy",
            "fuzzy_entropy",
            "permutation_entropy",
            "weighted_permutation_entropy",
            "dispersion_entropy",
        ):
            self.assertIn(getattr(pe, name), MEASURES)
            self.assertIn(getattr(ent, name), MEASURES)


class TestScaleOne(unittest.TestCase):
    def test_all_methods_equal_the_measure_at_scale_one(self):
        """Give exactly the single-scale value at scale 1, for every measure and method."""
        for measure, params in MEASURE_CASES:
            expected = _single(measure, X, **params)
            for method in METHODS:
                with self.subTest(measure=measure.__name__, method=method.__name__, params=params):
                    self.assertEqual(method(X, measure, 1, **params)[0], expected)


class TestLegacyFunctions(unittest.TestCase):
    def test_named_functions_are_the_shared_methods(self):
        """Give identical results for MSE, CMSE, RCMSE and MPE and the shared methods."""
        x = RANDOM_TIME_SERIES[:300]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # zero match counts at the largest scales
            pairs = self._pairs(x)
        for legacy, shared in pairs:
            np.testing.assert_array_equal(legacy, shared)

    @staticmethod
    def _pairs(x):
        return (
            (pe.multiscale_entropy(x, m=2, scale=6), pe.multiscale(x, pe.sample_entropy, 6, m=2)),
            (
                pe.composite_multiscale_entropy(x, m=2, scale=6),
                pe.composite_multiscale(x, pe.sample_entropy, 6, m=2),
            ),
            (pe.rcmse(x, m=2, scale=6), pe.refined_composite_multiscale(x, pe.sample_entropy, 6, m=2)),
            (
                pe.multiscale_permutation_entropy(x, 4, 1, 6),
                pe.multiscale(x, pe.permutation_entropy, 6, order=4, delay=1),
            ),
        )


class TestAgainstReferences(unittest.TestCase):
    def test_multiscale_applies_the_measure_to_each_coarse_series(self):
        """Apply the measure to the coarse-grained series at each scale."""
        for measure, params in MEASURE_CASES[:4]:
            with self.subTest(measure=measure.__name__):
                expected = [_single(measure, pe.util_granulate_time_series(X, k), **params) for k in (1, 2, 3)]
                np.testing.assert_allclose(pe.multiscale(X, measure, 3, **params), expected, rtol=1e-12)

    def test_composite_averages_all_shifts(self):
        """Average the measure over the k shifted coarse-grained series at scale k."""
        for measure, params in MEASURE_CASES[:4]:
            with self.subTest(measure=measure.__name__):
                expected = [np.mean([_single(measure, s, **params) for s in _shifts(X, k)]) for k in (1, 2, 3, 4)]
                np.testing.assert_allclose(pe.composite_multiscale(X, measure, 4, **params), expected, rtol=1e-12)

    def test_refined_permutation_averages_pattern_probabilities(self):
        """Average the ordinal pattern distributions of the shifts, then take the entropy (RCMPE)."""
        x = X[:101]  # odd length: the shifts have different numbers of windows
        got = pe.refined_composite_multiscale(x, pe.permutation_entropy, 4, order=3, delay=2)
        expected = [_averaged_pattern_entropy(_shifts(x, k), lambda s: _ordinal_patterns(s, 3, 2)) for k in range(1, 5)]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_refined_dispersion_uses_the_original_mapping(self):
        """Map every shift with the mean and std of the original series, average the distributions (RCMDE)."""
        x = X[:101]
        mu, sigma, classes = np.mean(x), np.std(x), 4
        normal = NormalDist(mu, sigma)

        def patterns(s):
            z = [min(math.floor(classes * normal.cdf(v)) + 1, classes) for v in s]
            return [tuple(z[i : i + 2]) for i in range(len(z) - 1)]

        got = pe.refined_composite_multiscale(x, pe.dispersion_entropy, 4, classes=classes, order=2)
        expected = [_averaged_pattern_entropy(_shifts(x, k), patterns) for k in range(1, 5)]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_multiscale_dispersion_keeps_the_original_mapping(self):
        """Keep the original series' mapping at every scale, unlike dispersion_entropy of the coarse series."""
        coarse = pe.util_granulate_time_series(X, 3)
        own = pe.dispersion_entropy(coarse, classes=4, order=2)
        self.assertNotEqual(pe.multiscale(X, pe.dispersion_entropy, 3, classes=4, order=2)[2], own)

    def test_refined_fuzzy_sums_similarities(self):
        """Use log(sum phi_m / sum phi_m+1) over the shifts (RCMFE)."""
        x = X[:90]

        def phi(s, length):
            count = len(s) - 2
            vectors = [s[i : i + length] - np.mean(s[i : i + length]) for i in range(count)]
            pairs = list(itertools.combinations(vectors, 2))
            return sum(math.exp(-((np.max(np.abs(a - b)) / R) ** 2)) for a, b in pairs) / len(pairs)

        expected = [
            math.log(sum(phi(s, 2) for s in _shifts(x, k)) / sum(phi(s, 3) for s in _shifts(x, k))) for k in (1, 2, 3)
        ]
        got = pe.refined_composite_multiscale(x, pe.fuzzy_entropy, 3, m=2, tolerance=R)
        np.testing.assert_allclose(got, expected, rtol=1e-10)


class TestEdgeCases(unittest.TestCase):
    def test_custom_callable(self):
        """Accept any function of a series for multiscale and composite, but not for refined."""

        def spread(s, factor=1.0):
            return factor * float(np.ptp(s))

        np.testing.assert_allclose(
            pe.multiscale(X, spread, 3, factor=2.0),
            [2.0 * np.ptp(pe.util_granulate_time_series(X, k)) for k in (1, 2, 3)],
        )
        self.assertEqual(pe.composite_multiscale(X, spread, 2).shape, (2,))
        self.assertRaises(TypeError, pe.refined_composite_multiscale, X, spread, 2)
        self.assertRaises(TypeError, pe.multiscale, X, "sample_entropy", 2)

    def test_too_short_scales_give_nan_with_one_warning_at_the_call_site(self):
        """Return NaN from the first scale that is too short, with a single warning."""
        x = X[:30]
        for method in METHODS:
            with self.subTest(method=method.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = method(x, pe.permutation_entropy, 12, order=4)
                too_short = [w for w in caught if "shorter than" in str(w.message)]
                self.assertEqual(len(too_short), 1)
                self.assertEqual(too_short[0].filename, __file__)
                self.assertTrue(np.all(np.isnan(res[10:])))
                self.assertTrue(np.all(np.isfinite(res[:6])))

    def test_parameters_are_checked_once_before_any_scale(self):
        """Reject bad parameters and series too short for the measure before computing."""
        for method in METHODS:
            with self.subTest(method=method.__name__):
                self.assertRaises(ValueError, method, X, pe.permutation_entropy, 3, order=0)
                self.assertRaises(ValueError, method, X, pe.sample_entropy, 3, m=2, tolerance=-1.0)
                self.assertRaises(TypeError, method, X, pe.sample_entropy, 3)
                self.assertRaises(ValueError, method, X, pe.permutation_entropy, 0)
                self.assertRaises(TypeError, method, X, pe.permutation_entropy, 2.0)
                self.assertRaises(ValueError, method, [1.0, 2.0], pe.permutation_entropy, 3, order=3)
                self.assertRaises(ValueError, method, [1.0, np.nan, 2.0], pe.permutation_entropy, 2)

    def test_series_too_short_for_sample_entropy(self):
        """Give NaN with a warning for multiscale and composite, as MSE always did; raise for refined, like RCMSE."""
        for method in (pe.multiscale, pe.composite_multiscale):
            with self.subTest(method=method.__name__):
                with self.assertWarnsRegex(RuntimeWarning, "shorter than 4 samples from scale 1"):
                    res = method([1.0, 2.0, 3.0], pe.sample_entropy, 2, m=2)
                self.assertTrue(np.all(np.isnan(res)))
        self.assertRaises(ValueError, pe.refined_composite_multiscale, [1.0, 2.0, 3.0], pe.sample_entropy, 2, m=2)

    def test_integer_input_stays_exact(self):
        """Compare large integers exactly at scale 1."""
        big = [2**53, 2**53 + 1, 2**53, 2**53 + 1, 2**53 + 3, 2**53 + 2]
        for method in METHODS:
            self.assertEqual(method(big, pe.permutation_entropy, 1, order=2)[0], pe.permutation_entropy(big, 2))

    def test_linear_mapping_of_block_means_below_the_minimum(self):
        """Keep class labels in range when a block mean rounds to just below the original minimum."""
        x = np.array([0.7] * 3 + [1.0, 2.0, 0.7, 0.7, 0.7, 5.0])
        self.assertLess(np.mean(x[:3]), x.min())
        for method in METHODS:
            res = method(x, pe.dispersion_entropy, 3, classes=3, order=1, mapping="linear")
            self.assertTrue(np.all(np.isfinite(res)))

    def test_constant_weighted_shifts_are_skipped(self):
        """Give 0.0, like the single-scale measure, when no window has any weight."""
        res = pe.refined_composite_multiscale(np.ones(50), pe.weighted_permutation_entropy, 3, order=3)
        np.testing.assert_array_equal(res, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
