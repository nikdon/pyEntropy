import math
import unittest
from collections import Counter
from statistics import NormalDist

import numpy as np

import pyentrp as pe

X = np.random.default_rng(9).standard_normal(500)


def _classes(x, c):
    """Map with the normal CDF of the series' mean and std, boundary values to the higher class."""
    nd = NormalDist(float(np.mean(x)), float(np.std(x)))
    return [min(math.floor(c * nd.cdf(v)) + 1, c) for v in x]


def _patterns(x, c, m, delay=1, fluctuation=False):
    z = _classes(x, c)
    if fluctuation:
        z = [z[i + delay] - z[i] for i in range(len(z) - delay)]
        m -= 1
    n = len(z) - (m - 1) * delay
    counts = Counter(tuple(z[i : i + m * delay : delay]) for i in range(n))
    return np.array(list(counts.values())) / n


class TestFluctuationDispersionEntropy(unittest.TestCase):
    def test_matches_class_change_patterns(self):
        """Count patterns of class changes z[t + d] - z[t] (Azami & Escudero 2018)."""
        for c, m, delay in ((3, 2, 1), (6, 3, 1), (4, 3, 2)):
            p = _patterns(X, c, m, delay, fluctuation=True)
            expected = float(-np.sum(p * np.log2(p)))
            with self.subTest(c=c, m=m, delay=delay):
                self.assertAlmostEqual(pe.dispersion_entropy(X, c, m, delay, fluctuation=True), expected, places=12)
                self.assertAlmostEqual(
                    pe.dispersion_entropy(X, c, m, delay, normalize=True, fluctuation=True),
                    expected / ((m - 1) * math.log2(2 * c - 1)),
                    places=12,
                )

    def test_needs_order_two(self):
        """Reject order 1 (no class changes) and a non-boolean flag."""
        self.assertRaises(ValueError, pe.dispersion_entropy, X, 3, 1, fluctuation=True)
        self.assertRaises(TypeError, pe.dispersion_entropy, X, 3, 2, fluctuation=1)


class TestGeneralizedDispersionEntropy(unittest.TestCase):
    def test_renyi_and_tsallis(self):
        """Use Renyi and Tsallis entropies of the dispersion patterns, normalized over classes ** order."""
        c, m, q = 4, 2, 2.5
        p = _patterns(X, c, m)
        k = c**m
        renyi = math.log2(np.sum(p**q)) / (1 - q)
        tsallis = (1 - np.sum(p**q)) / (q - 1)
        self.assertAlmostEqual(pe.dispersion_entropy(X, c, m, entropy="renyi", q=q), renyi, places=12)
        self.assertAlmostEqual(
            pe.dispersion_entropy(X, c, m, normalize=True, entropy="tsallis", q=q),
            tsallis / ((1 - k ** (1 - q)) / (q - 1)),
            places=12,
        )
        self.assertRaises(ValueError, pe.dispersion_entropy, X, entropy="renyi", q=1.0)


class TestReverseDispersionEntropy(unittest.TestCase):
    def test_sums_over_all_patterns(self):
        """Include unobserved patterns: sum over all K of (p - 1/K) ** 2 (Li et al. 2019)."""
        for c, m, fluctuation in ((3, 2, False), (6, 3, False), (4, 3, True)):
            observed = _patterns(X, c, m, fluctuation=fluctuation)
            k = (2 * c - 1) ** (m - 1) if fluctuation else c**m
            full = np.concatenate([observed, np.zeros(k - len(observed))])
            expected = float(np.sum((full - 1 / k) ** 2))
            with self.subTest(c=c, m=m, fluctuation=fluctuation):
                self.assertAlmostEqual(pe.reverse_dispersion_entropy(X, c, m, fluctuation=fluctuation), expected)
                self.assertAlmostEqual(
                    pe.reverse_dispersion_entropy(X, c, m, normalize=True, fluctuation=fluctuation),
                    expected / (1 - 1 / k),
                )

    def test_extremes(self):
        """Give 1 (normalized) for a single pattern and stay near 0 for long white noise."""
        self.assertAlmostEqual(pe.reverse_dispersion_entropy(np.ones(50), 3, 2, normalize=True), 1.0)
        noise = np.random.default_rng(3).standard_normal(100_000)
        self.assertLess(pe.reverse_dispersion_entropy(noise, 3, 2), 0.001)  # ncdf makes noise uniform

    def test_in_multiscale(self):
        """Work with every multiscale method; scale 1 is the measure itself."""
        expected = pe.reverse_dispersion_entropy(X, 3, 2)
        for method in (pe.multiscale, pe.composite_multiscale, pe.refined_composite_multiscale):
            self.assertEqual(method(X, pe.reverse_dispersion_entropy, 1, classes=3, order=2)[0], expected)


class TestHugePatternSpaces(unittest.TestCase):
    def test_fluctuation_rows_path(self):
        """Count fluctuation patterns exactly when (2c - 1) ** (m - 1) exceeds int64."""
        c, m = 40, 13  # 79 ** 12 > 2 ** 63
        p = _patterns(X, c, m, fluctuation=True)
        self.assertAlmostEqual(
            pe.dispersion_entropy(X, c, m, fluctuation=True), float(-np.sum(p * np.log2(p))), places=12
        )


if __name__ == "__main__":
    unittest.main()
