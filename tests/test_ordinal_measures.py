import itertools
import math
import unittest
from collections import Counter

import numpy as np

import pyentrp as pe

RNG = np.random.default_rng(5)
X = RNG.standard_normal(400)
TIES = RNG.integers(0, 3, 300)


def _patterns(x, order, delay):
    n = len(x) - (order - 1) * delay
    return [tuple(np.argsort(x[i : i + order * delay : delay], kind="stable")) for i in range(n)]


def _distribution(x, order, delay, weights=None):
    pats = _patterns(x, order, delay)
    weights = np.ones(len(pats)) if weights is None else weights
    mass = Counter()
    for pat, w in zip(pats, weights, strict=True):
        mass[pat] += w
    total = sum(mass.values())
    return {pat: w / total for pat, w in mass.items() if w > 0}


def _full_probabilities(x, order, delay):
    dist = _distribution(x, order, delay)
    return np.array([dist.get(perm, 0.0) for perm in itertools.permutations(range(order))])


class TestGeneralizedPermutationEntropy(unittest.TestCase):
    def test_renyi_and_tsallis_match_their_definitions(self):
        """Compute Renyi and Tsallis entropies of the ordinal distribution, plain and normalized."""
        for order, q in ((3, 0.5), (3, 2.0), (4, 3.0)):
            p = np.array(list(_distribution(X, order, 1).values()))
            n_states = math.factorial(order)
            renyi = math.log2(np.sum(p**q)) / (1 - q)
            tsallis = (1 - np.sum(p**q)) / (q - 1)
            with self.subTest(order=order, q=q):
                self.assertAlmostEqual(pe.permutation_entropy(X, order, entropy="renyi", q=q), renyi, places=12)
                self.assertAlmostEqual(pe.permutation_entropy(X, order, entropy="tsallis", q=q), tsallis, places=12)
                self.assertAlmostEqual(
                    pe.permutation_entropy(X, order, normalize=True, entropy="renyi", q=q),
                    renyi / math.log2(n_states),
                    places=12,
                )
                self.assertAlmostEqual(
                    pe.permutation_entropy(X, order, normalize=True, entropy="tsallis", q=q),
                    tsallis / ((1 - n_states ** (1 - q)) / (q - 1)),
                    places=12,
                )

    def test_shannon_is_the_default(self):
        """Keep the Shannon permutation entropy unchanged."""
        self.assertEqual(pe.permutation_entropy(X, 4), pe.permutation_entropy(X, 4, entropy="shannon", q=7.0))

    def test_rejects_bad_kind_and_q(self):
        """Reject unknown kinds and q <= 0 or q == 1."""
        self.assertRaises(ValueError, pe.permutation_entropy, X, 3, entropy="hartley")
        self.assertRaises(TypeError, pe.permutation_entropy, X, 3, entropy=2)
        for q in (0.0, -1.0, 1.0, float("nan"), float("inf")):
            self.assertRaises(ValueError, pe.weighted_permutation_entropy, X, 3, entropy="renyi", q=q)


class TestAmplitudeAwarePermutationEntropy(unittest.TestCase):
    def _reference(self, x, order, delay, amplitude):
        x = np.asarray(x, dtype=float)
        n = len(x) - (order - 1) * delay
        weights = []
        for i in range(n):
            w = x[i : i + order * delay : delay]
            step = np.abs(np.diff(w)).sum() / (order - 1) if order > 1 else 0.0
            weights.append(amplitude * np.abs(w).mean() + (1 - amplitude) * step)
        p = np.array(list(_distribution(x, order, delay, np.array(weights)).values()))
        return float(-np.sum(p * np.log2(p)))

    def test_matches_definition(self):
        """Weight each window by A * mean |x| + (1 - A) * mean |step| (Azami & Escudero 2016)."""
        for order, delay, amplitude in ((3, 1, 0.5), (4, 2, 0.0), (3, 1, 1.0), (5, 1, 0.3)):
            with self.subTest(order=order, delay=delay, amplitude=amplitude):
                self.assertAlmostEqual(
                    pe.amplitude_aware_permutation_entropy(X, order, delay, amplitude),
                    self._reference(X, order, delay, amplitude),
                    places=12,
                )

    def test_ties_and_zeros(self):
        """Rank ties by position and give 0.0 when every weight is zero."""
        self.assertAlmostEqual(
            pe.amplitude_aware_permutation_entropy(TIES, 3), self._reference(TIES, 3, 1, 0.5), places=12
        )
        self.assertEqual(pe.amplitude_aware_permutation_entropy(np.zeros(50), 3), 0.0)

    def test_validation(self):
        """Reject an amplitude outside [0, 1]."""
        for amplitude in (-0.1, 1.1, float("nan")):
            self.assertRaises(ValueError, pe.amplitude_aware_permutation_entropy, X, 3, 1, amplitude)


class TestStatisticalComplexity(unittest.TestCase):
    def _reference(self, x, order, delay=1):
        p = _full_probabilities(x, order, delay)
        n = len(p)
        u = np.full(n, 1.0 / n)

        def shannon(d):
            d = d[d > 0]
            return float(-np.sum(d * np.log(d)))

        js = shannon((p + u) / 2) - shannon(p) / 2 - shannon(u) / 2
        q0 = -2.0 / (((n + 1) / n) * math.log(n + 1) - 2 * math.log(2 * n) + math.log(n))
        return q0 * js * shannon(p) / math.log(n)

    def test_matches_definition(self):
        """Compute Q_J * H with the Jensen-Shannon divergence over all order! patterns (Rosso et al. 2007)."""
        signals = (X, np.sin(np.arange(300) / 4), np.cumsum(X))
        for x, order in itertools.product(signals, (2, 3, 4)):
            with self.subTest(order=order):
                self.assertAlmostEqual(pe.statistical_complexity(x, order), self._reference(x, order), places=12)

    def test_bounds(self):
        """Give 0 for a monotonic series, near 0 for long white noise, and stay within [0, 1]."""
        self.assertEqual(pe.statistical_complexity(np.arange(100.0), 3), 0.0)
        self.assertLess(pe.statistical_complexity(np.random.default_rng(1).standard_normal(50000), 3), 0.01)
        self.assertTrue(0.0 <= pe.statistical_complexity(np.sin(np.arange(500) / 3), 5) <= 1.0)

    def test_large_order_is_finite(self):
        """Stay accurate for huge pattern spaces (20! patterns)."""
        c = pe.statistical_complexity(np.random.default_rng(2).standard_normal(3000), 20)
        self.assertTrue(np.isfinite(c) and 0.0 <= c <= 1.0)


class TestBubbleEntropy(unittest.TestCase):
    @staticmethod
    def _swaps(window):
        w = list(window)
        swaps = 0
        for i in range(len(w)):
            for j in range(len(w) - 1 - i):
                if w[j] > w[j + 1]:
                    w[j], w[j + 1] = w[j + 1], w[j]
                    swaps += 1
        return swaps

    def _reference(self, x, m, delay=1):
        def renyi2(length):
            n = len(x) - (length - 1) * delay
            counts = Counter(self._swaps(x[i : i + length * delay : delay]) for i in range(n))
            p = np.array(list(counts.values())) / n
            return -math.log(np.sum(p**2))

        return (renyi2(m + 1) - renyi2(m)) / math.log((m + 1) / (m - 1))

    def test_matches_bubble_sort(self):
        """Count bubble sort swaps and compare Renyi-2 entropies at m and m + 1 (Manis et al. 2017)."""
        for x, m, delay in ((X[:200], 2, 1), (X[:200], 5, 2), (TIES, 4, 1), (np.sin(np.arange(150) / 3), 6, 1)):
            with self.subTest(m=m, delay=delay):
                self.assertAlmostEqual(pe.bubble_entropy(x, m, delay), self._reference(x, m, delay), places=12)

    def test_large_integers_are_compared_exactly(self):
        """Compare integers above 2**53 as integers."""
        big = np.array([2**53, 2**53 + 1, 2**53, 2**53 + 3, 2**53 + 2, 2**53 + 1] * 5, dtype=np.int64)
        self.assertAlmostEqual(pe.bubble_entropy(big, 2), self._reference(big.tolist(), 2), places=12)

    def test_validation(self):
        """Require m >= 2 and enough samples."""
        self.assertRaises(ValueError, pe.bubble_entropy, X, 1)
        self.assertRaises(TypeError, pe.bubble_entropy, X, 2.0)
        self.assertRaises(ValueError, pe.bubble_entropy, [1.0, 2.0, 3.0], 3)


class TestOrdinalMeasuresInMultiscale(unittest.TestCase):
    def test_new_measures_are_registered(self):
        """Use the new ordinal measures with all three multiscale methods; scale 1 is the measure itself."""
        for measure, params in (
            (pe.amplitude_aware_permutation_entropy, {"order": 3}),
            (pe.statistical_complexity, {"order": 4}),
            (pe.permutation_entropy, {"order": 3, "entropy": "renyi", "q": 2.0}),
        ):
            expected = measure(X, **params)
            for method in (pe.multiscale, pe.composite_multiscale, pe.refined_composite_multiscale):
                with self.subTest(measure=measure.__name__, method=method.__name__):
                    self.assertEqual(method(X, measure, 1, **params)[0], expected)


if __name__ == "__main__":
    unittest.main()
