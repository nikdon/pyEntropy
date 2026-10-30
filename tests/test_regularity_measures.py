import math
import unittest

import numpy as np

import pyentrp as pe
from pyentrp._regularity import closest_pair_distance

RNG = np.random.default_rng(13)
X = RNG.standard_normal(300)
SIGNALS = (X, np.sin(np.arange(250) / 3) + 0.1 * RNG.standard_normal(250), RNG.integers(0, 4, 200).astype(float))


def _windows(x, m, delay=1):
    n = len(x) - (m - 1) * delay
    return np.stack([x[k * delay : k * delay + n] for k in range(m)], axis=1)


def _apen(x, m, r):
    def phi(k):
        w = _windows(x, k)
        d = np.max(np.abs(w[:, None, :] - w[None, :, :]), axis=2)
        return np.mean(np.log(np.mean(d <= r, axis=1)))

    return phi(m) - phi(m + 1)


def _range_distance_matches(x, m, r):
    """Match matrix of the range distance for templates of m values (identical templates match)."""
    w = _windows(x, m)
    a = np.abs(w[:, None, :] - w[None, :, :])
    hi, lo = a.max(axis=2), a.min(axis=2)
    with np.errstate(invalid="ignore"):
        d = np.where(hi + lo == 0, 0.0, (hi - lo) / np.where(hi + lo == 0, 1.0, hi + lo))
    return d <= r


class TestApproximateEntropy(unittest.TestCase):
    def test_matches_pincus(self):
        """Use d <= r, self-matches and the 1 / (N - m + 1) normalization (Pincus 1991)."""
        for x in SIGNALS:
            for m in (1, 2, 3):
                r = 0.2 * np.std(x)
                with self.subTest(m=m):
                    self.assertAlmostEqual(pe.approximate_entropy(x, m, r), _apen(x, m, r), places=12)

    def test_long_series_use_the_sorted_band(self):
        """Give the same value through the sorted band search (>= 512 templates)."""
        x = RNG.standard_normal(900)
        self.assertAlmostEqual(pe.approximate_entropy(x, 2, 0.25), _apen(x, 2, 0.25), places=12)

    def test_defaults_and_edge_cases(self):
        """Default to r = 0.2 std, give 0 for a constant series, and accept the shortest series."""
        self.assertEqual(pe.approximate_entropy(X), pe.approximate_entropy(X, 2, 0.2 * np.std(X)))
        self.assertEqual(pe.approximate_entropy(np.ones(40)), 0.0)
        self.assertAlmostEqual(
            pe.approximate_entropy([1.0, 2.0, 1.5], 2, 1.0), _apen(np.array([1.0, 2.0, 1.5]), 2, 1.0)
        )

    def test_validation(self):
        """Reject bad m, tolerance and short series."""
        self.assertRaises(ValueError, pe.approximate_entropy, X, 0)
        self.assertRaises(TypeError, pe.approximate_entropy, X, 2.0)
        self.assertRaises(ValueError, pe.approximate_entropy, X, 2, -1.0)
        self.assertRaises(ValueError, pe.approximate_entropy, [1.0, 2.0], 2)

    def test_multiscale_but_not_refined(self):
        """Work with multiscale and composite; refuse refined composite (ApEn has no pooled form)."""
        self.assertEqual(pe.multiscale(X, pe.approximate_entropy, 1, m=2)[0], pe.approximate_entropy(X, 2))
        self.assertRaisesRegex(TypeError, "sample_entropy", pe.refined_composite_multiscale, X, pe.approximate_entropy)


class TestRangeEntropy(unittest.TestCase):
    def test_sample_variant(self):
        """Count pairs of distinct templates among the first N - m, as sample entropy does (RangeEn-B)."""
        for x in SIGNALS:
            for m, r in ((1, 0.2), (2, 0.3), (3, 0.5)):
                n1 = len(x) - m
                iu = np.triu_indices(n1, 1)
                b = np.count_nonzero(_range_distance_matches(x, m, r)[:n1, :n1][iu])
                a = np.count_nonzero(_range_distance_matches(x, m + 1, r)[iu])
                with self.subTest(m=m, r=r):
                    self.assertAlmostEqual(pe.range_entropy(x, m, r), math.log(b / a), places=12)

    def test_approximate_variant(self):
        """Include self-matches and average logs per template (RangeEn-A)."""
        for x in SIGNALS:
            for m, r in ((1, 0.2), (2, 0.4)):
                phi_m = np.mean(np.log(np.mean(_range_distance_matches(x, m, r), axis=1)))
                phi_m1 = np.mean(np.log(np.mean(_range_distance_matches(x, m + 1, r), axis=1)))
                with self.subTest(m=m, r=r):
                    self.assertAlmostEqual(pe.range_entropy(x, m, r, method="approximate"), phi_m - phi_m1, places=12)

    def test_amplitude_invariance(self):
        """Give the same value after scaling the signal (the range distance is scale-free)."""
        self.assertAlmostEqual(pe.range_entropy(X, 2, 0.3), pe.range_entropy(1000 * X, 2, 0.3), places=12)

    def test_validation(self):
        """Reject tolerances outside [0, 1] and unknown methods."""
        self.assertRaises(ValueError, pe.range_entropy, X, 2, 1.5)
        self.assertRaises(ValueError, pe.range_entropy, X, 2, 0.2, method="fuzzy")
        self.assertRaises(TypeError, pe.range_entropy, X, 2, 0.2, method=None)
        self.assertRaises(ValueError, pe.range_entropy, [1.0, 2.0, 3.0], 2)


class TestCosineSimilarityEntropy(unittest.TestCase):
    def _reference(self, x, m, r, delay=1, zero_median=False):
        x = x - np.median(x) if zero_median else x
        w = _windows(x, m, delay)
        norms = np.linalg.norm(w, axis=1)
        iu = np.triu_indices(len(w), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.clip((w @ w.T) / np.outer(norms, norms), -1, 1)
        angle = np.arccos(cos)[iu] / np.pi
        similar = np.where(np.isnan(angle), False, angle <= r)
        b = np.mean(similar)
        return 0.0 if b in (0.0, 1.0) else -(b * math.log2(b) + (1 - b) * math.log2(1 - b))

    def test_matches_angular_definition(self):
        """Count pairs with arccos(cos) / pi <= r and take the binary entropy (Chanwimalueang & Mandic 2017)."""
        for x in SIGNALS[:2]:
            for m, r, delay, zm in ((2, 0.07, 1, False), (3, 0.1, 2, True), (1, 0.3, 1, False)):
                with self.subTest(m=m, r=r, delay=delay, zero_median=zm):
                    self.assertAlmostEqual(
                        pe.cosine_similarity_entropy(x, m, r, delay, zero_median=zm),
                        self._reference(x, m, r, delay, zm),
                        places=12,
                    )

    def test_zero_norm_templates_match_nothing(self):
        """Leave out templates of zero norm, whose angle is undefined."""
        x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, -1.0, 0.5, 0.0, 0.0, 3.0])
        self.assertAlmostEqual(pe.cosine_similarity_entropy(x, 2, 0.2), self._reference(x, 2, 0.2), places=12)

    def test_validation(self):
        """Reject tolerances outside (0, 1)."""
        for r in (0.0, 1.0, -0.1):
            self.assertRaises(ValueError, pe.cosine_similarity_entropy, X, 2, r)


class TestDistributionEntropy(unittest.TestCase):
    def _reference(self, x, m, bins, delay=1, normalize=True):
        w = _windows(x, m, delay)
        iu = np.triu_indices(len(w), 1)
        d = np.max(np.abs(w[:, None, :] - w[None, :, :]), axis=2)[iu]
        counts = np.histogram(d, bins=bins)[0]
        p = counts[counts > 0] / len(d)
        h = -np.sum(p * np.log2(p))
        return h / math.log2(bins) if normalize else h

    def test_matches_histogram_of_all_distances(self):
        """Bin all pairwise Chebyshev distances between their smallest and largest value (Li et al. 2015)."""
        for x in SIGNALS:
            for m, bins, delay in ((2, 512, 1), (3, 64, 2), (1, 16, 1)):
                with self.subTest(m=m, bins=bins, delay=delay):
                    self.assertAlmostEqual(
                        pe.distribution_entropy(x, m, delay, bins), self._reference(x, m, bins, delay), places=12
                    )
        self.assertAlmostEqual(
            pe.distribution_entropy(X, 2, normalize=False), self._reference(X, 2, 512, normalize=False), places=12
        )

    def test_bin_rules(self):
        """Apply Sturges, Rice and square-root rules to the number of distances."""
        n_pairs = 299 * 298 // 2
        for rule, bins in (
            ("sturges", math.ceil(math.log2(n_pairs) + 1)),
            ("rice", math.ceil(2 * n_pairs ** (1 / 3))),
            ("sqrt", math.ceil(math.sqrt(n_pairs))),
        ):
            self.assertAlmostEqual(pe.distribution_entropy(X, 2, bins=rule), self._reference(X, 2, bins), places=12)

    def test_closest_pair(self):
        """Find the exact smallest Chebyshev distance, with and without ties."""
        for x in (X, RNG.integers(0, 50, 400).astype(float), np.cumsum(RNG.standard_normal(700))):
            w = _windows(x, 3)
            iu = np.triu_indices(len(w), 1)
            expected = np.max(np.abs(w[:, None, :] - w[None, :, :]), axis=2)[iu].min()
            self.assertEqual(closest_pair_distance(w), expected)

    def test_edge_cases_and_validation(self):
        """Give 0.0 when every distance is equal, and reject bad bins."""
        self.assertEqual(pe.distribution_entropy(np.ones(30)), 0.0)
        self.assertEqual(pe.distribution_entropy([1.0, 4.0, 2.0], 2), 0.0)  # a single distance
        self.assertRaises(ValueError, pe.distribution_entropy, X, bins=1)
        self.assertRaises(ValueError, pe.distribution_entropy, X, bins="doane")
        self.assertRaises(TypeError, pe.distribution_entropy, X, bins=2.5)


if __name__ == "__main__":
    unittest.main()
