import unittest
import warnings

import numpy as np

import pyentrp as pe

X = np.random.default_rng(17).standard_normal(1000)
R = 0.2 * float(np.std(X))
METHODS = (pe.multiscale, pe.composite_multiscale, pe.refined_composite_multiscale)


def _variance_blocks(x, k):
    b = len(x) // k
    return np.array([np.var(x[i * k : (i + 1) * k], ddof=1) for i in range(b)])


class TestVarianceCoarseGraining(unittest.TestCase):
    def test_multiscale_uses_unbiased_block_variances(self):
        """Apply the measure to the unbiased variances of the blocks (Costa & Goldberger 2015)."""
        res = pe.multiscale(X, pe.permutation_entropy, 5, coarse_graining="variance", order=3)
        self.assertTrue(np.isnan(res[0]))  # the variance of one sample is undefined
        expected = [pe.permutation_entropy(_variance_blocks(X, k), 3) for k in range(2, 6)]
        np.testing.assert_allclose(res[1:], expected, rtol=1e-12)

    def test_composite_averages_shifted_variance_series(self):
        """Average over the k shifted variance-coarse-grained series at scale k."""
        res = pe.composite_multiscale(X, pe.sample_entropy, 4, coarse_graining="variance", m=2, tolerance=0.05)
        expected = [
            np.mean([pe.sample_entropy(_variance_blocks(X[j:], k), m=2, tolerance=0.05)[-1] for j in range(k)])
            for k in range(2, 5)
        ]
        np.testing.assert_allclose(res[1:], expected, rtol=1e-12)

    def test_refined_and_mean_default(self):
        """Work with refined composite, and keep the mean as the default."""
        res = pe.refined_composite_multiscale(X, pe.dispersion_entropy, 3, coarse_graining="variance", classes=4)
        self.assertTrue(np.isnan(res[0]) and np.all(np.isfinite(res[1:])))
        for method in METHODS:
            np.testing.assert_array_equal(
                method(X, pe.permutation_entropy, 3, order=3),
                method(X, pe.permutation_entropy, 3, coarse_graining="mean", order=3),
            )

    def test_validation(self):
        """Reject unknown coarse-graining methods."""
        for method in METHODS:
            self.assertRaises(ValueError, method, X, pe.permutation_entropy, 3, coarse_graining="median")
            self.assertRaises(TypeError, method, X, pe.permutation_entropy, 3, coarse_graining=None)


class TestHierarchical(unittest.TestCase):
    def _nodes(self, x, levels):
        n = 1
        while n * 2 <= len(x):
            n *= 2
        nodes = [np.asarray(x[:n], dtype=float)]
        for k in range(2 ** (levels - 1) - 1):
            p = nodes[k]
            nodes += [(p[0::2] + p[1::2]) / 2, (p[0::2] - p[1::2]) / 2]
        return nodes

    def test_matches_tree_definition(self):
        """Apply the measure to every node: averages and half differences of consecutive pairs (Jiang et al. 2011)."""
        res = pe.hierarchical(X, pe.sample_entropy, 3, m=2, tolerance=R)
        expected = [pe.sample_entropy(node, m=2, tolerance=R)[-1] for node in self._nodes(X, 3)]
        self.assertEqual(len(res), 7)
        np.testing.assert_allclose(res, expected, rtol=1e-12)

    def test_root_is_the_series_cut_to_a_power_of_two(self):
        """Use the first 512 of 1000 samples at the root, and any callable as the measure."""
        res = pe.hierarchical(X, np.std, 1)
        self.assertEqual(res[0], np.std(X[:512]))

    def test_short_nodes_give_nan(self):
        """Give NaN with one warning once the nodes are too short."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = pe.hierarchical(X[:64], pe.permutation_entropy, 6, order=5)
        self.assertEqual(sum("level" in str(w.message) for w in caught), 1)
        self.assertTrue(np.isnan(res[-1]) and np.isfinite(res[0]))

    def test_validation(self):
        """Reject levels < 1 and invalid measure parameters."""
        self.assertRaises(ValueError, pe.hierarchical, X, pe.sample_entropy, 0, m=2)
        self.assertRaises(TypeError, pe.hierarchical, X, pe.sample_entropy, 2.0, m=2)
        self.assertRaises(ValueError, pe.hierarchical, X, pe.permutation_entropy, 2, order=0)


if __name__ == "__main__":
    unittest.main()
