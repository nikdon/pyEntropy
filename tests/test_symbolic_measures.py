import itertools
import math
import unittest
from collections import Counter

import numpy as np

import pyentrp as pe

RNG = np.random.default_rng(21)
X = RNG.standard_normal(400)


def _cce_curve(x, levels, max_length):
    """CCE(L) from the definition, with explicit pattern tuples."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    q = [min(int(levels * (v - x.min()) / span), levels - 1) for v in x] if span else [0] * len(x)

    def entropy_and_single(length):
        counts = Counter(tuple(q[i : i + length]) for i in range(len(q) - length + 1))
        n = sum(counts.values())
        p = np.array(list(counts.values())) / n
        return float(-np.sum(p * np.log(p))), sum(1 for c in counts.values() if c == 1) / n

    e1 = entropy_and_single(1)[0]
    curve, previous = [], 0.0
    for length in range(1, max_length + 1):
        e, single = entropy_and_single(length)
        curve.append(e - previous + single * e1)
        previous = e
    return np.array(curve), e1


class TestCorrectedConditionalEntropy(unittest.TestCase):
    def test_matches_definition(self):
        """Take the minimum over L of E(L) - E(L - 1) + perc(L) * E(1) (Porta et al. 1998)."""
        for x in (X, np.sin(np.arange(300) / 5), np.cumsum(X)):
            for levels in (3, 6):
                curve, e1 = _cce_curve(x, levels, 8)
                with self.subTest(levels=levels):
                    self.assertAlmostEqual(
                        pe.corrected_conditional_entropy(x, levels, max_length=8), curve.min(), places=12
                    )
                    self.assertAlmostEqual(
                        pe.corrected_conditional_entropy(x, levels, max_length=8, normalize=True),
                        curve.min() / e1,
                        places=12,
                    )

    def test_default_length_runs_until_every_pattern_is_single(self):
        """Grow L until every pattern occurs once; a longer search does not change the result here."""
        curve, _ = _cce_curve(X, 6, 12)
        self.assertAlmostEqual(pe.corrected_conditional_entropy(X), curve.min(), places=12)

    def test_constant_and_validation(self):
        """Give 0.0 for a constant series and reject bad parameters."""
        self.assertEqual(pe.corrected_conditional_entropy(np.ones(50)), 0.0)
        self.assertRaises(ValueError, pe.corrected_conditional_entropy, X, 1)
        self.assertRaises(ValueError, pe.corrected_conditional_entropy, X, 6, 0)
        self.assertRaises(TypeError, pe.corrected_conditional_entropy, X, 6.0)
        self.assertRaises(ValueError, pe.corrected_conditional_entropy, [1.0])


class TestSlopeEntropy(unittest.TestCase):
    PAPER = (8.2, 8.1, 4.4, 3.6, 5.3, 5.4, 8.3, 1.9, 3.7, 8.6, 9.6, 9, 6, 8.7, 6.7, 3.3, 2)
    PAPER += (2.5, 2.7, 4.6, 9.1, 1, 3.1, 1.7, 4.1, 3.8, 6.4, 1.3, 5.7, 3.4, 2.4, 2.1, 4.2)

    @staticmethod
    def _symbol(d, gamma, delta):
        if abs(d) <= delta:
            return 0
        if d > gamma:
            return 2
        if d > delta:
            return 1
        return -2 if d < -gamma else -1

    def _reference(self, x, m, gamma, delta, frequency):
        s = [self._symbol(b - a, gamma, delta) for a, b in itertools.pairwise(x)]
        counts = Counter(tuple(s[i : i + m - 1]) for i in range(len(s) - m + 2))
        c = np.array(list(counts.values()))
        p = c / (len(c) if frequency == "patterns" else c.sum())
        return float(-np.sum(p * np.log2(p)))

    def test_paper_example(self):
        """Reproduce the worked example of Cuesta-Frau (2019), Appendix A: 14 patterns, SlopEn 5.29."""
        self.assertAlmostEqual(pe.slope_entropy(self.PAPER, 3), 5.2967, places=4)

    def test_matches_definition(self):
        """Use the symbol rules of the paper, with thresholds on the boundaries included as written."""
        for x in (X, np.round(X, 1), self.PAPER):
            for m, gamma, delta in ((2, 1.0, 1e-3), (4, 0.5, 0.1), (5, 1.0, 0.0)):
                for frequency in ("patterns", "windows"):
                    with self.subTest(m=m, gamma=gamma, delta=delta, frequency=frequency):
                        self.assertAlmostEqual(
                            pe.slope_entropy(x, m, gamma, delta, frequency=frequency),
                            self._reference(list(x), m, gamma, delta, frequency),
                            places=12,
                        )

    def test_validation(self):
        """Reject m < 2, gamma <= delta, delta < 0 and unknown frequency."""
        self.assertRaises(ValueError, pe.slope_entropy, X, 1)
        self.assertRaises(ValueError, pe.slope_entropy, X, 3, 0.1, 0.1)
        self.assertRaises(ValueError, pe.slope_entropy, X, 3, 1.0, -0.1)
        self.assertRaises(ValueError, pe.slope_entropy, X, 3, frequency="counts")
        self.assertRaises(ValueError, pe.slope_entropy, [1.0, 2.0], 3)


class TestLempelZivComplexity(unittest.TestCase):
    @staticmethod
    def _phrases(seq):
        """LZ76 parsing, written out with list slicing."""
        n, i, c = len(seq), 0, 0
        while i < n:
            length = 1
            while i + length <= n:
                pattern = seq[i : i + length]
                history = seq[: i + length - 1]
                if not any(history[j : j + length] == pattern for j in range(len(history) - length + 1)):
                    break
                length += 1
            c += 1
            i += length
        return c

    def test_matches_lz76_parsing(self):
        """Count the phrases of the Lempel-Ziv 1976 parsing, for binary and larger alphabets."""
        for seq in ([0, 0, 0, 1, 1, 0, 1, 0, 1, 1], RNG.integers(0, 2, 300).tolist(), RNG.integers(0, 5, 200).tolist()):
            self.assertEqual(pe.lempel_ziv_complexity(seq, "none", normalize=False), self._phrases(seq))

    def test_binarization_and_normalization(self):
        """Binarize at the median or mean and normalize by n / log_k(n)."""
        b = (X > np.median(X)).astype(int).tolist()
        c = self._phrases(b)
        self.assertEqual(pe.lempel_ziv_complexity(X, normalize=False), c)
        self.assertAlmostEqual(pe.lempel_ziv_complexity(X), c * math.log2(len(b)) / len(b), places=12)
        m = (X > np.mean(X)).astype(int).tolist()
        self.assertEqual(pe.lempel_ziv_complexity(X, "mean", normalize=False), self._phrases(m))

    def test_extremes(self):
        """Give few phrases for a constant or periodic sequence and about 1 for long random data."""
        self.assertEqual(pe.lempel_ziv_complexity(np.ones(100), "none", normalize=False), 2)
        self.assertEqual(pe.lempel_ziv_complexity(np.ones(100), "none"), 0.0)
        self.assertLess(pe.lempel_ziv_complexity(np.tile([0, 1], 500), "none"), 0.05)
        self.assertGreater(pe.lempel_ziv_complexity(RNG.standard_normal(20000)), 0.9)

    def test_validation(self):
        """Reject unknown symbolization and empty input."""
        self.assertRaises(ValueError, pe.lempel_ziv_complexity, X, "quantile")
        self.assertRaises(ValueError, pe.lempel_ziv_complexity, [])
        self.assertRaises(TypeError, pe.lempel_ziv_complexity, X, normalize=1)


if __name__ == "__main__":
    unittest.main()
