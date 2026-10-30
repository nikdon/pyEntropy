import math
import unittest

import numpy as np

import pyentrp as pe
from pyentrp._spectral import DAUBECHIES, dwt_step

RNG = np.random.default_rng(31)
X = RNG.standard_normal(500)


def _entropy_bits(power):
    p = power[power > 0] / power.sum()
    return float(-np.sum(p * np.log2(p)))


def _filters(name):
    low = np.array(DAUBECHIES[name])
    taps = len(low)
    return low, np.array([(-1) ** (k + 1) * low[taps - 1 - k] for k in range(taps)])


class TestSpectralEntropy(unittest.TestCase):
    def _periodogram(self, x):
        """One-sided periodogram from the DFT definition, after removing the mean."""
        x = x - x.mean()
        n = len(x)
        k = np.arange(n // 2 + 1)
        dft = np.exp(-2j * np.pi * np.outer(k, np.arange(n)) / n) @ x
        power = np.abs(dft) ** 2 / n
        power[1 : None if n % 2 else -1] *= 2
        return power

    def test_periodogram(self):
        """Take the entropy of the normalized periodogram, for even and odd lengths."""
        for x in (X, X[:333], np.sin(np.arange(256) / 3)):
            power = self._periodogram(x)
            self.assertAlmostEqual(pe.spectral_entropy(x), _entropy_bits(power), places=10)
            self.assertAlmostEqual(
                pe.spectral_entropy(x, normalize=True), _entropy_bits(power) / math.log2(len(power)), places=10
            )

    def test_extremes(self):
        """Give about 0 for a pure tone on a frequency bin and 0.0 for a constant series."""
        t = np.arange(1000) / 100
        self.assertLess(pe.spectral_entropy(np.sin(2 * np.pi * 5 * t), sf=100, normalize=True), 1e-6)
        self.assertEqual(pe.spectral_entropy(np.ones(100)), 0.0)
        self.assertGreater(pe.spectral_entropy(RNG.standard_normal(4096), normalize=True), 0.9)

    def test_band_and_welch(self):
        """Restrict to a frequency band, and average half-overlapping Hann-windowed segments for Welch."""
        nperseg = 64
        step = nperseg // 2
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
        segments = [X[s : s + nperseg] for s in range(0, len(X) - nperseg + 1, step)]
        power = np.mean([np.abs(np.fft.rfft((seg - seg.mean()) * window)) ** 2 for seg in segments], axis=0)
        power[1:-1] *= 2
        self.assertAlmostEqual(pe.spectral_entropy(X, method="welch", nperseg=nperseg), _entropy_bits(power), places=12)
        freqs = np.fft.rfftfreq(nperseg, 1 / 10)
        keep = (freqs >= 1) & (freqs <= 3)
        self.assertAlmostEqual(
            pe.spectral_entropy(X, 10, "welch", nperseg, band=(1, 3)), _entropy_bits(power[keep]), places=12
        )

    def test_validation(self):
        """Reject bad sampling frequency, method, segment length and band."""
        self.assertRaises(ValueError, pe.spectral_entropy, X, 0)
        self.assertRaises(ValueError, pe.spectral_entropy, X, 1, "multitaper")
        self.assertRaises(ValueError, pe.spectral_entropy, X, 1, "welch", 1)
        self.assertRaises(ValueError, pe.spectral_entropy, X, 10, band=(3, 1))
        self.assertRaises(ValueError, pe.spectral_entropy, X, 10, band=(1, 6))
        self.assertRaises(ValueError, pe.spectral_entropy, [1.0])


class TestSvdEntropy(unittest.TestCase):
    def test_matches_definition(self):
        """Take the entropy of the normalized singular values of the embedding matrix (Roberts et al. 1999)."""
        for order, delay in ((2, 1), (3, 1), (5, 3)):
            n = len(X) - (order - 1) * delay
            embedded = np.stack([X[k * delay : k * delay + n] for k in range(order)], axis=1)
            s = np.linalg.svd(embedded, compute_uv=False)
            with self.subTest(order=order, delay=delay):
                self.assertAlmostEqual(pe.svd_entropy(X, order, delay), _entropy_bits(s), places=12)
                self.assertAlmostEqual(
                    pe.svd_entropy(X, order, delay, True), _entropy_bits(s) / math.log2(order), places=12
                )

    def test_rank_one(self):
        """Give 0 for a rank-one embedding (a constant series) and for order 1."""
        self.assertEqual(pe.svd_entropy(np.ones(50), 3), 0.0)
        self.assertEqual(pe.svd_entropy(X, 1), 0.0)


class TestWaveletEntropy(unittest.TestCase):
    def test_filters_are_orthonormal(self):
        """Use orthonormal Daubechies filters: unit norm, sum sqrt(2), orthogonal even shifts."""
        for name in DAUBECHIES:
            low, high = _filters(name)
            with self.subTest(name=name):
                self.assertAlmostEqual(np.dot(low, low), 1.0, places=12)
                self.assertAlmostEqual(low.sum(), math.sqrt(2), places=12)
                self.assertAlmostEqual(np.dot(low, high), 0.0, places=12)
                for shift in range(2, len(low), 2):
                    self.assertAlmostEqual(np.dot(low[shift:], low[:-shift]), 0.0, places=12)

    def test_transform_conserves_energy(self):
        """Keep the energy of an even-length signal in its approximation and detail coefficients."""
        for name in ("db1", "db4", "db10"):
            low, high = _filters(name)
            approx, detail = dwt_step(X[:256], low, high)
            self.assertAlmostEqual(np.dot(approx, approx) + np.dot(detail, detail), np.dot(X[:256], X[:256]))

    def test_matches_relative_energies(self):
        """Take the entropy of the detail energy of each level, relative to their sum (Rosso et al. 2001)."""
        low, high = _filters("db2")
        approx, energies = X, []
        for _ in range(5):
            approx, detail = dwt_step(approx, low, high)
            energies.append(np.dot(detail, detail))
        energies = np.array(energies)
        self.assertAlmostEqual(pe.wavelet_entropy(X, "db2", 5), _entropy_bits(energies), places=12)
        self.assertAlmostEqual(
            pe.wavelet_entropy(X, "db2", 5, normalize=True), _entropy_bits(energies) / math.log2(5), places=12
        )

    def test_haar_example(self):
        """Put all the detail energy of an alternating series in the first Haar level."""
        x = np.tile([1.0, -1.0], 64)
        self.assertEqual(pe.wavelet_entropy(x, "db1"), 0.0)

    def test_validation(self):
        """Reject unknown wavelets and too many levels."""
        self.assertRaises(ValueError, pe.wavelet_entropy, X, "sym4")
        self.assertRaises(ValueError, pe.wavelet_entropy, X, "db4", 50)
        self.assertRaises(ValueError, pe.wavelet_entropy, X, "db4", 1)
        self.assertRaises(ValueError, pe.wavelet_entropy, X[:10], "db10")


if __name__ == "__main__":
    unittest.main()
