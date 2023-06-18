import unittest

import numpy as np

from pyentrp import entropy as ent

np.random.seed(1234567)


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

RANDOM_TIME_SERIES = np.random.rand(1000)


class TestEntropy(unittest.TestCase):
    def test_shannon_entropy_string(self):
        self.assertEqual(round(ent.shannon_entropy(TIME_SERIES_STRING), 5), SHANNON_ENTROPY)

    def test_shannon_entropy_numerical(self):
        self.assertEqual(round(ent.shannon_entropy(TIME_SERIES), 5), SHANNON_ENTROPY)

    def test_sample_entropy(self):
        ts = TS_SAMPLE_ENTROPY
        std_ts = np.std(ts)
        sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
        np.testing.assert_allclose(sample_entropy, np.array([2.26881823, 2.11119024, 2.33537492, 1.79175947]))

    def test_multiscale_entropy(self):
        multi_scale_entropy = ent.multiscale_entropy(RANDOM_TIME_SERIES, 4, maxscale=4)
        np.testing.assert_allclose(
            multi_scale_entropy,
            np.array([2.52572864, 2.31911439, 1.65292302, 1.86075234]),
        )

    def test_permutation_entropy(self):
        self.assertEqual(
            np.round(ent.permutation_entropy(PERM_ENTROPY_BANDT, order=2, delay=1), 3),
            0.918,
        )
        self.assertEqual(
            np.round(ent.permutation_entropy(PERM_ENTROPY_BANDT, order=3, delay=1), 3),
            1.522,
        )
        # Assert that a fully random vector has an entropy of 0.99999...
        self.assertEqual(
            np.round(
                ent.permutation_entropy(RANDOM_TIME_SERIES, order=3, delay=1, normalize=True),
                3,
            ),
            0.999,
        )

    def test_weighted_permutation_entropy(self):
        self.assertEqual(
            np.round(
                ent.weighted_permutation_entropy(PERM_ENTROPY_BANDT, order=2, delay=1),
                3,
            ),
            0.913,
        )
        self.assertEqual(
            np.round(
                ent.weighted_permutation_entropy(PERM_ENTROPY_BANDT, order=3, delay=1),
                3,
            ),
            1.414,
        )
        # Assert that a fully random vector has an entropy of 0.99999...
        self.assertEqual(
            np.round(
                ent.weighted_permutation_entropy(RANDOM_TIME_SERIES, order=3, delay=1, normalize=True),
                3,
            ),
            0.999,
        )

    def test_multiscale_permutation_entropy(self):
        np.testing.assert_array_equal(
            np.round(ent.multiscale_permutation_entropy(TS_SAMPLE_ENTROPY, 3, 5, 2), 4),
            np.array([2.4699, 2.5649]),
        )

    def test_util_pattern_space(self):
        self.assertRaises(Exception, ent.util_pattern_space, (TIME_SERIES, 0, 2))
        self.assertRaises(Exception, ent.util_pattern_space, (TIME_SERIES, 10, 20))
        np.testing.assert_array_equal(
            ent.util_pattern_space(TIME_SERIES, 2, 3),
            np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]),
        )

    def test_composite_multiscale_entropy(self):
        signal = np.cos(np.linspace(start=0, stop=30, num=100))
        res = ent.composite_multiscale_entropy(signal, sample_length=3, scale=3)
        np.testing.assert_allclose(res, np.array([0.33085424, 0.19283124, 0.94056984]))


if __name__ == "__main__":
    unittest.main()
