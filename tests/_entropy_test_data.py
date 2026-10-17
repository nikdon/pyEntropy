import numpy as np

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
