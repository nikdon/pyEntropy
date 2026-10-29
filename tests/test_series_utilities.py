import unittest

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import TIME_SERIES


class TestEntropy(unittest.TestCase):
    def test_util_pattern_space(self):
        """Construct the expected embedded pattern space."""
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 0, 2)
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 10, 20)
        np.testing.assert_array_equal(
            ent.util_pattern_space(TIME_SERIES, 2, 3),
            np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]),
        )


class TestValidationHelpers(unittest.TestCase):
    def test_validate_series_returns_float_array(self):
        """Convert list and integer input to a 1D float array."""
        out = ent._validate_series([1, 2, 3])
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(out.shape, (3,))

    def test_validate_series_rejects_bad_shape_and_values(self):
        """Reject multi-dimensional and non-finite input."""
        self.assertRaises(ValueError, ent._validate_series, np.zeros((3, 3)))
        self.assertRaises(ValueError, ent._validate_series, [1.0, np.nan])
        self.assertRaises(ValueError, ent._validate_series, [1.0, np.inf])
        self.assertRaises(ValueError, ent._validate_series, [1.0, -np.inf])

    def test_check_integer(self):
        """Accept Python and NumPy integers; reject bools and floats by type, small values by value."""
        ent._check_integer("x", 3, 1)
        ent._check_integer("x", np.int64(3), 1)
        ent._check_integer("x", np.int32(1), 1)
        self.assertRaises(TypeError, ent._check_integer, "x", True, 1)
        self.assertRaises(TypeError, ent._check_integer, "x", np.bool_(True), 1)
        self.assertRaises(TypeError, ent._check_integer, "x", 2.0, 1)
        self.assertRaises(TypeError, ent._check_integer, "x", "2", 1)
        self.assertRaises(ValueError, ent._check_integer, "x", 0, 1)
        self.assertRaises(ValueError, ent._check_integer, "x", np.int64(-4), 1)


class TestUtilityValidation(unittest.TestCase):
    def test_time_delay_embedding_rejects_bad_parameters(self):
        """Reject zero or non-integer dimension and delay."""
        x = np.arange(5.0)
        self.assertRaises(ValueError, ent.time_delay_embedding, x, 3, 0)
        self.assertRaises(ValueError, ent.time_delay_embedding, x, 0, 1)
        self.assertRaises(TypeError, ent.time_delay_embedding, x, 3, 1.0)

    def test_granulate_rejects_bad_scale(self):
        """Reject zero or non-integer scale."""
        self.assertRaises(ValueError, ent.util_granulate_time_series, np.arange(5.0), 0)
        self.assertRaises(TypeError, ent.util_granulate_time_series, np.arange(5.0), 2.0)
