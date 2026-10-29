import unittest
import warnings

import numpy as np

from pyentrp import entropy as ent
from tests._entropy_test_data import TIME_SERIES


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

    def test_granulate_rejects_bad_series(self):
        """Reject non-1D and non-finite series, like every other public function."""
        self.assertRaises(ValueError, ent.util_granulate_time_series, np.ones((4, 4)), 2)
        self.assertRaises(ValueError, ent.util_granulate_time_series, [1.0, np.nan, 2.0, 3.0], 2)
        np.testing.assert_array_equal(ent.util_granulate_time_series([1, 3, 5, 7, 9], 2), [2.0, 6.0])


class TestPatternSpace(unittest.TestCase):
    """util_pattern_space is deprecated but must keep working until 4.0."""

    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_warns_deprecation(self):
        """Point users to time_delay_embedding."""
        with self.assertWarnsRegex(DeprecationWarning, "time_delay_embedding"):
            ent.util_pattern_space(TIME_SERIES, 2, 3)

    def test_util_pattern_space(self):
        """Construct the expected embedded pattern space."""
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 0, 2)
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 10, 20)
        np.testing.assert_array_equal(
            ent.util_pattern_space(TIME_SERIES, 2, 3),
            np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]),
        )

    def test_accepts_shortest_valid_series(self):
        """Accept a series with exactly lag * (dim - 1) + 1 samples."""
        np.testing.assert_array_equal(ent.util_pattern_space([1, 2, 3], 2, 2), [[1.0, 3.0]])
        self.assertRaises(ValueError, ent.util_pattern_space, [1, 2], 2, 2)

    def test_matches_time_delay_embedding_as_writable_float_copy(self):
        """Return the same rows as time_delay_embedding, as a writable float array."""
        x = np.arange(12)
        res = ent.util_pattern_space(x, 3, 3)
        np.testing.assert_array_equal(res, ent.time_delay_embedding(x, 3, 3))
        self.assertEqual(res.dtype, np.float64)
        self.assertTrue(res.flags.writeable)

    def test_rejects_bad_parameters(self):
        """Reject zero or non-integer lag and dim."""
        self.assertRaises(ValueError, ent.util_pattern_space, TIME_SERIES, 1, 0)
        self.assertRaises(TypeError, ent.util_pattern_space, TIME_SERIES, 1.0, 2)


class TestValidateSeriesKeepInteger(unittest.TestCase):
    def test_keep_integer_preserves_integer_dtype(self):
        """Return integer input unchanged with keep_integer=True, and float input as float."""
        for dtype in (np.int64, np.int32, np.uint64):
            out = ent._validate_series(np.array([1, 2, 3], dtype=dtype), keep_integer=True)
            self.assertEqual(out.dtype, dtype)
        self.assertEqual(ent._validate_series([2**53 + 1], keep_integer=True)[0], 2**53 + 1)
        self.assertEqual(ent._validate_series([1.5, 2.0], keep_integer=True).dtype, np.float64)
        self.assertEqual(ent._validate_series([True, False], keep_integer=True).dtype, np.float64)

    def test_keep_integer_still_validates(self):
        """Reject non-1D integer input and non-finite float input with keep_integer=True."""
        self.assertRaises(ValueError, ent._validate_series, np.zeros((2, 2), dtype=np.int64), keep_integer=True)
        self.assertRaises(ValueError, ent._validate_series, [1.0, np.nan], keep_integer=True)
