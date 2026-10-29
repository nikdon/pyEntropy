import inspect
import unittest
import warnings

import numpy as np

import pyentrp
from pyentrp import entropy as ent
from tests._entropy_test_data import RANDOM_TIME_SERIES

X = RANDOM_TIME_SERIES[:300]
R = 0.2 * float(np.std(X))


def _call_quietly(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return func(*args, **kwargs)


class TestSampleLength(unittest.TestCase):
    """sample_length (= m + 1) keeps its meaning, positional or keyword, and warns."""

    CALLS = (
        (ent.sample_entropy, {"tolerance": R}),
        (ent.multiscale_entropy, {"tolerance": R, "scale": 3}),
        (ent.composite_multiscale_entropy, {"tolerance": R, "scale": 3}),
        (ent.rcmse, {"tolerance": R, "scale": 3}),
    )

    def test_m_equals_sample_length_minus_one(self):
        """Give identical results for m=k and sample_length=k + 1 (keyword and positional)."""
        for func, kwargs in self.CALLS:
            for m in (0, 1, 2):
                with self.subTest(func=func.__name__, m=m):
                    new = func(X, m=m, **kwargs)
                    np.testing.assert_array_equal(new, _call_quietly(func, X, sample_length=m + 1, **kwargs))
                    np.testing.assert_array_equal(new, _call_quietly(func, X, m + 1, **kwargs))

    def test_sample_length_warns_at_the_call_site(self):
        """Warn once, attributed to the caller's file, and name the equivalent m."""
        for func, kwargs in self.CALLS:
            with self.subTest(func=func.__name__):
                with self.assertWarnsRegex(DeprecationWarning, r"m=2") as cm:
                    func(X, 3, **kwargs)
                self.assertEqual(cm.filename, __file__)

    def test_m_does_not_warn(self):
        """Use the new API without any DeprecationWarning."""
        for func, kwargs in self.CALLS:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                func(X, m=2, **kwargs)

    def test_both_or_neither_raise(self):
        """Reject passing both names, or neither."""
        for func, kwargs in self.CALLS:
            with self.subTest(func=func.__name__):
                self.assertRaises(TypeError, func, X, 3, m=2, **kwargs)
                self.assertRaises(TypeError, func, X, **kwargs)

    def test_m_validation(self):
        """Accept m = 0 and reject negative or non-integer m."""
        self.assertEqual(ent.sample_entropy(X, m=0, tolerance=R).shape, (1,))
        self.assertRaises(ValueError, ent.sample_entropy, X, m=-1)
        self.assertRaises(TypeError, ent.sample_entropy, X, m=1.0)
        self.assertRaises(TypeError, ent.sample_entropy, X, m=True)


class TestRenamedKeywords(unittest.TestCase):
    """Old keyword names keep working with a DeprecationWarning until 4.0."""

    CASES = (
        (ent.multiscale_entropy, (X,), {"m": 2, "tolerance": R}, "maxscale", "scale", 3),
        (ent.fuzzy_entropy, (X[:120],), {}, "sample_length", "m", 3),
        (ent.fuzzy_entropy, (X[:120],), {}, "n", "exponent", 1.5),
        (ent.multiscale_permutation_entropy, (X, 1, 3), {}, "m", "order", 4),
        (ent.time_delay_embedding, (X, 2), {}, "embedding_dimension", "order", 3),
    )

    def _swap_positional(self, func, args, new):
        # the cases pass the renamed argument by keyword; move order/delay positions accordingly
        if func is ent.multiscale_permutation_entropy:
            return (args[0],), {"delay": args[1], "scale": args[2]}
        if func is ent.time_delay_embedding:
            return (args[0],), {"delay": args[1]}
        return args, {}

    def test_old_name_matches_new_name_and_warns(self):
        """Give identical results for the old and new keyword, warning only for the old one."""
        for func, args, kwargs, old, new, value in self.CASES:
            with self.subTest(func=func.__name__, old=old):
                pos, extra = self._swap_positional(func, args, new)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", DeprecationWarning)
                    expected = func(*pos, **kwargs, **extra, **{new: value})
                with self.assertWarnsRegex(DeprecationWarning, rf"'{old}' is deprecated.*'{new}'") as cm:
                    got = func(*pos, **kwargs, **extra, **{old: value})
                self.assertEqual(cm.filename, __file__)
                np.testing.assert_array_equal(got, expected)

    def test_old_and_new_name_together_raise(self):
        """Reject passing both the old and the new keyword."""
        for func, args, kwargs, old, new, value in self.CASES:
            with self.subTest(func=func.__name__, old=old):
                pos, extra = self._swap_positional(func, args, new)
                self.assertRaises(TypeError, func, *pos, **kwargs, **extra, **{old: value, new: value})

    def test_signatures_show_new_names(self):
        """Expose the new names to help() and IDEs, not the wrapper's *args, **kwargs."""
        self.assertEqual(
            list(inspect.signature(ent.fuzzy_entropy).parameters), ["time_series", "m", "tolerance", "exponent"]
        )
        self.assertIn("scale", inspect.signature(ent.multiscale_entropy).parameters)
        self.assertIn("order", inspect.signature(ent.multiscale_permutation_entropy).parameters)
        self.assertIn("order", inspect.signature(ent.time_delay_embedding).parameters)


class TestPublicNamespace(unittest.TestCase):
    def test_package_exports_public_api(self):
        """Export the same public functions from pyentrp and pyentrp.entropy."""
        self.assertEqual(pyentrp.__all__, ent.__all__)
        for name in pyentrp.__all__:
            self.assertIs(getattr(pyentrp, name), getattr(ent, name))
            self.assertFalse(name.startswith("_"))
