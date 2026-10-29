"""Input checks shared by all entropy measures."""

import functools
import os
import sys
import warnings

import numpy as np

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn(message, category):
    """Issue a warning attributed to the first caller outside pyentrp, however deep the call."""
    frame, level = sys._getframe(0), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _check_integer(name, value, minimum, maximum=None):
    """Validate that `value` is an integer (booleans excluded) in [minimum, maximum].

    Raises
    ------
    TypeError
        If value is not an integer.
    ValueError
        If value is below the minimum or above the maximum (when given).

    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}.")


def _validate_series(time_series, keep_integer=False):
    """Return the time series as a 1D float array.

    Parameters
    ----------
    time_series : array_like
        Input series.
    keep_integer : bool, optional
        If True, return integer input with its dtype unchanged instead of converting it
        to float. Use this where values are only compared: float64 cannot tell apart
        integers above 2**53 (e.g. 2**53 and 2**53 + 1).

    Raises
    ------
    ValueError
        If the series is not 1D or contains NaN or infinite values.

    """
    time_series = np.asarray(time_series)
    if not (keep_integer and np.issubdtype(time_series.dtype, np.integer)):
        time_series = time_series.astype(float, copy=False)

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array.")

    if not np.all(np.isfinite(time_series)):
        raise ValueError("time_series must contain only finite values.")

    return time_series


def _check_tolerance(tolerance):
    """Validate an explicitly supplied tolerance.

    Raises
    ------
    ValueError
        If tolerance is not greater than 0 (this includes NaN).

    """
    if not tolerance > 0:
        raise ValueError("tolerance must be greater than 0.")


def _default_tolerance(time_series, factor):
    """Return the default template-matching tolerance, factor * std(time_series).

    A constant series has std 0, and a zero tolerance matches nothing under the strict
    `<` test. Every positive tolerance gives the same match counts on a constant series
    (all differences are 0), so return 1.0 there: the result then equals the one for any
    explicit tolerance, i.e. the limit as the tolerance goes to 0 from above.

    """
    if time_series.size and np.ptp(time_series) == 0:
        return 1.0
    return factor * np.std(time_series)


def _resolve_tolerance(time_series, tolerance, factor):
    """Return the tolerance to use: the default factor * std(time_series) if None, else the checked value."""
    if tolerance is None:
        return _default_tolerance(time_series, factor)
    _check_tolerance(tolerance)
    return tolerance


def _check_embedding_length(time_series, order, delay):
    """Check that the series holds at least one window of `order` samples spaced `delay` apart.

    Raises
    ------
    ValueError
        If the series has fewer than (order - 1) * delay + 1 samples.

    """
    min_length = (order - 1) * delay + 1
    if len(time_series) < min_length:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least (order - 1) * delay + 1 ({min_length})."
        )


def _renamed(**old_to_new):
    """Accept deprecated keyword names: ``@_renamed(old="new")`` maps ``old=`` to ``new=`` with a warning.

    Positional arguments are unaffected. Passing both names raises TypeError.

    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in old_to_new.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{func.__name__}() got both {old!r} and {new!r}; pass only {new!r}.")
                    _warn(
                        f"{func.__name__}(): {old!r} is deprecated and will be removed in pyentrp 4.0; use {new!r}.",
                        DeprecationWarning,
                    )
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def _template_length(sample_length, m):
    """Return the longest template length, m + 1, from `m` or from the deprecated `sample_length`.

    Raises
    ------
    TypeError
        If both or neither are given, or the given one is not an integer.
    ValueError
        If m < 0 or sample_length < 1.

    """
    if m is not None:
        if sample_length is not None:
            raise TypeError("pass only m; sample_length is the deprecated m + 1.")
        _check_integer("m", m, 0)
        return m + 1
    if sample_length is None:
        raise TypeError("missing required argument 'm'.")
    _check_integer("sample_length", sample_length, 1)
    _warn(
        f"sample_length is deprecated and will be removed in pyentrp 4.0; "
        f"use m=sample_length - 1 (sample_length={sample_length} is m={sample_length - 1}).",
        DeprecationWarning,
    )
    return sample_length
