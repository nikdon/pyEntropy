"""Embedding, coarse-graining, pattern counting and Shannon entropy."""

from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._validation import _check_embedding_length, _check_integer, _validate_series

# largest pattern space (order! or classes ** order) counted with np.bincount; larger spaces use np.unique.
_BINCOUNT_MAX_PATTERNS = 1_000_000


def time_delay_embedding(time_series, embedding_dimension, delay):
    """Calculate time-delayed embedding.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series, shape (n_times)
    embedding_dimension : int
        The embedding dimension (order).
    delay : int
        The delay between embedded points.

    Returns
    -------
    embedded : ndarray
        The embedded time series with shape (n_times - (order - 1) * delay, order).
        This is a read-only view on the input data (no copy is made), so copy it
        before modifying it in place.

    Raises
    ------
    TypeError
        If embedding_dimension or delay is not an integer (booleans excluded).
    ValueError
        If embedding_dimension or delay is below 1, or the series is too short.

    """
    _check_integer("embedding_dimension", embedding_dimension, 1)
    _check_integer("delay", delay, 1)
    window = (embedding_dimension - 1) * delay + 1
    view = sliding_window_view(time_series, window_shape=window)
    return view[:, ::delay] if delay > 1 else view


def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with a given lag and dimension.

    Same as ``time_delay_embedding(time_series, dim, lag)``, returned as a writable float copy.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    lag : int
        Lag between consecutive samples of a sequence.
    dim : int
        Number of samples in a sequence.

    Returns
    -------
    pattern_space: np.ndarray
        Float array of shape (n - lag * (dim - 1), dim).

    Raises
    ------
    TypeError
        If lag or dim is not an integer (booleans excluded).
    ValueError
        If lag or dim is below 1, time_series is not 1D or contains non-finite
        values, or has fewer than lag * (dim - 1) + 1 samples.

    """
    time_series = _validate_series(time_series)
    _check_integer("lag", lag, 1)
    _check_integer("dim", dim, 1)
    _check_embedding_length(time_series, dim, lag)
    return time_delay_embedding(time_series, dim, lag).copy()


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series.

    Parameters
    ----------
    time_series : np.ndarray
        Time series
    scale : int
        Scale factor

    Returns
    -------
    cts : np.ndarray
        Array of coarse-grained time series with a given scale factor

    Raises
    ------
    TypeError
        If scale is not an integer (booleans excluded).
    ValueError
        If scale is below 1, or time_series is not 1D or contains non-finite values.

    """
    time_series = _validate_series(time_series)
    _check_integer("scale", scale, 1)
    return _granulate(time_series, scale)


def _granulate(x, scale):
    """Average non-overlapping blocks of `scale` samples, dropping an incomplete last block."""
    b = len(x) // scale
    return np.mean(x[: b * scale].reshape(b, scale), axis=1)


def _count_ids(ids, n_space, weights=None):
    """Count (or sum weights of) integer pattern ids in [0, n_space - 1].

    Small pattern spaces use np.bincount over the whole space; larger ones relabel the observed ids
    with np.unique first, so memory stays proportional to the number of windows. Zero entries
    may be present in the result.

    """
    if n_space <= _BINCOUNT_MAX_PATTERNS:
        return np.bincount(ids, weights=weights, minlength=n_space)
    _, inv = np.unique(ids, return_inverse=True)
    return np.bincount(inv.ravel(), weights=weights)


def _entropy_bits(weights):
    """Return the Shannon entropy in bits of the distribution proportional to `weights`.

    Zero weights are ignored (0 * log 0 := 0). Gives +0.0, never -0.0, when a single symbol
    has all the weight or when there is no weight at all.

    """
    w = weights[weights > 0]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    h = float(-np.dot(p, np.log2(p)))
    return h if h > 0.0 else 0.0


def shannon_entropy(time_series):
    """Calculate Shannon Entropy of the sample data.

    Parameters
    ----------
    time_series: np.ndarray | list | str
        Sample data; each distinct value (or character of a string) is a symbol.

    Returns
    -------
    ent: float
        The Shannon Entropy in bits; 0.0 when only one symbol occurs.

    Raises
    ------
    ValueError
        If time_series is empty.

    """
    if isinstance(time_series, str):
        counts = np.fromiter(Counter(time_series).values(), dtype=float)
    else:
        _, counts = np.unique(time_series, return_counts=True)

    if counts.size == 0:
        raise ValueError("time_series must not be empty.")
    return _entropy_bits(counts)
