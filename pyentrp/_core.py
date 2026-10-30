"""Embedding, pattern counting and Shannon entropy."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._validation import check_embedding_length, check_integer, renamed, validate_series, warn

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# largest pattern space (order! or classes ** order) counted with np.bincount; larger spaces use np.unique.
BINCOUNT_MAX_PATTERNS = 1_000_000


@renamed(embedding_dimension="order")
def time_delay_embedding(time_series: ArrayLike, order: int, delay: int) -> NDArray[Any]:
    """Calculate time-delayed embedding.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series, shape (n_times)
    order : int
        The embedding dimension (formerly ``embedding_dimension``).
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
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, or the series is too short.

    """
    check_integer("order", order, 1)
    check_integer("delay", delay, 1)
    window = (order - 1) * delay + 1
    view = sliding_window_view(time_series, window_shape=window)
    return view[:, ::delay] if delay > 1 else view


def util_pattern_space(time_series: ArrayLike, lag: int, dim: int) -> NDArray[np.float64]:
    """Create a set of sequences with a given lag and dimension.

    Deprecated, will be removed in pyentrp 4.0: use ``time_delay_embedding(time_series, dim, lag).copy()``
    on a float array, which gives the same result.

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

    Warns
    -----
    DeprecationWarning
        Always.

    """
    warn(
        "util_pattern_space is deprecated and will be removed in pyentrp 4.0; "
        "use time_delay_embedding(np.asarray(time_series, dtype=float), dim, lag).copy().",
        DeprecationWarning,
    )
    time_series = validate_series(time_series)
    check_integer("lag", lag, 1)
    check_integer("dim", dim, 1)
    check_embedding_length(time_series, dim, lag)
    return time_delay_embedding(time_series, dim, lag).copy()


def window_length(order, delay, **_):
    """Return the number of samples spanned by one window."""
    return (order - 1) * delay + 1


def count_ids(ids, n_space, weights=None):
    """Count (or sum weights of) integer pattern ids in [0, n_space - 1].

    Small pattern spaces use np.bincount over the whole space; larger ones relabel the observed ids
    with np.unique first, so memory stays proportional to the number of windows. Zero entries
    may be present in the result.

    """
    if n_space <= BINCOUNT_MAX_PATTERNS:
        return np.bincount(ids, weights=weights, minlength=n_space)
    _, inv = np.unique(ids, return_inverse=True)
    return np.bincount(inv.ravel(), weights=weights)


def count_patterns(patterns, n_space, weights=None):
    """Count (or sum weights of) patterns: 1D integer ids in [0, n_space - 1], or the rows of a 2D array.

    2D rows are used when the pattern space is too large for integer ids (n_space is then ignored).

    """
    if patterns.ndim == 1:
        return count_ids(patterns, n_space, weights)
    _, inv = np.unique(patterns, axis=0, return_inverse=True)
    return np.bincount(inv.ravel(), weights=weights)


def pool_patterns(stats):
    """Pool the (patterns, weights) of several series so that each series has the same total weight.

    Counting the pooled patterns then gives the average of the per-series pattern distributions, as in
    refined composite multiscale permutation and dispersion entropy. A series with no weight is skipped.
    A single series is returned unchanged.

    """
    if len(stats) == 1:
        return stats[0]
    patterns, weights = [], []
    for p, w in stats:
        mass = np.ones(len(p)) if w is None else w
        total = mass.sum()
        if total > 0:
            patterns.append(p)
            weights.append(mass / total)
    if not patterns:
        return stats[0]
    return np.concatenate(patterns), np.concatenate(weights)


def entropy_bits(weights):
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


def generalized_entropy(weights, kind="shannon", q=2.0):
    """Return the Shannon, Renyi or Tsallis entropy of the distribution proportional to `weights`.

    Shannon and Renyi entropies are in bits; the Tsallis entropy has no unit. Zero weights are
    ignored. Gives +0.0 for a single symbol or no weight at all.

    - Shannon: -sum p log2 p
    - Renyi of order q: log2(sum p ** q) / (1 - q)
    - Tsallis of index q: (1 - sum p ** q) / (q - 1)

    """
    if kind == "shannon":
        return entropy_bits(weights)
    w = weights[weights > 0]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    power_sum = float(np.sum(p**q))
    h = math.log2(power_sum) / (1.0 - q) if kind == "renyi" else (1.0 - power_sum) / (q - 1.0)
    return h if h > 0.0 else 0.0


def max_generalized_entropy(n_states, kind="shannon", q=2.0):
    """Return the largest value of `generalized_entropy` over `n_states` symbols (the uniform distribution)."""
    if kind == "tsallis":
        return (1.0 - float(n_states) ** (1.0 - q)) / (q - 1.0)
    return math.log2(n_states)


def shannon_entropy(time_series: ArrayLike | str) -> float:
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
    return entropy_bits(counts)
