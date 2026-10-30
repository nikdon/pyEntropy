"""Dispersion entropy."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._core import (
    count_patterns,
    generalized_entropy,
    max_generalized_entropy,
    pool_patterns,
    time_delay_embedding,
    window_length,
)
from ._multiscale import Measure, register
from ._validation import check_embedding_length, check_entropy_kind, check_integer, validate_series

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def validate_dispersion_params(
    time_series,
    classes,
    order,
    delay,
    mapping,
    normalize,
):
    """Validate dispersion entropy inputs and canonicalize the mapping.

    Returns
    -------
    time_series : np.ndarray
        The validated time series as a float array.
    mapping : {"ncdf", "linear"}
        The mapping name, lowercased.

    Raises
    ------
    TypeError
        If classes, order, or delay is not an integer (booleans excluded),
        mapping is not a string, or normalize is not a bool.
    ValueError
        If the series is not 1D, contains non-finite values, or has fewer than
        (order - 1) * delay + 1 samples; classes, order, or delay is below its
        minimum of 2, 1, or 1, respectively; or mapping is not 'ncdf' or
        'linear' (case-insensitive).

    """
    time_series = validate_series(time_series)

    check_integer("classes", classes, 2)
    check_integer("order", order, 1)
    check_integer("delay", delay, 1)

    if not isinstance(mapping, str):
        raise TypeError(f"mapping must be a string, got {type(mapping).__name__}.")

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError(f"normalize must be a boolean, got {type(normalize).__name__}.")

    mapping = mapping.lower()
    if mapping not in ("ncdf", "linear"):
        raise ValueError("mapping must be either 'ncdf' or 'linear'.")

    check_embedding_length(time_series, order, delay)

    return time_series, mapping


def mapping_reference(time_series, mapping):
    """Return (loc, spread) used to scale values: mean and std for "ncdf", minimum and range for "linear".

    The spread is 0.0 for a constant series. That is checked with the range, not the std: the std of a
    constant float series can be a tiny nonzero number.

    """
    if np.ptp(time_series) == 0:
        return float(time_series[0]), 0.0
    if mapping == "ncdf":
        return np.mean(time_series), np.std(time_series)
    min_val = np.min(time_series)
    return min_val, np.max(time_series) - min_val


def map_to_classes(time_series, classes, mapping, reference=None):
    """Map a time series to integer classes.

    Parameters
    ----------
    time_series : np.ndarray
        Time series to map.
    classes : int
        Number of classes.
    mapping : {"ncdf", "linear"}
        Mapping used to scale values to [0, 1]. "ncdf" uses the normal CDF;
        "linear" uses min-max scaling.
    reference : tuple, optional
        (loc, spread) from `mapping_reference`. Defaults to that of time_series itself; the
        multiscale methods pass the one of the original series. Azami et al. do this for "ncdf"
        (mean and std); "linear" (minimum and range) follows the same rule.

    Returns
    -------
    class_labels : np.ndarray
        An int64 array with the same shape and labels from 1 through
        ``classes``. If the reference series is constant, every value goes to the
        class of the mean under "ncdf", min(floor(classes / 2) + 1, classes).

    Notes
    -----
    A scaled value y maps to floor(classes * y) + 1, clamped to [1, classes]. Values on
    class boundaries are assigned to the higher class. For "ncdf", class
    boundaries are the standard-normal quantiles k / classes for
    k = 1, ..., classes - 1.

    """
    loc, spread = mapping_reference(time_series, mapping) if reference is None else reference
    if spread == 0:
        return np.full(len(time_series), min(classes // 2 + 1, classes), dtype=np.int64)

    if mapping == "ncdf":
        zscores = (time_series - loc) / spread
        normal = NormalDist()
        boundaries = [normal.inv_cdf(k / classes) for k in range(1, classes)]
        if classes > 16:
            return np.searchsorted(boundaries, zscores, side="right").astype(np.int64) + 1
        # a few vectorized comparisons are faster than a binary search per value
        labels = np.ones(len(zscores), dtype=np.int64)
        for b in boundaries:
            labels += zscores >= b
        return labels

    y = (time_series - loc) / spread
    # clamp both ends: a block mean of the minimum can round to just below the original minimum
    return np.clip(np.floor(classes * y).astype(np.int64) + 1, 1, classes)


def prepare_dispersion(
    time_series, classes=3, order=3, delay=1, mapping="ncdf", normalize=False, *, fluctuation=False, **kind
):
    """Validate the input of dispersion entropy and fix the class mapping on this series."""
    x, mapping = validate_dispersion_params(time_series, classes, order, delay, mapping, normalize)
    if not isinstance(fluctuation, (bool, np.bool_)):
        raise TypeError(f"fluctuation must be a boolean, got {type(fluctuation).__name__}.")
    if fluctuation and order < 2:
        raise ValueError("order must be >= 2 with fluctuation=True (patterns are the order - 1 class changes).")
    if kind:
        check_entropy_kind(kind.get("entropy", "shannon"), kind.get("q", 2.0))
    # fluctuation-based patterns are the class changes z[t + delay] - z[t], one of 2 * classes - 1 values
    n_symbols, length = (2 * classes - 1, order - 1) if fluctuation else (classes, order)
    return x, {
        "classes": classes,
        "order": order,
        "delay": delay,
        "mapping": mapping,
        "reference": mapping_reference(x, mapping),
        "normalize": normalize,
        "fluctuation": fluctuation,
        "n_symbols": n_symbols,
        "length": length,
        **kind,
    }


def pattern_space_fits(n_symbols, length):
    """Return True when every pattern has an int64 id, i.e. n_symbols ** length <= 2 ** 63."""
    # compare logs first: n_symbols ** length can be an enormous integer
    return length * math.log2(n_symbols) <= 62 or int(n_symbols) ** int(length) <= 2**63


def dispersion_patterns(x, *, classes, delay, mapping, reference, fluctuation, n_symbols, length, **_):
    """Return the dispersion pattern of every window: integer ids, or rows for huge pattern spaces.

    The series is mapped to classes with the mean and std (or minimum and range) of the original series.
    With fluctuation=True a pattern is the sequence of class changes inside the window.

    """
    symbols = map_to_classes(x, classes, mapping, reference) - 1  # labels 0 .. classes - 1
    if fluctuation:
        symbols = symbols[delay:] - symbols[:-delay] + (classes - 1)  # changes, shifted to 0 .. 2 * classes - 2
    if not pattern_space_fits(n_symbols, length):
        # identical rows are counted later; integer ids would overflow int64
        return time_delay_embedding(symbols, length, delay), None

    # encode each pattern as a base-`n_symbols` integer
    n_win = len(symbols) - (length - 1) * delay
    ids = symbols[:n_win].copy()
    for k in range(1, length):
        s = k * delay
        ids *= n_symbols
        ids += symbols[s : s + n_win]
    return ids, None


def pattern_counts(stats, n_symbols, length):
    """Count the patterns and return (counts, size of the pattern space)."""
    patterns, weights = stats
    n_patterns = int(n_symbols) ** int(length) if pattern_space_fits(n_symbols, length) else None
    # count_ids uses np.bincount for small pattern spaces and np.unique for large ones
    return count_patterns(patterns, n_patterns, weights), float(n_symbols) ** length


def dispersion_value(stats, n_symbols, length, normalize, entropy="shannon", q=2.0, **_):
    """Return the dispersion entropy from the patterns (bits, or no unit for Tsallis)."""
    counts, n_patterns = pattern_counts(stats, n_symbols, length)
    de = generalized_entropy(counts, entropy, q)
    if normalize and de > 0.0:
        de /= max_generalized_entropy(n_patterns, entropy, q)
    return de


def reverse_dispersion_value(stats, n_symbols, length, normalize, **_):
    """Return sum over all patterns of (p - 1 / K) ** 2 = sum p ** 2 - 1 / K, the distance to white noise."""
    counts, n_patterns = pattern_counts(stats, n_symbols, length)
    p = counts[counts > 0] / counts.sum()
    rde = float(np.dot(p, p)) - 1.0 / n_patterns
    if normalize:
        rde /= 1.0 - 1.0 / n_patterns
    return rde if rde > 0.0 else 0.0


DISPERSION = Measure(prepare_dispersion, dispersion_patterns, dispersion_value, window_length, pool_patterns)
REVERSE_DISPERSION = Measure(
    prepare_dispersion, dispersion_patterns, reverse_dispersion_value, window_length, pool_patterns
)


def dispersion_entropy(
    time_series: ArrayLike,
    classes: int = 3,
    order: int = 3,
    delay: int = 1,
    mapping: Literal["ncdf", "linear"] = "ncdf",
    normalize: bool = False,
    *,
    fluctuation: bool = False,
    entropy: Literal["shannon", "renyi", "tsallis"] = "shannon",
    q: float = 2.0,
) -> float:
    """Calculate Dispersion Entropy (DispEn).

    Dispersion entropy evaluates the complexity of a time series by mapping values
    into a discrete number of classes and analyzing the frequency distribution of
    the resulting dispersion patterns.

    Parameters
    ----------
    time_series : list | np.ndarray
        One-dimensional series of finite values with at least
        (order - 1) * delay + 1 samples.
    classes : int, default=3
        Number of classes c used to discretize the time series (c >= 2).
    order : int, default=3
        Embedding dimension / pattern length (m >= 1).
    delay : int, default=1
        Time delay in samples between consecutive pattern points (d >= 1).
    mapping : {"ncdf", "linear"}, default="ncdf"
        Mapping function to project the signal into [0, 1]. Case-insensitive at runtime;
        type checkers accept the lowercase names.
        - "ncdf": Normal CDF using the series mean and standard deviation.
        - "linear": Linear min-max normalization.
        Values exactly on a class boundary go to the higher class.
    normalize : bool, default=False
        If True, divide by the largest value over all patterns (log2(classes ** order) for Shannon
        and Renyi), so the result is between 0 and 1.
    fluctuation : bool, default=False
        Keyword-only. Fluctuation-based dispersion entropy [2]: a pattern is the sequence of the
        order - 1 class changes inside the window (one of (2 * classes - 1) ** (order - 1)), which
        ignores the absolute level. Needs order >= 2.
    entropy : {"shannon", "renyi", "tsallis"}, default="shannon"
        Keyword-only. Entropy of the pattern distribution: Shannon (bits), Renyi of order q (bits)
        or Tsallis of index q (no unit).
    q : float, default=2.0
        Keyword-only. Order of the Renyi or index of the Tsallis entropy (q > 0, q != 1); unused for
        Shannon.

    Returns
    -------
    de : float
        Dispersion Entropy in bits (or normalized between 0 and 1).
        Returns 0.0 for a constant signal or a single observed pattern.

    Raises
    ------
    TypeError
        If classes, order, or delay is not an integer (booleans excluded),
        mapping or entropy is not a string, or normalize or fluctuation is not a bool.
    ValueError
        If time_series is not 1D, contains non-finite values, or is too short;
        classes, order, or delay is below its minimum of 2, 1, or 1,
        respectively; mapping is not 'ncdf' or 'linear' (case-insensitive);
        order < 2 with fluctuation=True; or entropy or q is invalid.

    References
    ----------
    .. [1] Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A new efficient
           approach to measure the complexity of time series. IEEE Signal
           Processing Letters, 23(5), 610-614.
    .. [2] Azami, H., & Escudero, J. (2018). Amplitude- and fluctuation-based dispersion
           entropy. Entropy, 20(3), 210.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> round(dispersion_entropy(x, classes=3, order=2, delay=1, normalize=True), 3)
    0.667

    """
    return DISPERSION(
        time_series,
        classes=classes,
        order=order,
        delay=delay,
        mapping=mapping,
        normalize=normalize,
        fluctuation=fluctuation,
        entropy=entropy,
        q=q,
    )


register(dispersion_entropy, DISPERSION)


def reverse_dispersion_entropy(
    time_series: ArrayLike,
    classes: int = 3,
    order: int = 3,
    delay: int = 1,
    mapping: Literal["ncdf", "linear"] = "ncdf",
    normalize: bool = False,
    *,
    fluctuation: bool = False,
) -> float:
    """Calculate reverse dispersion entropy (RDE): the distance of the dispersion patterns from white noise.

    RDE = sum over all K possible patterns of (p - 1 / K) ** 2 = sum p ** 2 - 1 / K, with unobserved
    patterns included. It is 0 for a uniform pattern distribution (white noise) and grows with
    regularity, the reverse of dispersion entropy.

    Parameters
    ----------
    time_series : list | np.ndarray
        One-dimensional series of finite values with at least (order - 1) * delay + 1 samples.
    classes, order, delay, mapping
        As in `dispersion_entropy`.
    normalize : bool, default=False
        If True, divide by 1 - 1 / K, the value for a single pattern, so the result is between 0 and 1.
    fluctuation : bool, default=False
        Keyword-only. Use the fluctuation-based patterns of `dispersion_entropy`; K is then
        (2 * classes - 1) ** (order - 1).

    Returns
    -------
    rde : float
        Reverse dispersion entropy (no unit), at least 0.

    Raises
    ------
    TypeError, ValueError
        As in `dispersion_entropy`.

    References
    ----------
    .. [1] Li, Y., Gao, X., & Wang, L. (2019). Reverse dispersion entropy: A new complexity measure
           for sensor signal. Sensors, 19(23), 5203.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> round(reverse_dispersion_entropy(x, classes=3, order=2, normalize=True), 3)
    0.167

    """
    return REVERSE_DISPERSION(
        time_series,
        classes=classes,
        order=order,
        delay=delay,
        mapping=mapping,
        normalize=normalize,
        fluctuation=fluctuation,
    )


register(reverse_dispersion_entropy, REVERSE_DISPERSION)
