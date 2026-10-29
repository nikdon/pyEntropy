"""Dispersion entropy."""

import math
from statistics import NormalDist

import numpy as np

from ._core import _BINCOUNT_MAX_PATTERNS, _count_ids, _entropy_bits, time_delay_embedding
from ._validation import _check_embedding_length, _check_integer, _validate_series


def _validate_dispersion_params(  # noqa: PLR0913
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
    time_series = _validate_series(time_series)

    _check_integer("classes", classes, 2)
    _check_integer("order", order, 1)
    _check_integer("delay", delay, 1)

    if not isinstance(mapping, str):
        raise TypeError(f"mapping must be a string, got {type(mapping).__name__}.")

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError(f"normalize must be a boolean, got {type(normalize).__name__}.")

    mapping = mapping.lower()
    if mapping not in ("ncdf", "linear"):
        raise ValueError("mapping must be either 'ncdf' or 'linear'.")

    _check_embedding_length(time_series, order, delay)

    return time_series, mapping


def _map_to_classes(time_series, classes, mapping):
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

    Returns
    -------
    class_labels : np.ndarray or None
        An int64 array with the same shape and labels from 1 through
        ``classes``. Return None for a constant series.

    Notes
    -----
    A scaled value y maps to min(floor(classes * y) + 1, classes). Values on
    class boundaries are assigned to the higher class. For "ncdf", class
    boundaries are the standard-normal quantiles k / classes for
    k = 1, ..., classes - 1.

    """
    # checked with the range, not the std: the std of a constant float series can be a tiny nonzero number
    if np.ptp(time_series) == 0:
        return None

    if mapping == "ncdf":
        zscores = (time_series - np.mean(time_series)) / np.std(time_series)
        normal = NormalDist()
        boundaries = np.array([normal.inv_cdf(k / classes) for k in range(1, classes)])
        return np.searchsorted(boundaries, zscores, side="right").astype(np.int64) + 1

    min_val = np.min(time_series)
    y = (time_series - min_val) / (np.max(time_series) - min_val)
    return np.minimum(np.floor(classes * y).astype(np.int64) + 1, classes)


def dispersion_entropy(  # noqa: PLR0913
    time_series,
    classes=3,
    order=3,
    delay=1,
    mapping="ncdf",
    normalize=False,
):
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
        Case-insensitive mapping function to project the signal into [0, 1]:
        - "ncdf": Normal CDF using the series mean and standard deviation.
        - "linear": Linear min-max normalization.
        Values exactly on a class boundary go to the higher class.
    normalize : bool, default=False
        If True, divide the entropy by log2(classes**order) to normalize
        the metric between 0 and 1.

    Returns
    -------
    de : float
        Dispersion Entropy in bits (or normalized between 0 and 1).
        Returns 0.0 for a constant signal or a single observed pattern.

    Raises
    ------
    TypeError
        If classes, order, or delay is not an integer (booleans excluded),
        mapping is not a string, or normalize is not a bool.
    ValueError
        If time_series is not 1D, contains non-finite values, or is too short;
        classes, order, or delay is below its minimum of 2, 1, or 1,
        respectively; or mapping is not 'ncdf' or 'linear' (case-insensitive).

    References
    ----------
    .. [1] Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A new efficient
           approach to measure the complexity of time series. IEEE Signal
           Processing Letters, 23(5), 610-614.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> round(dispersion_entropy(x, classes=3, order=2, delay=1, normalize=True), 3)
    0.667

    """
    time_series, mapping = _validate_dispersion_params(time_series, classes, order, delay, mapping, normalize)

    z = _map_to_classes(time_series, classes, mapping)
    if z is None:
        return 0.0

    # compare logs first: classes ** order can be an enormous integer
    if order * math.log(classes) <= math.log(_BINCOUNT_MAX_PATTERNS):
        n_patterns = int(classes) ** int(order)
        # encode each pattern as a base-`classes` integer
        n_win = len(z) - (order - 1) * delay
        z = z - 1  # labels 0 .. classes - 1
        ids = z[:n_win].copy()
        for k in range(1, order):
            s = k * delay
            ids *= classes
            ids += z[s : s + n_win]
        counts = _count_ids(ids, n_patterns)
    else:
        # count identical rows; no integer encoding, which could overflow int64
        patterns = time_delay_embedding(z, order, delay)
        _, counts = np.unique(patterns, axis=0, return_counts=True)

    de = _entropy_bits(counts)
    if normalize and de > 0.0:
        de /= order * math.log2(classes)
    return de
