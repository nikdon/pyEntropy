"""Permutation entropy and its weighted and multiscale variants."""

import math

import numpy as np

from ._core import coarse_grain, count_ids, entropy_bits
from ._validation import check_embedding_length, check_integer, renamed, validate_series, warn

# largest permutation order: pattern ids go up to order! - 1, and 20! - 1 is the last that fits in int64.
MAX_PERMUTATION_ORDER = 20


def motif_lehmer_ids(x, order, delay):
    """Encode the ordinal pattern of every window as an integer in [0, order! - 1].

    The pattern is encoded as a Lehmer code (inversion table) in the factorial number
    system. It is built from vectorized pairwise comparisons, so no per-window sorting
    is needed. Ties are ranked by position (the earlier sample comes first), which
    matches a stable argsort.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    order : int
        Pattern length (>= 1).
    delay : int
        Time delay between the samples of a pattern.

    Returns
    -------
    ids : np.ndarray
        Integer array with n - (order - 1) * delay entries (smallest unsigned dtype that
        holds order! - 1, int64 for order >= 13).

    Raises
    ------
    ValueError
        If the series is too short for the given order and delay.

    """
    n = len(x)
    n_win = n - (order - 1) * delay
    if n_win < 0:
        raise ValueError("time_series is too short for the given order and delay.")

    if order == 1:
        return np.zeros(n_win, dtype=np.uint8)

    # comps[k - 1][i] == 1 when x[i + k * delay] < x[i]
    comps = [(x[k * delay :] < x[: n - k * delay]).view(np.uint8) for k in range(1, order)]
    if order == 2:
        return comps[0]

    max_id = math.factorial(order)
    dtype = (
        np.uint8
        if max_id <= 256
        else (np.uint16 if max_id <= 65536 else (np.uint32 if max_id <= 4294967296 else np.int64))
    )

    # digit 0: number of later samples that are smaller than the first sample of the window
    l_buf = comps[0][:n_win].copy()
    for k in range(2, order):
        l_buf += comps[k - 1][:n_win]

    # digit i (radix order - i): the same count for the i-th sample, Horner-style
    lehmer = l_buf.astype(dtype)
    for i in range(1, order - 1):
        s = i * delay
        n_terms = order - 1 - i
        lehmer *= order - i
        if n_terms == 1:
            lehmer += comps[0][s : s + n_win]
        else:
            np.copyto(l_buf, comps[0][s : s + n_win])
            for k in range(2, n_terms + 1):
                l_buf += comps[k - 1][s : s + n_win]
            lehmer += l_buf
    return lehmer


def window_weights(x, order, delay):
    """Compute unnormalized, shift-invariant variance weights of every window.

    Returns order**2 * var(window) for each window (computed from differences to the first
    sample, which avoids cancellation for series with a large offset). The constant factor
    cancels once the weights are normalized.

    Parameters
    ----------
    x : np.ndarray
        1D float time series.
    order : int
        Window length.
    delay : int
        Time delay between the samples of a window.

    Returns
    -------
    weights : np.ndarray
        Float array with n - (order - 1) * delay entries.

    """
    n_win = len(x) - (order - 1) * delay
    if order == 1:
        return np.zeros(n_win)

    x0 = x[:n_win]
    d = x[delay : delay + n_win] - x0
    if order == 2:
        return d * d

    sum_d = d.copy()
    sum_d2 = np.square(d)
    for k in range(2, order):
        np.subtract(x[k * delay : k * delay + n_win], x0, out=d)
        sum_d += d
        np.square(d, out=d)
        sum_d2 += d

    # order * sum(d**2) - sum(d)**2 == order**2 * var(window)
    sum_d2 *= order
    np.square(sum_d, out=d)
    sum_d2 -= d
    np.maximum(sum_d2, 0.0, out=sum_d2)
    return sum_d2


def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """Calculate Permutation Entropy.

    Parameters
    ----------
    time_series : list | np.ndarray
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, order is above 20, time_series is not 1D
        or contains non-finite values, or has fewer than (order - 1) * delay + 1
        samples.

    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf

    .. [2] Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
        measure for time series. Physical Review Letters, 88(17), 174102.

    Examples
    --------
    1. Permutation entropy with order 2

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> round(permutation_entropy(x, order=2), 3)
        0.918

    2. Normalized permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1
        >>> round(permutation_entropy(x, order=3, normalize=True), 3)
        0.589

    """
    # keep integer input as is: converting to float would merge large integers before they are compared
    x = validate_series(time_series, keep_integer=True)
    check_integer("order", order, 1, MAX_PERMUTATION_ORDER)
    check_integer("delay", delay, 1)
    check_embedding_length(x, order, delay)
    # associate a unique integer in [0, order! - 1] to each ordinal pattern
    pe = permutation_bits(x, order, delay)
    n_motifs = math.factorial(order)
    # order 1 has a single pattern, so pe is 0 and there is nothing to normalize by
    if normalize and pe > 0.0:
        pe /= math.log2(n_motifs)
    return pe


def permutation_bits(x, order, delay):
    """Return the unnormalized permutation entropy in bits of a validated series."""
    return entropy_bits(count_ids(motif_lehmer_ids(x, order, delay), math.factorial(order)))


@renamed(m="order")
def multiscale_permutation_entropy(time_series, order, delay, scale):
    """Calculate the Multiscale Permutation Entropy: permutation entropy at scales 1 .. scale.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    order : int
        Order of permutation entropy (formerly ``m``).
    delay : int
        Time delay.
    scale : int
        Maximum scale, inclusive.

    Returns
    -------
    mspe : np.ndarray
        Array of length scale in bits; index i corresponds to scale i + 1. An entry
        is NaN when the coarse-grained series at that scale has fewer than
        (order - 1) * delay + 1 samples.

    Raises
    ------
    TypeError
        If order, delay, or scale is not an integer (booleans excluded).
    ValueError
        If order, delay, or scale is below 1, order is above 20, or time_series
        is not 1D or contains non-finite values.

    Warns
    -----
    RuntimeWarning
        When the coarse-grained series is too short at some scales; those
        entries are NaN.

    References
    ----------
    .. [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer`s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
    .. [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m

    """
    time_series = validate_series(time_series, keep_integer=True)
    check_integer("order", order, 1, MAX_PERMUTATION_ORDER)
    check_integer("delay", delay, 1)
    check_integer("scale", scale, 1)

    min_length = (order - 1) * delay + 1
    mspe = np.full(scale, np.nan)
    for i in range(scale):
        # scale 1 is the series itself; averaging would turn integers into floats and could merge them
        coarse_time_series = time_series if i == 0 else coarse_grain(time_series, i + 1)
        if len(coarse_time_series) < min_length:
            # coarse series only get shorter with scale, so every remaining scale is too short
            warn(
                f"Coarse-grained series is shorter than (order - 1) * delay + 1 ({min_length}) "
                f"from scale {i + 1} on. Returning NaN for those scales.",
                RuntimeWarning,
            )
            break
        mspe[i] = permutation_bits(coarse_time_series, order, delay)
    return mspe


def weighted_permutation_entropy(time_series, order=2, delay=1, normalize=False):
    """Calculate the weighted permutation entropy.

    Weighted permutation entropy captures the information in the amplitude of a signal where
    standard permutation entropy only measures the information in the ordinal pattern, "motif".

    Parameters
    ----------
    time_series : list | np.ndarray
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    wpe : float
        Weighted Permutation Entropy

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, order is above 20, time_series is not 1D
        or contains non-finite values, or has fewer than (order - 1) * delay + 1
        samples.

    References
    ----------
    .. [1] Fadlallah, B., Chen, B., Keil, A., & Príncipe, J. (2013). Weighted-permutation
           entropy: A complexity measure for time series incorporating amplitude information.
           Physical Review E, 87(2), 022911.

    Examples
    --------
    1. Weighted permutation entropy with order 2

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> round(weighted_permutation_entropy(x, order=2), 3)
        0.913

    2. Normalized weighted permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1
        >>> round(weighted_permutation_entropy(x, order=3, normalize=True), 3)
        0.547

    """
    # patterns come from the original values (integers are not rounded); the weights need float
    x = validate_series(time_series, keep_integer=True)
    check_integer("order", order, 1, MAX_PERMUTATION_ORDER)
    check_integer("delay", delay, 1)
    check_embedding_length(x, order, delay)

    lehmer = motif_lehmer_ids(x, order=order, delay=delay)
    weights = window_weights(x.astype(np.float64, copy=False), order=order, delay=delay)

    n_motifs = math.factorial(order)
    # motifs without weight (e.g. only constant windows) contribute nothing
    wpe = entropy_bits(count_ids(lehmer, n_motifs, weights))
    if normalize and wpe > 0.0:
        wpe /= math.log2(n_motifs)
    return wpe
