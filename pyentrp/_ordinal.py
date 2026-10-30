"""Ordinal pattern measures: permutation entropy and its variants, statistical complexity, bubble entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._core import count_patterns, generalized_entropy, max_generalized_entropy, pool_patterns, window_length
from ._multiscale import Measure, multiscale, register
from ._validation import (
    check_embedding_length,
    check_entropy_kind,
    check_fraction,
    check_integer,
    renamed,
    validate_series,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

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


def prepare_ordinal(time_series, order=3, delay=1, normalize=False, *, entropy="shannon", q=2.0):
    """Validate the input of the permutation entropies."""
    # keep integer input as is: converting to float would merge large integers before they are compared
    x = validate_series(time_series, keep_integer=True)
    check_integer("order", order, 1, MAX_PERMUTATION_ORDER)
    check_integer("delay", delay, 1)
    check_embedding_length(x, order, delay)
    check_entropy_kind(entropy, q)
    return x, {"order": order, "delay": delay, "normalize": normalize, "entropy": entropy, "q": q}


def prepare_amplitude_aware(time_series, order=3, delay=1, amplitude=0.5, normalize=False, *, entropy="shannon", q=2.0):
    """Validate the input of amplitude-aware permutation entropy."""
    x, p = prepare_ordinal(time_series, order, delay, normalize, entropy=entropy, q=q)
    check_fraction("amplitude", amplitude, low_open=False, high_open=False)
    return x, {**p, "amplitude": amplitude}


def ordinal_patterns(x, order, delay, **_):
    """Return the ordinal pattern id of every window, without weights."""
    return motif_lehmer_ids(x, order, delay), None


def weighted_ordinal_patterns(x, order, delay, **_):
    """Return the ordinal pattern id of every window, weighted by the window's variance."""
    # the weights need float; the patterns come from the original values
    return motif_lehmer_ids(x, order, delay), window_weights(x.astype(np.float64, copy=False), order, delay)


def amplitude_aware_patterns(x, order, delay, amplitude, **_):
    """Return the ordinal pattern id of every window, weighted by its mean absolute value and mean absolute step.

    The weight of a window x_0 .. x_(m-1) is A / m * sum |x_k| + (1 - A) / (m - 1) * sum |x_k - x_(k-1)|.

    """
    xf = x.astype(np.float64, copy=False)
    n_win = len(xf) - (order - 1) * delay
    level = np.abs(xf[:n_win])
    step = np.zeros(n_win)
    for k in range(1, order):
        level += np.abs(xf[k * delay : k * delay + n_win])
        step += np.abs(xf[k * delay : k * delay + n_win] - xf[(k - 1) * delay : (k - 1) * delay + n_win])
    weights = (amplitude / order) * level
    if order > 1:
        weights += ((1.0 - amplitude) / (order - 1)) * step
    return motif_lehmer_ids(x, order, delay), weights


def ordinal_entropy(stats, order, normalize, entropy="shannon", q=2.0, **_):
    """Return the (weighted) permutation entropy from pattern ids and weights (bits, or no unit for Tsallis)."""
    patterns, weights = stats
    n_motifs = math.factorial(order)
    # motifs without weight (e.g. only constant windows) contribute nothing
    pe = generalized_entropy(count_patterns(patterns, n_motifs, weights), entropy, q)
    # order 1 has a single pattern, so pe is 0 and there is nothing to normalize by
    if normalize and pe > 0.0:
        pe /= max_generalized_entropy(n_motifs, entropy, q)
    return pe


def ordinal_complexity(stats, order, **_):
    """Return the Jensen-Shannon statistical complexity C = Q_J[P, P_e] * H[P] of the ordinal distribution."""
    patterns, weights = stats
    n_states = math.factorial(order)
    counts = count_patterns(patterns, n_states, weights)
    observed = counts[counts > 0]
    if observed.size <= 1 or n_states <= 1:
        return 0.0  # a single pattern: H = 0
    p = observed / observed.sum()
    log_n = math.log(n_states)
    shannon = float(-np.dot(p, np.log(p)))
    # Jensen-Shannon divergence to the uniform distribution 1 / N over all N patterns; the unobserved ones
    # contribute (N - k) * u * log(u) with u = 1 / (2 N) to the entropy of the mixture
    mixture = (p + 1.0 / n_states) / 2.0
    u = 1.0 / (2.0 * n_states)
    mixture_entropy = float(-np.dot(mixture, np.log(mixture))) - (n_states - observed.size) * u * math.log(u)
    divergence = mixture_entropy - shannon / 2.0 - log_n / 2.0
    # 1 / Q0 is the largest divergence, reached when P is a single pattern; written to stay accurate for large N
    max_divergence = -(math.log1p(1.0 / n_states) + math.log1p(n_states) / n_states - 2.0 * math.log(2.0)) / 2.0
    c = (divergence / max_divergence) * (shannon / log_n)
    return c if c > 0.0 else 0.0


PERMUTATION = Measure(prepare_ordinal, ordinal_patterns, ordinal_entropy, window_length, pool_patterns)
WEIGHTED_PERMUTATION = Measure(
    prepare_ordinal, weighted_ordinal_patterns, ordinal_entropy, window_length, pool_patterns
)
AMPLITUDE_AWARE_PERMUTATION = Measure(
    prepare_amplitude_aware, amplitude_aware_patterns, ordinal_entropy, window_length, pool_patterns
)
STATISTICAL_COMPLEXITY = Measure(prepare_ordinal, ordinal_patterns, ordinal_complexity, window_length, pool_patterns)


def permutation_entropy(
    time_series: ArrayLike,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False,
    *,
    entropy: Literal["shannon", "renyi", "tsallis"] = "shannon",
    q: float = 2.0,
) -> float:
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
        If True, divide by the largest value for factorial(order) patterns (log2(factorial(order))
        for Shannon and Renyi), so the result is between 0 and 1.
    entropy : {"shannon", "renyi", "tsallis"}, default="shannon"
        Keyword-only. Entropy of the pattern distribution: Shannon (bits), Renyi of order q (bits)
        [Zunino et al. 2008] or Tsallis of index q (no unit).
    q : float, default=2.0
        Keyword-only. Order of the Renyi or index of the Tsallis entropy (q > 0, q != 1); unused for
        Shannon.

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
    return PERMUTATION(time_series, order=order, delay=delay, normalize=normalize, entropy=entropy, q=q)


register(permutation_entropy, PERMUTATION)


@renamed(m="order")
def multiscale_permutation_entropy(time_series: ArrayLike, order: int, delay: int, scale: int) -> NDArray[np.float64]:
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
    return multiscale(time_series, permutation_entropy, scale, order=order, delay=delay)


def weighted_permutation_entropy(
    time_series: ArrayLike,
    order: int = 2,
    delay: int = 1,
    normalize: bool = False,
    *,
    entropy: Literal["shannon", "renyi", "tsallis"] = "shannon",
    q: float = 2.0,
) -> float:
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
        If True, divide by the largest value for factorial(order) patterns (log2(factorial(order))
        for Shannon and Renyi), so the result is between 0 and 1.
    entropy : {"shannon", "renyi", "tsallis"}, default="shannon"
        Keyword-only. Entropy of the pattern distribution: Shannon (bits), Renyi of order q (bits)
        [Zunino et al. 2008] or Tsallis of index q (no unit).
    q : float, default=2.0
        Keyword-only. Order of the Renyi or index of the Tsallis entropy (q > 0, q != 1); unused for
        Shannon.

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
    return WEIGHTED_PERMUTATION(time_series, order=order, delay=delay, normalize=normalize, entropy=entropy, q=q)


register(weighted_permutation_entropy, WEIGHTED_PERMUTATION)


def amplitude_aware_permutation_entropy(
    time_series: ArrayLike,
    order: int = 3,
    delay: int = 1,
    amplitude: float = 0.5,
    normalize: bool = False,
    *,
    entropy: Literal["shannon", "renyi", "tsallis"] = "shannon",
    q: float = 2.0,
) -> float:
    """Calculate amplitude-aware permutation entropy (AAPE).

    Like weighted permutation entropy, each window adds its weight, not 1, to its ordinal pattern.
    The weight of a window x_0 .. x_(m-1) mixes its mean absolute value and its mean absolute step:
    A / m * sum |x_k| + (1 - A) / (m - 1) * sum |x_k - x_(k-1)|. This makes the measure sensitive to
    the amplitude and to spikes, which plain permutation entropy ignores.

    Parameters
    ----------
    time_series : list | np.ndarray
        1D series of finite values.
    order : int, default=3
        Pattern length (1 <= order <= 20).
    delay : int, default=1
        Time delay between the samples of a pattern.
    amplitude : float, default=0.5
        A in [0, 1]: the share of the mean absolute value in the weight; the rest goes to the mean
        absolute step.
    normalize : bool, default=False
        If True, divide by the largest value for factorial(order) patterns, so the result is between
        0 and 1.
    entropy : {"shannon", "renyi", "tsallis"}, default="shannon"
        Keyword-only. Entropy of the pattern distribution: Shannon (bits), Renyi of order q (bits)
        [Zunino et al. 2008] or Tsallis of index q (no unit).
    q : float, default=2.0
        Keyword-only. Order of the Renyi or index of the Tsallis entropy (q > 0, q != 1); unused for
        Shannon.

    Returns
    -------
    aape : float
        Amplitude-aware permutation entropy in bits (no unit for Tsallis). 0.0 when the series has no
        weight at all (only zeros).

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded), or entropy is not a string.
    ValueError
        If order or delay is below 1, order is above 20, amplitude is outside [0, 1], entropy or q is
        invalid, or time_series is not 1D, contains non-finite values or is too short.

    References
    ----------
    .. [1] Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike
           detection and signal segmentation. Computer Methods and Programs in Biomedicine, 128, 40-51.

    Examples
    --------
    >>> round(amplitude_aware_permutation_entropy([4, 7, 9, 10, 6, 11, 3], order=3), 3)
    1.547

    """
    return AMPLITUDE_AWARE_PERMUTATION(
        time_series, order=order, delay=delay, amplitude=amplitude, normalize=normalize, entropy=entropy, q=q
    )


register(amplitude_aware_permutation_entropy, AMPLITUDE_AWARE_PERMUTATION)


def statistical_complexity(time_series: ArrayLike, order: int = 3, delay: int = 1) -> float:
    """Calculate the Jensen-Shannon statistical complexity of the ordinal patterns.

    C = Q_J[P, P_e] * H[P], where H is the normalized permutation entropy of the pattern distribution P
    and Q_J its normalized Jensen-Shannon divergence to the uniform distribution P_e over all
    factorial(order) patterns. C is 0 for a fully ordered and for a fully random series, and largest for
    structured, deterministic dynamics. Together with ``permutation_entropy(..., normalize=True)`` it
    places a series in the complexity-entropy causality plane, which separates chaos from noise.

    Parameters
    ----------
    time_series : list | np.ndarray
        1D series of finite values.
    order : int, default=3
        Pattern length (1 <= order <= 20).
    delay : int, default=1
        Time delay between the samples of a pattern.

    Returns
    -------
    complexity : float
        Statistical complexity in [0, 1].

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, order is above 20, or time_series is not 1D, contains non-finite
        values or is too short.

    References
    ----------
    .. [1] Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007).
           Distinguishing noise from chaos. Physical Review Letters, 99(15), 154102.
    .. [2] Lamberti, P. W., Martin, M. T., Plastino, A., & Rosso, O. A. (2004). Intensive entropic
           non-triviality measure. Physica A, 334(1-2), 119-131.

    Examples
    --------
    >>> round(statistical_complexity([4, 7, 9, 10, 6, 11, 3], order=3), 3)
    0.29

    """
    return STATISTICAL_COMPLEXITY(time_series, order=order, delay=delay)


register(statistical_complexity, STATISTICAL_COMPLEXITY)


def inversion_counts(x, order, delay):
    """Return the number of inversions (pairs a < b with x_b < x_a) of every window of `order` samples.

    This is the number of swaps bubble sort makes to sort the window, which counts only strictly
    larger earlier samples, so ties are never swapped.

    """
    n_win = len(x) - (order - 1) * delay
    counts = np.zeros(n_win, dtype=np.int64)
    for b in range(1, order):
        later = x[b * delay : b * delay + n_win]
        for a in range(b):
            counts += later < x[a * delay : a * delay + n_win]
    return counts


def renyi2_of_counts(values, n_bins):
    """Return the Renyi entropy of order 2, -ln(sum p ** 2), of the distribution of small integers."""
    p = np.bincount(values, minlength=n_bins) / len(values)
    return -math.log(float(np.dot(p, p)))


def bubble_entropy(time_series: ArrayLike, m: int, delay: int = 1) -> float:
    """Calculate bubble entropy.

    Each window is ranked by the number of swaps bubble sort needs to sort it (its number of
    inversions). Bubble entropy compares the Renyi entropy of order 2 of these swap counts for windows
    of m + 1 and m samples: (H(m + 1) - H(m)) / ln((m + 1) / (m - 1)). It needs no tolerance, and depends
    little on m.

    Parameters
    ----------
    time_series : list | np.ndarray
        1D series of finite values.
    m : int
        Embedding dimension (m >= 2). There is no default: the authors study a wide range of m, and
        the measure depends little on it.
    delay : int, default=1
        Time delay between the samples of a window.

    Returns
    -------
    bubble : float
        Bubble entropy (no unit; the ratio of logarithms does not depend on their base).

    Raises
    ------
    TypeError
        If m or delay is not an integer (booleans excluded).
    ValueError
        If m < 2, delay < 1, or time_series is not 1D, contains non-finite values or has fewer than
        m * delay + 1 samples.

    References
    ----------
    .. [1] Manis, G., Aktaruzzaman, M., & Sassi, R. (2017). Bubble entropy: An entropy almost free of
           parameters. IEEE Transactions on Biomedical Engineering, 64(11), 2711-2718.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.arange(200) / 5)
    >>> round(bubble_entropy(x, m=3), 3)
    0.203

    """
    x = validate_series(time_series, keep_integer=True)
    check_integer("m", m, 2)
    check_integer("delay", delay, 1)
    check_embedding_length(x, m + 1, delay)

    inversions_m = inversion_counts(x, m, delay)
    # a window of m + 1 samples adds the inversions of its last sample to those of its first m samples
    n_win = len(x) - m * delay
    last = x[m * delay : m * delay + n_win]
    inversions_m1 = inversions_m[:n_win].copy()
    for a in range(m):
        inversions_m1 += last < x[a * delay : a * delay + n_win]

    h_m = renyi2_of_counts(inversions_m, m * (m - 1) // 2 + 1)
    h_m1 = renyi2_of_counts(inversions_m1, (m + 1) * m // 2 + 1)
    return (h_m1 - h_m) / math.log((m + 1) / (m - 1))
