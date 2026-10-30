"""Symbolic measures: corrected conditional entropy, slope entropy and Lempel-Ziv complexity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._validation import check_integer, validate_series

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def quantize(x, levels):
    """Map values to levels 0 .. levels - 1 by equal-width bins over [min, max] (the maximum goes to the top)."""
    span = np.ptp(x)
    if span == 0:
        return np.zeros(len(x), dtype=np.int64)
    return np.minimum(np.floor(levels * (x - np.min(x)) / span).astype(np.int64), levels - 1)


def check_bool(name, value):
    """Validate a boolean flag (NumPy booleans included)."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")


def corrected_conditional_entropy(
    time_series: ArrayLike, levels: int = 6, max_length: int | None = None, normalize: bool = False
) -> float:
    """Calculate the corrected conditional entropy (CCE) regularity index.

    The series is quantized into `levels` equal-width levels. For pattern lengths L = 1, 2, ..., the
    conditional entropy CE(L) = E(L) - E(L - 1) of the next level given the previous L - 1 is corrected
    for the patterns seen only once, which carry no information about what follows:
    CCE(L) = CE(L) + perc(L) * E(1), with perc(L) the fraction of patterns of length L that occur once.
    The minimum over L is the index; it is low for regular series.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    levels : int, default=6
        Number of quantization levels (>= 2); 6 is the value used by Porta et al.
    max_length : int, optional
        Largest pattern length L. By default L grows until every pattern occurs once, after which the
        correction makes CCE(L) close to E(1) and the minimum cannot move much.
    normalize : bool, default=False
        Divide by E(1), the Shannon entropy of the levels, so the result is between 0 and 1
        (1 - result is the regularity index of Porta et al.).

    Returns
    -------
    cce : float
        Minimum corrected conditional entropy over L, in nats (or normalized). 0.0 for a constant series.

    Raises
    ------
    TypeError
        If levels or max_length is not an integer (booleans excluded), or normalize is not a bool.
    ValueError
        If levels < 2, max_length < 1, or time_series is not 1D, contains non-finite values or has fewer
        than two samples.

    References
    ----------
    .. [1] Porta, A., Baselli, G., Liberati, D., Montano, N., Cogliati, C., Gnecchi-Ruscone, T., Malliani,
           A., & Cerutti, S. (1998). Measuring regularity by means of a corrected conditional entropy in
           sympathetic outflow. Biological Cybernetics, 78(1), 71-78.

    Examples
    --------
    >>> import numpy as np
    >>> round(corrected_conditional_entropy(np.sin(np.arange(500) / 4)), 3)
    0.105

    """
    x = validate_series(time_series)
    check_integer("levels", levels, 2)
    if max_length is not None:
        check_integer("max_length", max_length, 1)
    check_bool("normalize", normalize)
    if len(x) < 2:
        raise ValueError("time_series must have at least two samples.")

    symbols = quantize(x, levels)
    n = len(symbols)
    patterns = symbols
    best = math.inf
    e_previous = e_first = 0.0
    length = 0
    while patterns.size and (max_length is None or length < max_length):
        length += 1
        if length > 1:
            # extend each pattern by its next level and relabel, so ids stay small for any length
            extended = patterns[:-1] * levels + symbols[length - 1 :]
            _, patterns = np.unique(extended, return_inverse=True)
            patterns = patterns.ravel()
        counts = np.bincount(patterns)
        counts = counts[counts > 0]
        p = counts / patterns.size
        e = float(-np.dot(p, np.log(p)))
        if length == 1:
            e_first = e
        single = np.count_nonzero(counts == 1) / patterns.size
        best = min(best, e - e_previous + single * e_first)
        e_previous = e
        if single == 1.0 or patterns.size == 1 or length == n:
            break

    if e_first == 0.0:
        return 0.0
    cce = best / e_first if normalize else best
    return cce if cce > 0.0 else 0.0


def slope_symbols(x, gamma, delta):
    """Map each difference of consecutive samples to one of -2, -1, 0, 1, 2 (stored as 0 .. 4)."""
    d = np.diff(x)
    # thresholds as in the paper: 0 for |d| <= delta, +-1 up to gamma inclusive, +-2 beyond
    symbols = np.full(len(d), 2, dtype=np.int64)
    symbols[d > delta] = 3
    symbols[d > gamma] = 4
    symbols[d < -delta] = 1
    symbols[d < -gamma] = 0
    return symbols


def slope_entropy(
    time_series: ArrayLike,
    m: int = 3,
    gamma: float = 1.0,
    delta: float = 1e-3,
    *,
    frequency: Literal["patterns", "windows"] = "patterns",
) -> float:
    """Calculate slope entropy (SlopEn).

    Each difference of consecutive samples becomes a symbol by its slope: 0 when |d| <= delta, +1 or -1
    up to gamma, +2 or -2 beyond. A window of m samples gives a pattern of m - 1 symbols, and slope
    entropy is the Shannon entropy (bits) of the pattern frequencies. Unlike permutation entropy, it
    keeps some amplitude information.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    m : int, default=3
        Window length (m >= 2).
    gamma : float, default=1.0
        Threshold between small and large slopes (gamma > delta); 1 is a 45 degree slope.
    delta : float, default=1e-3
        Threshold for a flat step (delta >= 0).
    frequency : {"patterns", "windows"}, default="patterns"
        Keyword-only. Divide each pattern count by the number of distinct patterns found, as in the
        original algorithm (the values are then not probabilities and the entropy can be negative),
        or by the number of windows, which gives a true probability distribution.

    Returns
    -------
    slopen : float
        Slope entropy in bits.

    Raises
    ------
    TypeError
        If m is not an integer (booleans excluded), or frequency is not a string.
    ValueError
        If m < 2, delta < 0, gamma <= delta, frequency is unknown, or time_series is not 1D, contains
        non-finite values or has fewer than m samples.

    References
    ----------
    .. [1] Cuesta-Frau, D. (2019). Slope entropy: A new time series complexity estimator based on both
           symbolic patterns and amplitude information. Entropy, 21(12), 1167.

    Examples
    --------
    The worked example of the paper (Appendix A), which reports 5.29 (truncated):

    >>> x = [8.2, 8.1, 4.4, 3.6, 5.3, 5.4, 8.3, 1.9, 3.7, 8.6, 9.6, 9, 6, 8.7, 6.7, 3.3, 2,
    ...      2.5, 2.7, 4.6, 9.1, 1, 3.1, 1.7, 4.1, 3.8, 6.4, 1.3, 5.7, 3.4, 2.4, 2.1, 4.2]
    >>> round(slope_entropy(x, m=3), 4)
    5.2967

    """
    x = validate_series(time_series)
    check_integer("m", m, 2)
    if not delta >= 0:
        raise ValueError("delta must be >= 0.")
    if not gamma > delta:
        raise ValueError("gamma must be greater than delta.")
    if not isinstance(frequency, str):
        raise TypeError(f"frequency must be a string, got {type(frequency).__name__}.")
    if frequency not in ("patterns", "windows"):
        raise ValueError("frequency must be 'patterns' or 'windows'.")
    if len(x) < m:
        raise ValueError(f"Time series length ({len(x)}) must be at least m ({m}).")

    symbols = slope_symbols(x, gamma, delta)
    n_win = len(symbols) - (m - 2)
    ids = symbols[:n_win].copy()
    for k in range(1, m - 1):
        ids = ids * 5 + symbols[k : k + n_win]
    counts = np.unique(ids, return_counts=True)[1]
    p = counts / (len(counts) if frequency == "patterns" else n_win)
    return float(-np.dot(p, np.log2(p)))


def lempel_ziv_phrases(symbols):
    """Count the phrases of the Lempel-Ziv (1976) parsing of a sequence of symbols, in O(n).

    Each new phrase is the shortest substring starting where the previous one ended that cannot be
    copied from the text before its last character (Kaspar & Schuster 1987). Whether the current phrase
    can still be copied is answered by a suffix automaton of the text read so far, extended one symbol
    at a time.

    """
    transitions = [{}]
    link = [-1]
    longest = [0]
    last = 0
    state, matched = 0, 0  # automaton state of the current phrase so far, and its length
    phrases = 0
    for c in symbols:
        target = transitions[state].get(c)
        if target is None:  # the phrase cannot be copied any more: it ends here
            phrases += 1
            state, matched = 0, 0
        else:
            state, matched = target, matched + 1

        # add c to the automaton (standard online construction)
        new = len(longest)
        transitions.append({})
        longest.append(longest[last] + 1)
        link.append(0)
        p = last
        while p != -1 and c not in transitions[p]:
            transitions[p][c] = new
            p = link[p]
        if p != -1:
            q = transitions[p][c]
            if longest[p] + 1 == longest[q]:
                link[new] = q
            else:
                clone = len(longest)
                transitions.append(dict(transitions[q]))
                longest.append(longest[p] + 1)
                link.append(link[q])
                while p != -1 and transitions[p].get(c) == q:
                    transitions[p][c] = clone
                    p = link[p]
                link[q] = link[new] = clone
        last = new
        # a split may have moved the current phrase to a shorter (cloned) state
        while matched and longest[link[state]] >= matched:
            state = link[state]
    return phrases + (1 if matched else 0)


def lempel_ziv_complexity(
    time_series: ArrayLike,
    symbolize: Literal["median", "mean", "none"] = "median",
    normalize: bool = True,
) -> float:
    """Calculate the Lempel-Ziv complexity (LZ76) of a series.

    The series is turned into symbols and parsed into the phrases of Lempel & Ziv (1976): each phrase is
    the shortest new substring that cannot be copied from what came before. The number of phrases
    grows with the amount of new patterns, so it measures randomness.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    symbolize : {"median", "mean", "none"}, default="median"
        Binarize at the median or the mean (values above it are 1), or use the values themselves as
        symbols (for data that are already symbolic).
    normalize : bool, default=True
        Return c * log_k(n) / n for n symbols from an alphabet of k, which is close to 1 for a random
        sequence; otherwise return the number of phrases c.

    Returns
    -------
    lzc : float
        Normalized complexity, or the number of phrases. A series with a single symbol gives c = 1, and
        0.0 when normalized.

    Raises
    ------
    TypeError
        If symbolize is not a string or normalize is not a bool.
    ValueError
        If symbolize is unknown, or time_series is empty, not 1D or contains non-finite values.

    References
    ----------
    .. [1] Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions on
           Information Theory, 22(1), 75-81.
    .. [2] Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the complexity of
           spatiotemporal patterns. Physical Review A, 36(2), 842-848.

    Examples
    --------
    >>> lempel_ziv_complexity([0, 0, 0, 1, 1, 0, 1, 0, 1, 1], symbolize="none", normalize=False)
    4.0

    """
    x = validate_series(time_series)
    if not isinstance(symbolize, str):
        raise TypeError(f"symbolize must be a string, got {type(symbolize).__name__}.")
    if symbolize not in ("median", "mean", "none"):
        raise ValueError("symbolize must be 'median', 'mean' or 'none'.")
    check_bool("normalize", normalize)
    if len(x) == 0:
        raise ValueError("time_series must not be empty.")

    if symbolize == "none":
        symbols = np.unique(x, return_inverse=True)[1].ravel()
    else:
        threshold = np.median(x) if symbolize == "median" else np.mean(x)
        symbols = (x > threshold).astype(np.int64)
    alphabet = int(symbols.max()) + 1
    phrases = lempel_ziv_phrases(symbols.tolist())
    if not normalize:
        return float(phrases)
    if alphabet < 2:
        return 0.0
    n = len(symbols)
    return phrases * math.log(n, alphabet) / n
