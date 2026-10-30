"""Template-comparison measures: approximate, range, cosine similarity and distribution entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._matching import band_blocks, count_matches_per_window
from ._multiscale import Measure, register
from ._validation import (
    check_embedding_length,
    check_fraction,
    check_integer,
    resolve_tolerance,
    validate_series,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# rows of templates compared at once in the dense passes; does not change the results
CHUNK_ROWS = 64


def prepare_approximate(time_series, m=2, tolerance=None):
    """Validate the input of approximate entropy and resolve the default tolerance on this series."""
    x = validate_series(time_series)
    check_integer("m", m, 1)
    if len(x) < m + 1:
        raise ValueError(f"Time series length ({len(x)}) must be at least m + 1 ({m + 1}).")
    return x, {"m": m, "tolerance": resolve_tolerance(x, tolerance, 0.2)}


def approximate_phis(x, *, m, tolerance):
    """Return (Phi^m, Phi^(m+1)): the mean log fraction of templates within the tolerance, self included."""
    n = len(x)
    # matches of the n - m windows of m + 1 values, on their first m and on all m + 1 values
    per_window = count_matches_per_window(x, m + 1, tolerance, inclusive=True)
    count_m1 = per_window[m] + 1
    count_m = np.append(per_window[m - 1], 0) + 1
    # the last template of m values has no (m + 1)-th value, so it is not one of those windows
    last = x[n - m :]
    hits = np.max(np.abs(sliding_window_view(x[: n - 1], m) - last), axis=1) <= tolerance
    count_m[: n - m] += hits
    count_m[n - m] += np.count_nonzero(hits)
    phi_m = float(np.mean(np.log(count_m / (n - m + 1))))
    phi_m1 = float(np.mean(np.log(count_m1 / (n - m))))
    return phi_m, phi_m1


def approximate_value(phis, **_):
    """Return ApEn = Phi^m - Phi^(m+1)."""
    return phis[0] - phis[1]


def approximate_min_length(m, **_):
    """Return m + 1, the shortest series with an (m + 1)-template."""
    return m + 1


# ApEn averages logarithms per template, so it has no pooled (refined composite) form
APPROXIMATE = Measure(prepare_approximate, approximate_phis, approximate_value, approximate_min_length)


def approximate_entropy(time_series: ArrayLike, m: int = 2, tolerance: float | None = None) -> float:
    """Calculate approximate entropy ApEn(m, r).

    ApEn = Phi^m - Phi^(m+1), where Phi^k is the mean over all templates of k values of the log of the
    fraction of templates within the tolerance (Chebyshev distance d <= r). Each template matches
    itself, so ApEn is always defined; that self-match also biases it towards regularity on short
    series, which sample entropy avoids.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values, with at least m + 1 samples.
    m : int, default=2
        Embedding dimension: templates of m values are compared with templates of m + 1 values (m >= 1).
    tolerance : float, optional
        The radius r, in the units of the series. Defaults to 0.2 * std(time_series) (population std);
        for a constant series any positive tolerance gives the same result, which is used instead of 0.

    Returns
    -------
    apen : float
        Approximate entropy in nats.

    Raises
    ------
    TypeError
        If m is not an integer (booleans excluded).
    ValueError
        If m < 1, an explicitly supplied tolerance is not greater than 0, or time_series is not 1D,
        contains non-finite values or has fewer than m + 1 samples.

    References
    ----------
    .. [1] Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. Proceedings of
           the National Academy of Sciences, 88(6), 2297-2301.

    Examples
    --------
    >>> import numpy as np
    >>> round(approximate_entropy(np.sin(np.arange(300) / 4), m=2), 3)
    0.165

    """
    return APPROXIMATE(time_series, m=m, tolerance=tolerance)


register(approximate_entropy, APPROXIMATE)


def within_range_distance(largest, smallest, tolerance):
    """Return (largest - smallest) / (largest + smallest) <= tolerance, with 0 / 0 (identical templates) True."""
    total = largest + smallest
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((largest - smallest) / total <= tolerance) | (total == 0)


def range_match_blocks(x, m, tolerance):
    """Yield the template pairs within the range distance, block by block, for m and m + 1 values.

    The range distance of two templates is (max |d_k| - min |d_k|) / (max |d_k| + min |d_k|) over their
    element-wise differences d_k, computed as written so that values on the threshold compare as in the
    definition (common for integer data). Identical templates (0 / 0) match.

    Yields
    ------
    start, match_m, match_m1
        ``match_m[r, c]``: templates start + r and start + 1 + c (of m values, c >= r) match.
        ``match_m1``: the same for the leading rows and columns that have an (m + 1)-th value.

    """
    n = len(x)
    n_m, n_m1 = n - m + 1, n - m
    for start in range(0, n_m - 1, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n_m - 1)
        # largest and smallest absolute difference over the m values of each pair (row, start + 1 + c)
        diff = np.abs(x[start:stop, None] - x[None, start + 1 : n_m])
        largest, smallest = diff, diff.copy()
        for k in range(1, m):
            diff = np.abs(x[start + k : stop + k, None] - x[None, start + 1 + k : n_m + k])
            np.maximum(largest, diff, out=largest)
            np.minimum(smallest, diff, out=smallest)
        match_m = within_range_distance(largest, smallest, tolerance)
        # only pairs whose column starts after the row
        match_m &= np.arange(match_m.shape[1])[None, :] >= np.arange(stop - start)[:, None]

        # extend the pairs that both have an (m + 1)-th value
        r1, c1 = max(min(stop, n_m1) - start, 0), max(n_m1 - start - 1, 0)
        diff = np.abs(x[start + m : start + m + r1, None] - x[None, start + 1 + m : start + 1 + m + c1])
        match_m1 = match_m[:r1, :c1] & within_range_distance(
            np.maximum(largest[:r1, :c1], diff), np.minimum(smallest[:r1, :c1], diff), tolerance
        )
        yield start, match_m, match_m1


def range_pair_counts(x, m, tolerance):
    """Return the matching pairs among the first n - m templates, for m and m + 1 values (RangeEn-B)."""
    b = a = 0
    for _start, match_m, match_m1 in range_match_blocks(x, m, tolerance):
        b += int(np.count_nonzero(match_m[: match_m1.shape[0], : match_m1.shape[1]]))
        a += int(np.count_nonzero(match_m1))
    return b, a


def range_window_counts(x, m, tolerance):
    """Return how many other templates each template matches, for m values and for m + 1 values (RangeEn-A)."""
    n_m, n_m1 = len(x) - m + 1, len(x) - m
    counts_m, counts_m1 = np.zeros(n_m, dtype=np.int64), np.zeros(n_m1, dtype=np.int64)
    for start, match_m, match_m1 in range_match_blocks(x, m, tolerance):
        counts_m[start : start + match_m.shape[0]] += np.count_nonzero(match_m, axis=1)
        counts_m[start + 1 : n_m] += np.count_nonzero(match_m, axis=0)
        counts_m1[start : start + match_m1.shape[0]] += np.count_nonzero(match_m1, axis=1)
        counts_m1[start + 1 : start + 1 + match_m1.shape[1]] += np.count_nonzero(match_m1, axis=0)
    return counts_m, counts_m1


def range_entropy(
    time_series: ArrayLike,
    m: int = 2,
    tolerance: float = 0.2,
    *,
    method: Literal["sample", "approximate"] = "sample",
) -> float:
    """Calculate range entropy (RangeEn).

    Sample entropy (method="sample", RangeEn-B) or approximate entropy (method="approximate", RangeEn-A)
    with the Chebyshev distance replaced by the range distance
    (max |d_k| - min |d_k|) / (max |d_k| + min |d_k|) of the element-wise differences d_k of two
    templates. The range distance lies in [0, 1], so the tolerance needs no scaling by the standard
    deviation, and the measure is robust to changes of the signal amplitude.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    m : int, default=2
        Embedding dimension (m >= 1).
    tolerance : float, default=0.2
        Threshold r in [0, 1] for the range distance (d <= r).
    method : {"sample", "approximate"}, default="sample"
        Keyword-only. RangeEn-B (pairs of distinct templates, as in sample entropy) or RangeEn-A (self
        matches included, as in approximate entropy).

    Returns
    -------
    rangeen : float
        Range entropy in nats. For method="sample", inf when no (m + 1)-templates match and NaN when no
        templates match at all, as in sample entropy.

    Notes
    -----
    The range distance is 0 / 0 for identical templates; they are counted as matches. This gives the
    self-matches of RangeEn-A and treats repeated templates of RangeEn-B as the closest possible
    pair.

    Raises
    ------
    TypeError
        If m is not an integer (booleans excluded), or method is not a string.
    ValueError
        If m < 1, tolerance is outside [0, 1], method is unknown, or time_series is not 1D, contains
        non-finite values or has fewer than m + 2 samples.

    References
    ----------
    .. [1] Omidvarnia, A., Mesbah, M., Pedersen, M., & Jackson, G. (2018). Range entropy: A bridge
           between signal complexity and self-similarity. Entropy, 20(12), 962.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).standard_normal(500)
    >>> round(range_entropy(x, m=2, tolerance=0.2), 3)
    1.475

    """
    x = validate_series(time_series)
    check_integer("m", m, 1)
    check_fraction("tolerance", tolerance, low_open=False, high_open=False)
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    if method not in ("sample", "approximate"):
        raise ValueError("method must be 'sample' or 'approximate'.")
    if len(x) < m + 2:
        raise ValueError(f"Time series length ({len(x)}) must be at least m + 2 ({m + 2}).")

    if method == "sample":
        b, a = range_pair_counts(x, m, tolerance)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.log(b / a))
    counts_m, counts_m1 = range_window_counts(x, m, tolerance)
    phi_m = np.mean(np.log((counts_m + 1) / len(counts_m)))
    phi_m1 = np.mean(np.log((counts_m1 + 1) / len(counts_m1)))
    return float(phi_m - phi_m1)


def cosine_similarity_entropy(
    time_series: ArrayLike,
    m: int = 2,
    tolerance: float = 0.07,
    delay: int = 1,
    *,
    zero_median: bool = False,
) -> float:
    """Calculate cosine similarity entropy (CoSiEn).

    Two templates are similar when the angle between them, arccos(cosine similarity) / pi, is at most the
    tolerance. CoSiEn is the binary Shannon entropy of the fraction B of similar pairs:
    -(B log2 B + (1 - B) log2 (1 - B)). Comparing directions instead of amplitudes makes it robust to
    spikes.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    m : int, default=2
        Embedding dimension (m >= 1).
    tolerance : float, default=0.07
        Angular threshold r in (0, 1); 0.07 is the value selected by the authors.
    delay : int, default=1
        Time delay between the samples of a template.
    zero_median : bool, default=False
        Keyword-only. Subtract the median first, the authors' optional step against a baseline offset.

    Returns
    -------
    cosien : float
        Cosine similarity entropy in bits, in [0, 1]. Templates of zero norm have no angle and match
        nothing.

    Raises
    ------
    TypeError
        If m or delay is not an integer (booleans excluded).
    ValueError
        If m or delay is below 1, tolerance is outside (0, 1), or time_series is not 1D, contains
        non-finite values or has fewer than two templates.

    References
    ----------
    .. [1] Chanwimalueang, T., & Mandic, D. P. (2017). Cosine similarity entropy: Self-correlation-based
           complexity analysis of dynamical systems. Entropy, 19(12), 652.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).standard_normal(500)
    >>> round(cosine_similarity_entropy(x, m=2), 3)
    0.367

    """
    x = validate_series(time_series)
    check_integer("m", m, 1)
    check_integer("delay", delay, 1)
    check_fraction("tolerance", tolerance)
    if not isinstance(zero_median, (bool, np.bool_)):
        raise TypeError(f"zero_median must be a boolean, got {type(zero_median).__name__}.")
    check_embedding_length(x, m, delay)
    n_win = len(x) - (m - 1) * delay
    if n_win < 2:
        raise ValueError("time_series must have at least two templates.")
    if zero_median:
        x = x - np.median(x)

    coords = [x[k * delay : k * delay + n_win] for k in range(m)]
    norms = np.sqrt(sum(c * c for c in coords))
    # angle <= r  <=>  cos >= cos(pi r)  <=>  dot >= cos(pi r) * |u| |v|, for nonzero u and v
    threshold = math.cos(math.pi * tolerance)
    scaled = threshold * norms
    valid = norms > 0
    similar = 0
    for start in range(0, n_win - 1, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n_win - 1)
        dot = coords[0][start:stop, None] * coords[0][None, start + 1 :]
        for c in coords[1:]:
            dot += c[start:stop, None] * c[None, start + 1 :]
        hit = dot >= scaled[start:stop, None] * norms[None, start + 1 :]
        hit &= valid[start:stop, None] & valid[None, start + 1 :]
        hit &= np.arange(hit.shape[1])[None, :] >= np.arange(stop - start)[:, None]
        similar += int(np.count_nonzero(hit))

    b = similar / (n_win * (n_win - 1) / 2)
    if b <= 0.0 or b >= 1.0:
        return 0.0
    return float(-(b * math.log2(b) + (1 - b) * math.log2(1 - b)))


def closest_pair_distance(windows):
    """Return the smallest Chebyshev distance between two distinct rows of `windows` (at least two rows).

    Sorting by the first column bounds the search: a pair can only beat the current best distance if
    its first values differ by less than it, and those pairs are neighbours after sorting.

    """
    order = np.argsort(windows[:, 0], kind="stable")
    w = windows[order]
    # neighbours in the sorted order give a first upper bound
    best = float(np.min(np.max(np.abs(w[1:] - w[:-1]), axis=1)))
    if best == 0.0:
        return 0.0
    first = np.ascontiguousarray(w[:, 0])
    starts, stops, ends = band_blocks(first, best)
    for start, stop, end in zip(starts.tolist(), stops.tolist(), ends.tolist(), strict=True):
        if end <= start + 1:
            continue
        d = np.max(np.abs(w[start:stop, None, :] - w[None, start + 1 : end, :]), axis=2)
        d[np.arange(d.shape[1])[None, :] < np.arange(stop - start)[:, None]] = np.inf
        best = min(best, float(np.min(d)))
    return best


def histogram_bins(bins, n_pairs):
    """Return the number of histogram bins, from an integer or a rule applied to the number of distances."""
    if isinstance(bins, str):
        rules = {
            "sturges": math.ceil(math.log2(n_pairs) + 1),
            "rice": math.ceil(2 * n_pairs ** (1 / 3)),
            "sqrt": math.ceil(math.sqrt(n_pairs)),
        }
        if bins not in rules:
            raise ValueError("bins must be an integer >= 2 or 'sturges', 'rice' or 'sqrt'.")
        return max(rules[bins], 2)
    check_integer("bins", bins, 2)
    return int(bins)


def distribution_entropy(
    time_series: ArrayLike,
    m: int = 2,
    delay: int = 1,
    bins: int | Literal["sturges", "rice", "sqrt"] = 512,
    normalize: bool = True,
) -> float:
    """Calculate distribution entropy (DistEn).

    The Shannon entropy of the histogram of the Chebyshev distances between all pairs of distinct
    templates. The histogram has equal-width bins over [smallest, largest] distance. Using the whole
    distance distribution instead of a tolerance keeps DistEn stable on very short series.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    m : int, default=2
        Embedding dimension (m >= 1).
    delay : int, default=1
        Time delay between the samples of a template.
    bins : int or {"sturges", "rice", "sqrt"}, default=512
        Number of bins (>= 2), or a rule applied to the number of distances. 512 is the choice of the
        original paper.
    normalize : bool, default=True
        Divide by log2(bins), as in the original paper, so the result is between 0 and 1.

    Returns
    -------
    distEn : float
        Distribution entropy (normalized, or in bits). 0.0 when all distances are equal.

    Raises
    ------
    TypeError
        If m, delay or bins is not an integer (booleans excluded) or one of the rule names.
    ValueError
        If m or delay is below 1, bins is below 2 or an unknown rule, or time_series is not 1D, contains
        non-finite values or has fewer than two templates.

    References
    ----------
    .. [1] Li, P., Liu, C., Li, K., Zheng, D., Liu, C., & Hou, Y. (2015). Assessing the complexity of
           short-term heartbeat interval series by distribution entropy. Medical & Biological
           Engineering & Computing, 53(1), 77-87.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).standard_normal(300)
    >>> round(distribution_entropy(x, m=2), 3)
    0.902

    """
    x = validate_series(time_series)
    check_integer("m", m, 1)
    check_integer("delay", delay, 1)
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError(f"normalize must be a boolean, got {type(normalize).__name__}.")
    check_embedding_length(x, m, delay)
    n_win = len(x) - (m - 1) * delay
    if n_win < 2:
        raise ValueError("time_series must have at least two templates.")
    n_pairs = n_win * (n_win - 1) // 2
    n_bins = histogram_bins(bins, n_pairs)

    windows = np.stack([x[k * delay : k * delay + n_win] for k in range(m)], axis=1)
    # the largest distance is the largest range of one coordinate; the smallest needs a search
    largest = float(np.max(np.ptp(windows, axis=0)))
    smallest = closest_pair_distance(windows)
    if largest == smallest:
        return 0.0

    counts = np.zeros(n_bins, dtype=np.int64)
    for start in range(0, n_win - 1, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n_win - 1)
        d = np.abs(windows[start:stop, None, 0] - windows[None, start + 1 :, 0])
        for k in range(1, m):
            np.maximum(d, np.abs(windows[start:stop, None, k] - windows[None, start + 1 :, k]), out=d)
        later = np.arange(d.shape[1])[None, :] >= np.arange(stop - start)[:, None]
        # np.histogram with a fixed range bins every chunk exactly as it would bin all distances at once
        counts += np.histogram(d[later], bins=n_bins, range=(smallest, largest))[0]

    p = counts[counts > 0] / n_pairs
    h = float(-np.dot(p, np.log2(p)))
    if normalize:
        h /= math.log2(n_bins)
    return h if h > 0.0 else 0.0
