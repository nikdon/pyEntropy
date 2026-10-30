"""Sample entropy and its multiscale variants (MSE, CMSE, RCMSE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._multiscale import (
    Measure,
    composite_multiscale,
    multiscale,
    refined_composite_multiscale,
    register,
    sum_stats,
)
from ._validation import renamed, resolve_tolerance, template_length, validate_series

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# the sorted band search pays off from about this many windows (its setup costs more on short series)
BANDED_MIN_WINDOWS = 512
# and below this estimated cost (band fraction * template length, relative to comparing all pairs);
# the measured break-even is about 1
BANDED_MAX_COST = 0.75


def count_template_matches(time_series, sample_length, tolerance):
    """Count template matches of lengths 0 to sample_length using Chebyshev norm.

    Two exact strategies give identical counts. The banded one sorts the windows by their first
    value: only windows within the tolerance in that value can match, and after sorting they are
    neighbours, so only a band of pairs is compared. It is used when that band is small (typical
    tolerances such as 0.1-0.2 std) and the series is long enough; comparing all pairs is used
    otherwise.

    Parameters
    ----------
    time_series : np.ndarray
        Time series, 1-d vector.
    sample_length : int
        Length of longest template vector.
    tolerance : float
        Strict upper bound on the Chebyshev distance between templates,
        in the same units as the time series.

    Returns
    -------
    N_temp : np.ndarray
        Array of length sample_length + 1, where N_temp[0] = n * (n - 1) / 2
        and n is the series length. For k >= 1, N_temp[k] counts matching
        unordered pairs of length-k prefixes of full sample_length windows,
        excluding self-matches. These counts are zero if fewer than two full
        windows are available.

    """
    # Floating point input avoids wrap-around in the differences of unsigned integer series
    x = np.asarray(time_series, dtype=float)

    n = len(x)
    n_win = n - sample_length + 1  # number of full-length windows

    counts = np.zeros(sample_length + 1)
    counts[0] = n * (n - 1) / 2
    if n_win <= 1:
        return counts
    if n_win < BANDED_MIN_WINDOWS:
        count_matches_dense(x, sample_length, tolerance, counts)
        return counts

    windows = sliding_window_view(x, sample_length)[:n_win]
    order = np.argsort(windows[:, 0], kind="stable")
    sorted_windows = [np.ascontiguousarray(windows[order, k]) for k in range(sample_length)]
    blocks = band_blocks(sorted_windows[0], tolerance)

    starts, stops, ends = blocks
    band_pairs = float(np.sum((stops - starts) * (ends - starts - 1)))
    if band_pairs * sample_length < BANDED_MAX_COST * n_win * (n_win - 1) / 2:
        count_matches_banded(sorted_windows, tolerance, blocks, counts)
    else:
        count_matches_dense(x, sample_length, tolerance, counts)
    return counts


def band_blocks(first, tolerance, block_size=64):
    """Split sorted first values into blocks of rows and the end of the column band of each block.

    Row p can only match columns p + 1 .. end - 1, where end is the first value more than the
    tolerance above row p. The block's end is the one of its last row, which covers all its rows.

    Returns
    -------
    starts, stops, ends : np.ndarray
        Rows start .. stop - 1 of a block are compared with columns start + 1 .. end - 1.

    """
    n_win = len(first)
    # a superset of the band: the slack covers rounding in first + tolerance, the exact test decides
    slack = 4 * np.finfo(float).eps * (np.max(np.abs(first)) + tolerance)
    band_end = np.searchsorted(first, first + (tolerance + slack), side="right")
    starts = np.arange(0, n_win - 1, block_size)
    stops = np.minimum(starts + block_size, n_win - 1)
    ends = np.maximum(band_end[stops - 1], starts + 1)
    return starts, stops, ends


def count_matches_banded(sorted_windows, tolerance, blocks, counts):
    """Add the template matches of windows sorted by their first value, comparing only the band."""
    starts, stops, ends = blocks
    rows_max = int(np.max(stops - starts))
    width_max = int(np.max(ends - starts - 1))
    if width_max <= 0:
        return

    size = rows_max * width_max
    diff_buf, hit_buf, cur_buf = np.empty(size), np.empty(size, dtype=np.bool_), np.empty(size, dtype=np.bool_)
    # triu[r, c] is True when c >= r, i.e. when column window start + 1 + c comes after row window start + r
    triu = np.triu(np.ones((rows_max, rows_max), dtype=np.bool_))

    for start, stop, end in zip(starts.tolist(), stops.tolist(), ends.tolist(), strict=True):
        rows, width = stop - start, end - start - 1
        if width <= 0:
            continue
        diff = diff_buf[: rows * width].reshape(rows, width)
        hit = hit_buf[: rows * width].reshape(rows, width)
        cur = cur_buf[: rows * width].reshape(rows, width)

        # cur[r, c]: sorted windows start + r and start + 1 + c match on every value compared so far
        for k, column in enumerate(sorted_windows):
            np.subtract(column[start:stop, None], column[None, start + 1 : end], out=diff)
            np.abs(diff, out=diff)
            if k == 0:
                np.less(diff, tolerance, out=cur)
                sq = min(rows - 1, width)
                if sq > 0:
                    cur[:, :sq] &= triu[:rows, :sq]  # each unordered pair once, no self-matches
            else:
                np.less(diff, tolerance, out=hit)
                cur &= hit
            c = np.count_nonzero(cur)
            if c == 0:
                break
            counts[k + 1] += c


def count_matches_dense(x, sample_length, tolerance, counts, chunk_size=32):
    """Add the template matches of all pairs of windows, comparing every pair once."""
    n_win = len(x) - sample_length + 1

    # triu_full[r, c] is True when c >= r, i.e. when the second window starts after the first
    triu_full = np.triu(np.ones((chunk_size + sample_length - 1,) * 2, dtype=np.bool_))

    # windows start .. end - 1 (rows) are compared with all later windows (columns)
    for start in range(0, n_win - 1, chunk_size):
        end = min(start + chunk_size, n_win - 1)
        b_len = end - start
        w_len = n_win - start - 1

        # B[r, c] is True when |x[start + r] - x[start + 1 + c]| < tolerance. Entry (r + t, c + t)
        # is then the match of the pair (start + r, start + 1 + c) at offset t inside the templates
        diff = np.subtract(x[start : end + sample_length - 1, None], x[None, start + 1 :])
        np.abs(diff, out=diff)
        B = diff < tolerance

        # keep only pairs whose second window starts after the first (no self- or double-matches)
        sq = min(b_len + sample_length - 2, B.shape[1])
        if sq > 0:
            B[:, :sq] &= triu_full[: B.shape[0], :sq]

        cur = B[:b_len, :w_len].copy()
        counts[1] += np.count_nonzero(cur)

        # extend the matches one sample at a time; `cur` can only shrink
        for length in range(2, sample_length + 1):
            s = length - 1
            cur &= B[s : b_len + s, s : w_len + s]
            c = np.count_nonzero(cur)
            if c == 0:
                break
            counts[length] += c


def prepare_sample(time_series, sample_length=None, tolerance=None, *, m=None):
    """Validate the input of sample entropy and resolve the default tolerance on this series."""
    length = template_length(sample_length, m)
    x = validate_series(time_series)
    return x, {"sample_length": length, "tolerance": resolve_tolerance(x, tolerance, 0.1)}


def sample_value(counts, **_):
    """Return SampEn(m, r) = log(N[m] / N[m + 1]) from the match counts."""
    return np.log(counts[-2] / counts[-1])


def sample_min_length(sample_length, **_):
    """Return m + 2, the shortest series with a pair of (m + 1)-templates."""
    return sample_length + 1


SAMPLE = Measure(prepare_sample, count_template_matches, sample_value, sample_min_length, sum_stats)


def sample_entropy(
    time_series: ArrayLike, sample_length: int | None = None, tolerance: float | None = None, *, m: int | None = None
) -> NDArray[np.float64]:
    """Calculate the sample entropy SampEn(m, r) of a time series, using the Chebyshev distance.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    sample_length : int, optional
        Deprecated; use ``m = sample_length - 1``. Longest template length, m + 1.
    tolerance : float, optional
        The radius r: strict upper bound on the Chebyshev distance between templates,
        in the units of the series. Defaults to 0.1 * std(time_series) (population std);
        for a constant series any positive tolerance gives the same result, which is
        used instead of 0. Pass tolerance=0.2 * np.std(x) for the common r = 0.2 SD.
    m : int
        Keyword-only. Embedding dimension: templates of length m are compared with
        templates of length m + 1 (m >= 0). Required unless sample_length is given.

    Returns
    -------
    sampen : np.ndarray
        Array of length m + 1 in nats; the last entry is SampEn(m, r). SE[k] is
        -log(N[k+1] / N[k]), where N[k] counts matching pairs of length-k prefixes
        of full (m + 1)-sample windows, excluding self-matches, and
        N[0] = n * (n - 1) / 2 for a series of length n. A zero numerator
        with a positive denominator yields inf; two zero counts yield NaN.

    Raises
    ------
    TypeError
        If m is not an integer (booleans excluded), or both or neither of m and
        sample_length are given.
    ValueError
        If m < 0, time_series is not 1D or contains non-finite values, or an
        explicitly supplied tolerance is not greater than 0.

    Warns
    -----
    RuntimeWarning
        For zero match counts, under NumPy's default error settings.

    Notes
    -----
    Counting follows the
    PhysioNet sampen implementation [2]: the strict test d < tolerance and N[0] = n(n - 1) / 2.

    References
    ----------
    .. [1] Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using
           approximate entropy and sample entropy. American Journal of Physiology-Heart and
           Circulatory Physiology, 278(6), H2039-H2049.
    .. [2] Sample Entropy Estimation (sampen) software, PhysioNet.
           https://physionet.org/content/sampen/1.0.0/

    """
    x, p = prepare_sample(time_series, sample_length, tolerance, m=m)
    counts = count_template_matches(x, **p)
    # SE[k] for k = 0 .. m, written so it never gives -0.0
    return np.log(counts[:-1] / counts[1:])


register(sample_entropy, SAMPLE)


@renamed(maxscale="scale")
def multiscale_entropy(
    time_series: ArrayLike,
    sample_length: int | None = None,
    tolerance: float | None = None,
    scale: int | None = 20,
    *,
    m: int | None = None,
) -> NDArray[np.float64]:
    """Calculate Multiscale Entropy: SampEn(m, r) of the coarse-grained series at scales 1 .. scale.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    sample_length : int, optional
        Deprecated; use ``m = sample_length - 1``. Longest template length, m + 1.
    tolerance : float, optional
        Strict upper bound on the Chebyshev distance between templates, in the
        units of the series. Defaults to 0.1 * std(time_series) (population std),
        computed on the original series and fixed across scales; for a constant
        series any positive tolerance gives the same result, which is used instead of 0.
    scale : int, default=20
        Maximum scale, inclusive (formerly ``maxscale``). The default is the range of
        scales used by Costa et al. (2002). None is accepted as an alias for 20. A
        scale whose coarse-grained series is too short for matching pairs gives NaN
        or inf, with a RuntimeWarning.
    m : int
        Keyword-only. Embedding dimension: templates of length m are compared with
        templates of length m + 1 (m >= 0). Required unless sample_length is given.

    Returns
    -------
    mse : np.ndarray
        Array of length scale in nats; index i corresponds to scale i + 1.

    Raises
    ------
    TypeError
        If m or scale is not an integer (booleans excluded), or both or neither
        of m and sample_length are given.
    ValueError
        If m < 0, scale < 1, time_series is not 1D or contains non-finite values,
        or an explicitly supplied tolerance is not greater than 0.

    References
    ----------
    .. [1] Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of
           complex physiologic time series. Physical Review Letters, 89(6), 068102.

    """
    if scale is None:
        scale = 20
    return multiscale(time_series, sample_entropy, scale, sample_length=sample_length, tolerance=tolerance, m=m)


def composite_multiscale_entropy(
    time_series: ArrayLike,
    sample_length: int | None = None,
    scale: int = 20,
    tolerance: float | None = None,
    *,
    m: int | None = None,
) -> NDArray[np.float64]:
    """Calculate Composite Multiscale Entropy: at each scale, average SampEn(m, r) over all shifted coarse series.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    sample_length : int, optional
        Deprecated; use ``m = sample_length - 1``. Longest template length, m + 1.
    scale : int, default=20
        Maximum scale, inclusive.
    tolerance : float, optional
        Strict upper bound on the Chebyshev distance between templates, in the
        units of the series. Defaults to 0.1 * std(time_series) (population std),
        computed on the original series and fixed across scales; for a constant
        series any positive tolerance gives the same result, which is used instead of 0.
    m : int
        Keyword-only. Embedding dimension: templates of length m are compared with
        templates of length m + 1 (m >= 0). Required unless sample_length is given.

    Returns
    -------
    cmse : np.ndarray
        Array of length scale in nats; index i corresponds to scale i + 1.

    Raises
    ------
    TypeError
        If m or scale is not an integer (booleans excluded), or both or neither
        of m and sample_length are given.
    ValueError
        If m < 0, scale < 1, time_series is not 1D or contains non-finite values,
        or an explicitly supplied tolerance is not greater than 0.

    References
    ----------
    .. [1] Wu, Shuen-De, et al. "Time series analysis using
        composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.

    """
    return composite_multiscale(
        time_series, sample_entropy, scale, sample_length=sample_length, tolerance=tolerance, m=m
    )


def rcmse(
    time_series: ArrayLike,
    sample_length: int | None = None,
    scale: int = 20,
    tolerance: float | None = None,
    *,
    m: int | None = None,
) -> NDArray[np.float64]:
    """Calculate Refined Composite Multiscale Entropy (RCMSE).

    At each scale, pool template matches across all shifted series formed by
    averaging non-overlapping blocks, discarding incomplete trailing blocks.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    sample_length : int, optional
        Deprecated; use ``m = sample_length - 1``. Longest template length, m + 1.
    scale : int, default=20
        Maximum scale, inclusive. Each scale is a block length in samples.
    tolerance : float, optional
        Strict upper bound on the Chebyshev distance between templates, in the
        units of the series. Defaults to 0.1 * std(time_series) (population std),
        computed on the original series and fixed across scales; for a constant
        series any positive tolerance gives the same result, which is used instead of 0.
    m : int
        Keyword-only. Embedding dimension: templates of length m are compared with
        templates of length m + 1 (m >= 0). Required unless sample_length is given.

    Returns
    -------
    rcmse : np.ndarray
        Array of length scale in nats; index i corresponds to scale i + 1.
        As in sample_entropy, a zero pooled count for length m + 1 gives inf
        (with a positive count for length m) and two zero counts give NaN. An
        entry is NaN when every coarse-grained series is shorter than m + 2.

    Raises
    ------
    TypeError
        If m or scale is not an integer (booleans excluded), or both or neither
        of m and sample_length are given.
    ValueError
        If m < 0, scale < 1, time_series is not 1D or contains non-finite values,
        time_series is shorter than m + 2, or an explicitly supplied tolerance
        is not greater than 0.

    Warns
    -----
    RuntimeWarning
        For zero pooled match counts (under NumPy's default error settings), and
        when the coarse-grained series are too short from some scale on.

    References
    ----------
    .. [1] Wu, S. D., et al. "Refined composite multiscale entropy: a novel measure
           for time series complexity analysis." IEEE Signal Processing Letters (2014).

    """
    return refined_composite_multiscale(
        time_series, sample_entropy, scale, sample_length=sample_length, tolerance=tolerance, m=m
    )


# Descriptive alias; `rcmse` remains the primary name.
refined_composite_multiscale_entropy = rcmse
