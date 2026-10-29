"""Sample entropy and its multiscale variants (MSE, CMSE, RCMSE)."""

import numpy as np

from ._multiscale import Measure, coarse_grain, register, sum_stats
from ._validation import check_integer, renamed, resolve_tolerance, template_length, validate_series, warn


def count_template_matches(time_series, sample_length, tolerance, chunk_size=32):
    """Count template matches of lengths 0 to sample_length using Chebyshev norm.

    Parameters
    ----------
    time_series : np.ndarray
        Time series, 1-d vector.
    sample_length : int
        Length of longest template vector.
    tolerance : float
        Strict upper bound on the Chebyshev distance between templates,
        in the same units as the time series.
    chunk_size : int, optional
        Number of templates compared against all later templates at once.
        Larger chunks mean fewer Python-level iterations but a larger
        temporary (about chunk_size * n * 8 bytes). Does not change the result.

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
    time_series = np.asarray(time_series, dtype=float)

    n = len(time_series)
    n_win = n - sample_length + 1  # number of full-length windows

    N_temp = np.zeros(sample_length + 1)
    N_temp[0] = n * (n - 1) / 2
    if n_win <= 1:
        return N_temp

    # triu_full[r, c] is True when c >= r, i.e. when the second window starts after the first
    triu_full = np.triu(np.ones((chunk_size + sample_length - 1,) * 2, dtype=np.bool_))

    # windows start .. end - 1 (rows) are compared with all later windows (columns)
    for start in range(0, n_win - 1, chunk_size):
        end = min(start + chunk_size, n_win - 1)
        b_len = end - start
        w_len = n_win - start - 1

        # B[r, c] is True when |x[start + r] - x[start + 1 + c]| < tolerance. Entry (r + t, c + t)
        # is then the match of the pair (start + r, start + 1 + c) at offset t inside the templates
        diff = np.subtract(time_series[start : end + sample_length - 1, None], time_series[None, start + 1 :])
        np.abs(diff, out=diff)
        B = diff < tolerance

        # keep only pairs whose second window starts after the first (no self- or double-matches)
        sq = min(b_len + sample_length - 2, B.shape[1])
        if sq > 0:
            B[:, :sq] &= triu_full[: B.shape[0], :sq]

        cur = B[:b_len, :w_len].copy()
        N_temp[1] += np.count_nonzero(cur)

        # extend the matches one sample at a time; `cur` can only shrink
        for length in range(2, sample_length + 1):
            s = length - 1
            cur &= B[s : b_len + s, s : w_len + s]
            c = np.count_nonzero(cur)
            if c == 0:
                break
            N_temp[length] += c

    return N_temp


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


def sample_entropy(time_series, sample_length=None, tolerance=None, *, m=None):
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
def multiscale_entropy(time_series, sample_length=None, tolerance=None, scale=20, *, m=None):
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
    length = template_length(sample_length, m)
    time_series = validate_series(time_series)
    tolerance = resolve_tolerance(time_series, tolerance, 0.1)
    if scale is None:
        scale = 20
    check_integer("scale", scale, 1)

    return np.array(
        [
            sample_value(count_template_matches(coarse_grain(time_series, k), length, tolerance))
            for k in range(1, scale + 1)
        ]
    )


def composite_multiscale_entropy(time_series, sample_length=None, scale=20, tolerance=None, *, m=None):
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
    length = template_length(sample_length, m)
    time_series = validate_series(time_series)
    check_integer("scale", scale, 1)
    tolerance = resolve_tolerance(time_series, tolerance, 0.1)

    cmse = np.zeros(scale)
    for i in range(scale):
        k = i + 1
        for j in range(k):
            cmse[i] += sample_value(count_template_matches(coarse_grain(time_series[j:], k), length, tolerance)) / k
    return cmse


def rcmse(time_series, sample_length=None, scale=20, tolerance=None, *, m=None):
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
        An entry is NaN if either pooled match count is zero, including when
        the coarse-grained series are too short to provide matching pairs.
        (Unlike sample_entropy, a zero numerator gives NaN rather than inf.)

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
        When either pooled match count is zero at a scale; that entry is NaN.

    References
    ----------
    .. [1] Wu, S. D., et al. "Refined composite multiscale entropy: a novel measure
           for time series complexity analysis." IEEE Signal Processing Letters (2014).

    """
    length = template_length(sample_length, m)
    time_series = validate_series(time_series)
    check_integer("scale", scale, 1)

    if len(time_series) < length + 1:
        raise ValueError(f"Time series length ({len(time_series)}) must be at least m + 2 ({length + 1}).")

    tolerance = resolve_tolerance(time_series, tolerance, 0.1)

    rcmse_values = np.zeros(scale)

    for i in range(scale):
        total_matches = np.zeros(length + 1)
        for j in range(i + 1):
            total_matches += count_template_matches(coarse_grain(time_series[j:], i + 1), length, tolerance)

        n_m_plus_1 = total_matches[length]
        n_m = total_matches[length - 1]

        if n_m_plus_1 == 0 or n_m == 0:
            warn(
                f"No template matches found at scale {i + 1}. Returning NaN.",
                RuntimeWarning,
            )
            rcmse_values[i] = np.nan
        else:
            rcmse_values[i] = np.log(n_m / n_m_plus_1)

    return rcmse_values


# Descriptive alias; `rcmse` remains the primary name.
refined_composite_multiscale_entropy = rcmse
