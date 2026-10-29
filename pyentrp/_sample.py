"""Sample entropy and its multiscale variants (MSE, CMSE, RCMSE)."""

import warnings

import numpy as np

from ._core import _granulate
from ._validation import _check_integer, _resolve_tolerance, _validate_series


def _count_template_matches(time_series, sample_length, tolerance, chunk_size=32):
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


def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate the sample entropy of degree m of a time_series.

    This method uses Chebyshev norm.
    It is quite fast for random data but can be slower is there is
    structure in the input time series.

    Parameters
    ----------
    time_series : np.ndarray
        Time series, 1-d vector
    sample_length : int
        length of longest template vector
    tolerance : float, optional
        Strict upper bound on the Chebyshev distance between templates,
        in the same units as the time series. Defaults to 0.1 * std(time_series);
        for a constant series any positive tolerance gives the same result, which
        is used instead of 0. The multiscale functions share this default, while
        fuzzy_entropy uses 0.2 * std. Pass tolerance=0.2 * np.std(x) for the
        common r = 0.2 SD.

    Returns
    -------
    sampen: np.ndarray
        Array of length sample_length in nats. SE[k] is -log(N[k+1] / N[k]),
        where N[k] counts matching pairs of length-k prefixes of full
        sample_length windows, excluding self-matches, and
        N[0] = n * (n - 1) / 2 for a series of length n. A zero numerator
        with a positive denominator yields inf; two zero counts yield NaN.

    Raises
    ------
    TypeError
        If sample_length is not an integer (booleans excluded).
    ValueError
        If sample_length < 1, time_series is not 1D or contains non-finite
        values, or an explicitly supplied tolerance is not greater than 0.

    Warns
    -----
    RuntimeWarning
        For zero match counts, under NumPy's default error settings.

    Notes
    -----
    The parameter 'sample_length' is equal to m + 1 in Ref [1]. Counting follows the
    PhysioNet sampen implementation [2]: the strict test d < tolerance and N[0] = n(n - 1) / 2.

    References
    ----------
    .. [1] Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using
           approximate entropy and sample entropy. American Journal of Physiology-Heart and
           Circulatory Physiology, 278(6), H2039-H2049.
    .. [2] Sample Entropy Estimation (sampen) software, PhysioNet.
           https://physionet.org/content/sampen/1.0.0/

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)

    tolerance = _resolve_tolerance(time_series, tolerance, 0.1)

    return _sampen(time_series, sample_length, tolerance)


def _sampen(x, sample_length, tolerance):
    """Return SE[k] = log(N[k] / N[k + 1]) for k = 0 .. sample_length - 1 (written so it never gives -0.0)."""
    counts = _count_template_matches(x, sample_length, tolerance)
    return np.log(counts[:-1] / counts[1:])


def _sampen_last(x, sample_length, tolerance):
    """Return only SE[sample_length - 1], the sample entropy for m = sample_length - 1."""
    counts = _count_template_matches(x, sample_length, tolerance)
    return np.log(counts[-2] / counts[-1])


def multiscale_entropy(time_series, sample_length, tolerance=None, maxscale=20):
    """Calculate Multiscale Entropy considering different time-scales of the time series.

    Parameters
    ----------
    time_series : np.ndarray
        Input time series for analysis.
    sample_length : int
        Bandwidth or group of points
    tolerance : float
        Tolerance value (default is 0.1 times the standard deviation of the `time_series`;
        a constant series is treated as in `sample_entropy`)
    maxscale : int, default=20
        Maximum scale, inclusive. The default is the range of scales used by
        Costa et al. (2002). None is accepted as an alias for 20. A scale whose
        coarse-grained series is too short for matching pairs gives NaN or inf,
        with a RuntimeWarning.

    Returns
    -------
    mse : np.ndarray
        Array of Multiscale Entropies

    Raises
    ------
    TypeError
        If sample_length or maxscale is not an integer (booleans excluded).
    ValueError
        If sample_length or maxscale is below 1, time_series is not 1D or
        contains non-finite values, or an explicitly supplied tolerance is
        not greater than 0.

    References
    ----------
    .. [1] Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of
           complex physiologic time series. Physical Review Letters, 89(6), 068102.

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)

    tolerance = _resolve_tolerance(time_series, tolerance, 0.1)

    if maxscale is None:
        maxscale = 20
    _check_integer("maxscale", maxscale, 1)

    return np.array(
        [_sampen_last(_granulate(time_series, k), sample_length, tolerance) for k in range(1, maxscale + 1)]
    )


def composite_multiscale_entropy(time_series, sample_length, scale, tolerance=None):
    """Calculate Composite Multiscale Entropy.

    Parameters
    ----------
    time_series : np.ndarray
        Time series for analysis
    sample_length : int
        Number of sequential points of the time series
    scale : int
        Scale factor
    tolerance : float
        Tolerance (default = 0.1 * std(time_series); a constant series is treated
        as in `sample_entropy`)

    Returns
    -------
    cmse : np.ndarray
        Array of Composite Multiscale Entropies

    Raises
    ------
    TypeError
        If sample_length or scale is not an integer (booleans excluded).
    ValueError
        If sample_length or scale is below 1, time_series is not 1D or
        contains non-finite values, or an explicitly supplied tolerance is
        not greater than 0.

    References
    ----------
    .. [1] Wu, Shuen-De, et al. "Time series analysis using
        composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)
    _check_integer("scale", scale, 1)

    tolerance = _resolve_tolerance(time_series, tolerance, 0.1)

    cmse = np.zeros(scale)
    for i in range(scale):
        k = i + 1
        for j in range(k):
            cmse[i] += _sampen_last(_granulate(time_series[j:], k), sample_length, tolerance) / k
    return cmse


def rcmse(time_series, sample_length, scale, tolerance=None):
    """Calculate Refined Composite Multiscale Entropy (RCMSE).

    At each scale, pool template matches across all shifted series formed by
    averaging non-overlapping blocks, discarding incomplete trailing blocks.

    Parameters
    ----------
    time_series : np.ndarray | list
        Time series for analysis, 1-d vector of finite values.
    sample_length : int
        Longest template length. Compare matches of this length with matches
        of length sample_length - 1, as in sample_entropy.
    scale : int
        Maximum scale, inclusive. Each scale is a block length in samples.
    tolerance : float, optional
        Strict upper bound on the Chebyshev distance between templates,
        in the same units as the time series. Defaults to 0.1 * std(time_series)
        from the original series and remains fixed across scales. For a constant
        series any positive tolerance gives the same result, which is used instead of 0.

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
        If scale or sample_length is not an integer (booleans excluded).
    ValueError
        If scale < 1, sample_length < 1, time_series is not 1D or contains
        non-finite values, time_series is shorter than sample_length + 1,
        or an explicitly supplied tolerance is not greater than 0.

    Warns
    -----
    RuntimeWarning
        When either pooled match count is zero at a scale; that entry is NaN.

    References
    ----------
    .. [1] Wu, S. D., et al. "Refined composite multiscale entropy: a novel measure
           for time series complexity analysis." IEEE Signal Processing Letters (2014).

    """
    time_series = _validate_series(time_series)
    _check_integer("scale", scale, 1)
    _check_integer("sample_length", sample_length, 1)

    if len(time_series) < sample_length + 1:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least sample_length + 1 ({sample_length + 1})."
        )

    tolerance = _resolve_tolerance(time_series, tolerance, 0.1)

    rcmse_values = np.zeros(scale)

    for i in range(scale):
        total_matches = np.zeros(sample_length + 1)
        for j in range(i + 1):
            total_matches += _count_template_matches(_granulate(time_series[j:], i + 1), sample_length, tolerance)

        n_m_plus_1 = total_matches[sample_length]
        n_m = total_matches[sample_length - 1]

        if n_m_plus_1 == 0 or n_m == 0:
            warnings.warn(
                f"No template matches found at scale {i + 1}. Returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            rcmse_values[i] = np.nan
        else:
            rcmse_values[i] = np.log(n_m / n_m_plus_1)

    return rcmse_values


# Descriptive alias; `rcmse` remains the primary name.
refined_composite_multiscale_entropy = rcmse
