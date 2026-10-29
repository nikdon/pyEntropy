"""Fuzzy entropy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._validation import _check_integer, _renamed, _resolve_tolerance, _validate_series, _warn


def _prepare_centered_T(x, order, n_vectors, inv_tol):
    """Prepare pre-scaled, mean-centered, transposed embedding vectors for fuzzy entropy.

    Parameters
    ----------
    x : np.ndarray
        1D float time series.
    order : int
        Embedding dimension (window length).
    n_vectors : int
        Number of (leading) windows to use.
    inv_tol : float
        1 / tolerance. The vectors are scaled by it, so distances are already in units of
        the tolerance.

    Returns
    -------
    cent_T : np.ndarray or None
        Array of shape (m_eff, n_vectors) with one row per coordinate, or None for
        order == 1 (all centered windows are zero vectors).

    """
    if order == 1:
        return None

    x = x[: n_vectors + order - 1]
    if order == 2:
        # a centered 2-window is ((a - b) / 2, (b - a) / 2): one coordinate carries all the distance
        c0 = (x[:-1] - x[1:]) * (0.5 * inv_tol)
        return c0[None, :]

    emb = sliding_window_view(x, order)
    cent = (emb - np.mean(emb, axis=1, keepdims=True)) * inv_tol
    return np.ascontiguousarray(cent.T)


def _compute_phi(cent_T, exponent, chunk_size=32):
    """Compute the mean fuzzy similarity over all unordered pairs of distinct vectors.

    Parameters
    ----------
    cent_T : np.ndarray or None
        Output of `_prepare_centered_T`.
    exponent : int or float
        Power of the exponential membership function, exp(-d ** exponent).
    chunk_size : int, optional
        Number of rows processed at once. Does not change the result.

    Returns
    -------
    phi : float
        Mean of exp(-d ** exponent) over pairs, with d the Chebyshev distance of the
        (already tolerance-scaled) centered vectors.

    """
    if cent_T is None:
        return 1.0

    m_eff, n_vec = cent_T.shape
    n_pairs = n_vec * (n_vec - 1) // 2
    total = 0.0

    # triu_mask[r, c] is True when c >= r, i.e. when the column vector comes after the row vector
    triu_mask = np.triu(np.ones((chunk_size, chunk_size - 1), dtype=np.bool_))
    flat_dist = np.empty(chunk_size * (n_vec - 1))
    flat_tmp = np.empty(chunk_size * (n_vec - 1)) if m_eff > 1 else None

    for start in range(0, n_vec - 1, chunk_size):
        end = min(start + chunk_size, n_vec - 1)
        b_len = end - start
        w_len = n_vec - start - 1
        size = b_len * w_len

        dist_1d = flat_dist[:size]
        dist = dist_1d.reshape(b_len, w_len)
        np.subtract(cent_T[0, start:end, None], cent_T[0, start + 1 :], out=dist)

        if m_eff > 1:
            # chebyshev distance: running maximum of the absolute coordinate differences
            np.abs(dist_1d, out=dist_1d)
            tmp_1d = flat_tmp[:size]
            tmp = tmp_1d.reshape(b_len, w_len)
            for k in range(1, m_eff):
                np.subtract(cent_T[k, start:end, None], cent_T[k, start + 1 :], out=tmp)
                np.abs(tmp_1d, out=tmp_1d)
                np.maximum(dist_1d, tmp_1d, out=dist_1d)
        elif exponent != 2:
            np.abs(dist_1d, out=dist_1d)

        if exponent == 2:
            np.square(dist_1d, out=dist_1d)
        elif exponent != 1:
            np.power(dist_1d, exponent, out=dist_1d)

        np.negative(dist_1d, out=dist_1d)
        np.exp(dist_1d, out=dist_1d)

        # zero out pairs (row, column) with column vector <= row vector (self- and double-counts)
        if b_len > 1:
            dist[:, : b_len - 1] *= triu_mask[:b_len, : b_len - 1]

        total += float(np.sum(dist_1d))

    return total / n_pairs


@_renamed(sample_length="m", n="exponent")
def fuzzy_entropy(time_series, m=2, tolerance=None, exponent=2):
    """Calculate Fuzzy Entropy (FuzzyEn) using exponential membership function.

    Fuzzy entropy characterizes the regularity of a time series by replacing
    the rigid Heaviside step boundary of sample entropy with a continuous
    exponential membership function and centering vectors around their local means.

    Parameters
    ----------
    time_series : np.ndarray or list
        Input time series for analysis, 1-d vector of finite values.
    m : int, default=2
        Embedding dimension: templates of length m are compared with templates of
        length m + 1 (formerly ``sample_length``, with the same meaning).
    tolerance : float, optional
        Positive radius r in the same units as the time series, used in
        exp(-(d / r)**exponent), where d is the Chebyshev distance between centered
        templates. This is the form of Chen et al. (2007), without the ln(2)
        scaling used in some later papers. Defaults to 0.2 * std(time_series),
        unlike sample_entropy and the multiscale functions, which default to
        0.1 * std(time_series).
    exponent : int or float, default=2
        Weight of the fuzzy boundary, the power in the membership function
        (formerly ``n``).

    Returns
    -------
    fuzzyen : float
        Negative natural logarithm of the ratio of mean similarities for
        lengths m + 1 and m, in nats. Self-matches
        are excluded. Returns 0.0 for a constant series (with the default
        tolerance) and NaN if either mean similarity vanishes to zero.

    Raises
    ------
    TypeError
        If m is not an integer (booleans excluded).
    ValueError
        If m < 1, exponent <= 0, an explicitly supplied tolerance is not
        greater than 0, time_series is not 1D or contains non-finite values,
        or time_series is shorter than m + 2.

    Warns
    -----
    RuntimeWarning
        When either mean similarity vanishes to zero; the result is NaN.

    References
    ----------
    .. [1] Chen, W., et al. "Characterization of surface EMG signal based on fuzzy entropy."
           IEEE Transactions on Neural Systems and Rehabilitation Engineering 15.2 (2007): 266-272.

    """
    time_series = _validate_series(time_series)
    _check_integer("m", m, 1)

    if not exponent > 0:
        raise ValueError("exponent must be greater than 0.")

    if len(time_series) < m + 2:
        raise ValueError(f"Time series length ({len(time_series)}) must be at least m + 2 ({m + 2}).")

    tolerance = _resolve_tolerance(time_series, tolerance, 0.2)

    # both lengths use the same number of vectors (the first n_vectors windows)
    n_vectors = len(time_series) - m
    inv_tol = 1.0 / tolerance

    phi_m = _compute_phi(_prepare_centered_T(time_series, m, n_vectors, inv_tol), exponent)
    phi_m1 = _compute_phi(_prepare_centered_T(time_series, m + 1, n_vectors, inv_tol), exponent)

    if phi_m1 == 0 or phi_m == 0:
        _warn(
            "Fuzzy similarity degrees vanished to zero. Returning NaN.",
            RuntimeWarning,
        )
        return float(np.nan)

    return float(np.log(phi_m / phi_m1))
