import math
import warnings
from collections import Counter
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# largest pattern space (order! or classes ** order) counted with np.bincount; larger spaces use np.unique.
_BINCOUNT_MAX_PATTERNS = 1_000_000

# largest permutation order: pattern ids go up to order! - 1, and 20! - 1 is the last that fits in int64.
_MAX_PERMUTATION_ORDER = 20


def _check_integer(name, value, minimum, maximum=None):
    """Validate that `value` is an integer (booleans excluded) in [minimum, maximum].

    Raises
    ------
    TypeError
        If value is not an integer.
    ValueError
        If value is below the minimum or above the maximum (when given).

    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}.")


def _validate_series(time_series, keep_integer=False):
    """Return the time series as a 1D float array.

    Parameters
    ----------
    time_series : array_like
        Input series.
    keep_integer : bool, optional
        If True, return integer input with its dtype unchanged instead of converting it
        to float. Use this where values are only compared: float64 cannot tell apart
        integers above 2**53 (e.g. 2**53 and 2**53 + 1).

    Raises
    ------
    ValueError
        If the series is not 1D or contains NaN or infinite values.

    """
    time_series = np.asarray(time_series)
    if not (keep_integer and np.issubdtype(time_series.dtype, np.integer)):
        time_series = time_series.astype(float, copy=False)

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array.")

    if not np.all(np.isfinite(time_series)):
        raise ValueError("time_series must contain only finite values.")

    return time_series


def _check_tolerance(tolerance):
    """Validate an explicitly supplied tolerance.

    Raises
    ------
    ValueError
        If tolerance is not greater than 0 (this includes NaN).

    """
    if not tolerance > 0:
        raise ValueError("tolerance must be greater than 0.")


def _default_tolerance(time_series, factor):
    """Return the default template-matching tolerance, factor * std(time_series).

    A constant series has std 0, and a zero tolerance matches nothing under the strict
    `<` test. Every positive tolerance gives the same match counts on a constant series
    (all differences are 0), so return 1.0 there: the result then equals the one for any
    explicit tolerance, i.e. the limit as the tolerance goes to 0 from above.

    """
    if time_series.size and np.ptp(time_series) == 0:
        return 1.0
    return factor * np.std(time_series)


def _check_embedding_length(time_series, order, delay):
    """Check that the series holds at least one window of `order` samples spaced `delay` apart.

    Raises
    ------
    ValueError
        If the series has fewer than (order - 1) * delay + 1 samples.

    """
    min_length = (order - 1) * delay + 1
    if len(time_series) < min_length:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least (order - 1) * delay + 1 ({min_length})."
        )


def time_delay_embedding(time_series, embedding_dimension, delay):
    """Calculate time-delayed embedding.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series, shape (n_times)
    embedding_dimension : int
        The embedding dimension (order).
    delay : int
        The delay between embedded points.

    Returns
    -------
    embedded : ndarray
        The embedded time series with shape (n_times - (order - 1) * delay, order).
        This is a read-only view on the input data (no copy is made), so copy it
        before modifying it in place.

    Raises
    ------
    TypeError
        If embedding_dimension or delay is not an integer (booleans excluded).
    ValueError
        If embedding_dimension or delay is below 1, or the series is too short.

    """
    _check_integer("embedding_dimension", embedding_dimension, 1)
    _check_integer("delay", delay, 1)
    window = (embedding_dimension - 1) * delay + 1
    view = sliding_window_view(time_series, window_shape=window)
    return view[:, ::delay] if delay > 1 else view


def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with a given lag and dimension.

    Same as ``time_delay_embedding(time_series, dim, lag)``, returned as a writable float copy.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    lag : int
        Lag between consecutive samples of a sequence.
    dim : int
        Number of samples in a sequence.

    Returns
    -------
    pattern_space: np.ndarray
        Float array of shape (n - lag * (dim - 1), dim).

    Raises
    ------
    TypeError
        If lag or dim is not an integer (booleans excluded).
    ValueError
        If lag or dim is below 1, time_series is not 1D or contains non-finite
        values, or has fewer than lag * (dim - 1) + 1 samples.

    """
    time_series = _validate_series(time_series)
    _check_integer("lag", lag, 1)
    _check_integer("dim", dim, 1)
    _check_embedding_length(time_series, dim, lag)
    return time_delay_embedding(time_series, dim, lag).copy()


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series.

    Parameters
    ----------
    time_series : np.ndarray
        Time series
    scale : int
        Scale factor

    Returns
    -------
    cts : np.ndarray
        Array of coarse-grained time series with a given scale factor

    Raises
    ------
    TypeError
        If scale is not an integer (booleans excluded).
    ValueError
        If scale is below 1.

    """
    _check_integer("scale", scale, 1)
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    n = time_series.shape[0]
    b = n // scale
    cts = np.mean(time_series[: b * scale].reshape(b, scale), axis=1)
    return cts


def shannon_entropy(time_series):
    """Calculate Shannon Entropy of the sample data.

    Parameters
    ----------
    time_series: np.ndarray | list | str
        Sample data; each distinct value (or character of a string) is a symbol.

    Returns
    -------
    ent: float
        The Shannon Entropy in bits; 0.0 when only one symbol occurs.

    Raises
    ------
    ValueError
        If time_series is empty.

    """
    if isinstance(time_series, str):
        counts = np.fromiter(Counter(time_series).values(), dtype=float)
    else:
        _, counts = np.unique(time_series, return_counts=True)

    if counts.size == 0:
        raise ValueError("time_series must not be empty.")

    p = counts / counts.sum()
    ent = float(-np.dot(p, np.log2(p)))
    # a single symbol gives -0.0
    return ent if ent > 0.0 else 0.0


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
    if sample_length < 1:
        raise ValueError("sample_length must be a positive integer.")

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

    Warns
    -----
    RuntimeWarning
        For zero match counts, under NumPy's default error settings.

    Notes
    -----
    The parameter 'sample_length' is equal to m + 1 in Ref[1].

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Sample_Entropy
    .. [2] http://physionet.incor.usp.br/physiotools/sampen/
    .. [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis of biological signals

    Raises
    ------
    TypeError
        If sample_length is not an integer (booleans excluded).
    ValueError
        If sample_length < 1, time_series is not 1D or contains non-finite
        values, or an explicitly supplied tolerance is not greater than 0.

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)

    if tolerance is None:
        tolerance = _default_tolerance(time_series, 0.1)
    else:
        _check_tolerance(tolerance)

    N_temp = _count_template_matches(time_series, sample_length, tolerance)
    sampen = np.log(N_temp[:-1] / N_temp[1:])
    return sampen


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

    References
    ----------
    .. [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
            Can be viewed at https://web.archive.org/web/20170207221539/http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html

    Raises
    ------
    TypeError
        If sample_length or maxscale is not an integer (booleans excluded).
    ValueError
        If sample_length or maxscale is below 1, time_series is not 1D or
        contains non-finite values, or an explicitly supplied tolerance is
        not greater than 0.

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)

    if tolerance is None:
        tolerance = _default_tolerance(time_series, 0.1)
    else:
        _check_tolerance(tolerance)

    if maxscale is None:
        maxscale = 20
    _check_integer("maxscale", maxscale, 1)

    mse = np.zeros(maxscale)
    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i + 1)
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
    return mse


def _motif_lehmer_ids(x, order, delay):
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


def _window_weights(x, order, delay):
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

    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf

    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf

    Notes
    -----
    Last updated (Oct 2018) by Raphael Vallat (raphaelvallat9@gmail.com):
    - Major speed improvements
    - Use of base 2 instead of base e
    - Added normalization

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

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, order is above 20, time_series is not 1D
        or contains non-finite values, or has fewer than (order - 1) * delay + 1
        samples.

    """
    # keep integer input as is: converting to float would merge large integers before they are compared
    x = _validate_series(time_series, keep_integer=True)
    _check_integer("order", order, 1, _MAX_PERMUTATION_ORDER)
    _check_integer("delay", delay, 1)
    _check_embedding_length(x, order, delay)
    # associate a unique integer in [0, order! - 1] to each ordinal pattern
    lehmer = _motif_lehmer_ids(x, order=order, delay=delay)
    n_motifs = math.factorial(order)
    # return the counts
    if n_motifs <= _BINCOUNT_MAX_PATTERNS:
        c = np.bincount(lehmer, minlength=n_motifs)
    else:
        c = np.unique(lehmer, return_counts=True)[1]
    c = c[c > 0]
    p = c / c.sum()
    pe = float(-np.dot(p, np.log2(p)))
    if pe <= 0.0:
        return 0.0  # a single pattern; avoid returning -0.0
    # order 1 has a single pattern, so the entropy is 0 and there is nothing to normalize by
    if normalize and n_motifs > 1:
        pe /= math.log2(n_motifs)
    return pe


def multiscale_permutation_entropy(time_series, m, delay, scale):
    """Calculate the Multiscale Permutation Entropy.

    Parameters
    ----------
    time_series : np.ndarray
        Time series for analysis
    m : int
        Order of permutation entropy
    delay : int
        Time delay
    scale : int
        Scale factor

    Returns
    -------
    mspe : np.ndarray
        Array of length scale; index i corresponds to scale i + 1. An entry is
        NaN when the coarse-grained series at that scale has fewer than
        (m - 1) * delay + 1 samples.

    References
    ----------
    .. [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer`s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
    .. [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m

    Raises
    ------
    TypeError
        If m, delay, or scale is not an integer (booleans excluded).
    ValueError
        If m, delay, or scale is below 1, m is above 20, or time_series is not
        1D or contains non-finite values.

    Warns
    -----
    RuntimeWarning
        When the coarse-grained series is too short at some scales; those
        entries are NaN.

    """
    time_series = _validate_series(time_series, keep_integer=True)
    _check_integer("m", m, 1, _MAX_PERMUTATION_ORDER)
    _check_integer("delay", delay, 1)
    _check_integer("scale", scale, 1)

    min_length = (m - 1) * delay + 1
    mspe = np.full(scale, np.nan)
    for i in range(scale):
        # scale 1 is the series itself; averaging would turn integers into floats and could merge them
        coarse_time_series = time_series if i == 0 else util_granulate_time_series(time_series, i + 1)
        if len(coarse_time_series) < min_length:
            # coarse series only get shorter with scale, so every remaining scale is too short
            warnings.warn(
                f"Coarse-grained series is shorter than (m - 1) * delay + 1 ({min_length}) "
                f"from scale {i + 1} on. Returning NaN for those scales.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        mspe[i] = permutation_entropy(coarse_time_series, order=m, delay=delay)
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

    References
    ----------
    .. [1] Bilal Fadlallah, Badong Chen, Andreas Keil, and José Príncipe
           Phys. Rev. E 87, 022911 - Published 20 February 2013

    Notes
    -----
    - Updated in Jun 2023 by Nikolay Donets
    - Updated in March 2021 by Samuel Dotson (samgdotson@gmail.com)

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

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded).
    ValueError
        If order or delay is below 1, order is above 20, time_series is not 1D
        or contains non-finite values, or has fewer than (order - 1) * delay + 1
        samples.

    """
    # patterns come from the original values (integers are not rounded); the weights need float
    x = _validate_series(time_series, keep_integer=True)
    _check_integer("order", order, 1, _MAX_PERMUTATION_ORDER)
    _check_integer("delay", delay, 1)
    _check_embedding_length(x, order, delay)

    lehmer = _motif_lehmer_ids(x, order=order, delay=delay)
    weights = _window_weights(x.astype(np.float64, copy=False), order=order, delay=delay)

    n_motifs = math.factorial(order)
    if n_motifs <= _BINCOUNT_MAX_PATTERNS:
        pw = np.bincount(lehmer, weights=weights, minlength=n_motifs)
    else:
        _, inv = np.unique(lehmer, return_inverse=True)
        pw = np.bincount(inv.ravel(), weights=weights)

    # motifs without weight (e.g. only constant windows) contribute 0 * log(0) := 0
    pw = pw[pw > 0]
    if len(pw) == 0:
        return 0.0
    pw /= pw.sum()

    wpe = float(-np.dot(pw, np.log2(pw)))
    if wpe <= 0.0:
        return 0.0  # a single weighted pattern; avoid returning -0.0

    if normalize:
        wpe /= math.log2(n_motifs)

    return wpe


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

    References
    ----------
    .. [1] Wu, Shuen-De, et al. "Time series analysis using
        composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.

    Raises
    ------
    TypeError
        If sample_length or scale is not an integer (booleans excluded).
    ValueError
        If sample_length or scale is below 1, time_series is not 1D or
        contains non-finite values, or an explicitly supplied tolerance is
        not greater than 0.

    """
    time_series = _validate_series(time_series)
    _check_integer("sample_length", sample_length, 1)
    _check_integer("scale", scale, 1)

    if tolerance is None:
        tolerance = _default_tolerance(time_series, 0.1)
    else:
        _check_tolerance(tolerance)

    cmse = np.zeros(scale)

    for i in range(scale):
        for j in range(i + 1):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            se = sample_entropy(tmp, sample_length, tolerance)[-1]
            cmse[i] += se / (i + 1)
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

    if tolerance is None:
        tolerance = _default_tolerance(time_series, 0.1)
    else:
        _check_tolerance(tolerance)

    rcmse_values = np.zeros(scale)

    for i in range(scale):
        total_matches = np.zeros(sample_length + 1)
        for j in range(i + 1):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            total_matches += _count_template_matches(tmp, sample_length, tolerance)

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


def fuzzy_entropy(time_series, sample_length=2, tolerance=None, n=2):
    """Calculate Fuzzy Entropy (FuzzyEn) using exponential membership function.

    Fuzzy entropy characterizes the regularity of a time series by replacing
    the rigid Heaviside step boundary of sample entropy with a continuous
    exponential membership function and centering vectors around their local means.

    Parameters
    ----------
    time_series : np.ndarray or list
        Input time series for analysis, 1-d vector of finite values.
    sample_length : int, optional
        Embedding dimension (length of template vector m). Defaults to 2.
    tolerance : float, optional
        Positive radius r in the same units as the time series, used in
        exp(-(d / r)**n), where d is the Chebyshev distance between centered
        templates. This is the form of Chen et al. (2007), without the ln(2)
        scaling used in some later papers. Defaults to 0.2 * std(time_series),
        unlike sample_entropy and the multiscale functions, which default to
        0.1 * std(time_series).
    n : int or float, optional
        Weight of the fuzzy boundary (power of the exponential function).
        Defaults to 2.

    Returns
    -------
    fuzzyen : float
        Negative natural logarithm of the ratio of mean similarities for
        lengths sample_length + 1 and sample_length, in nats. Self-matches
        are excluded. Returns 0.0 for a constant series (with the default
        tolerance) and NaN if either mean similarity vanishes to zero.

    Raises
    ------
    TypeError
        If sample_length is not an integer (booleans excluded).
    ValueError
        If sample_length < 1, n <= 0, an explicitly supplied tolerance is not
        greater than 0, time_series is not 1D or contains non-finite values,
        or time_series is shorter than sample_length + 2.

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
    _check_integer("sample_length", sample_length, 1)

    if not n > 0:
        raise ValueError("n must be greater than 0.")

    if len(time_series) < sample_length + 2:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least sample_length + 2 ({sample_length + 2})."
        )

    if tolerance is None:
        std = np.std(time_series)
        if std == 0:
            return 0.0
        tolerance = 0.2 * std
    else:
        _check_tolerance(tolerance)

    # both lengths use the same number of vectors (the first n_vectors windows)
    n_vectors = len(time_series) - sample_length
    inv_tol = 1.0 / tolerance

    cent_m = _prepare_centered_T(time_series, sample_length, n_vectors, inv_tol)
    cent_m1 = _prepare_centered_T(time_series, sample_length + 1, n_vectors, inv_tol)
    phi_m = _compute_phi(cent_m, n)
    phi_m1 = _compute_phi(cent_m1, n)

    if phi_m1 == 0 or phi_m == 0:
        warnings.warn(
            "Fuzzy similarity degrees vanished to zero. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(np.nan)

    return float(np.log(phi_m / phi_m1))


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
        The mapping name.

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

    _check_embedding_length(time_series, order, delay)

    return time_series, mapping.lower()


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
        ``classes``. Return None when the standard deviation ("ncdf") or
        min-max range ("linear") is zero.

    Raises
    ------
    ValueError
        If mapping is not 'ncdf' or 'linear'.

    Notes
    -----
    A scaled value y maps to min(floor(classes * y) + 1, classes). Values on
    class boundaries are assigned to the higher class. For "ncdf", class
    boundaries are the standard-normal quantiles k / classes for
    k = 1, ..., classes - 1.

    """
    match mapping:
        case "ncdf":
            sigma = np.std(time_series)
            if sigma == 0:
                return None
            zscores = (time_series - np.mean(time_series)) / sigma
            normal = NormalDist()
            boundaries = np.array([normal.inv_cdf(k / classes) for k in range(1, classes)])
            return np.searchsorted(boundaries, zscores, side="right").astype(np.int64) + 1
        case "linear":
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            if min_val == max_val:
                return None
            y = (time_series - min_val) / (max_val - min_val)
            return np.minimum(np.floor(classes * y).astype(np.int64) + 1, classes)
        case _:
            raise ValueError("mapping must be either 'ncdf' or 'linear'.")


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

    # classes >= 2, so order >= 20 always exceeds the limit (this skips computing a huge integer power)
    if order < 20 and int(classes) ** int(order) <= _BINCOUNT_MAX_PATTERNS:
        # small pattern space: encode each pattern as a base-`classes` integer and count with bincount
        n_win = len(z) - (order - 1) * delay
        z = z - 1  # labels 0 .. classes - 1
        hashval = z[:n_win].copy()
        for k in range(1, order):
            s = k * delay
            hashval *= classes
            hashval += z[s : s + n_win]
        counts = np.bincount(hashval)
        counts = counts[counts > 0]
    else:
        # count identical patterns row-wise; no integer hashing, so no overflow for large classes ** order
        patterns = time_delay_embedding(z, embedding_dimension=order, delay=delay)
        _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    de = float(-np.sum(p * np.log2(p)))

    if de <= 0.0:
        return 0.0

    if normalize:
        de /= order * math.log2(classes)

    return de
