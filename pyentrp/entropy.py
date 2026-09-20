import math
import warnings
from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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

    """
    series_length = len(time_series)
    embedded_series = np.empty((embedding_dimension, series_length - (embedding_dimension - 1) * delay))
    for i in range(embedding_dimension):
        embedded_series[i] = time_series[i * delay : i * delay + embedded_series.shape[1]]
    return embedded_series.T


def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with a given lag and dimension.

    Parameters
    ----------
    time_series : np.ndarray
        Vector or string of the sample data
    lag : int
        Lag between the beginning of sequences
    dim : int
        Dimension (number of patterns)

    Returns
    -------
    pattern_space: np.ndarray
        2D array of vectors

    Raises
    ------
    ValueError: If the lag is less than 1 or the result matrix exceeds the size limit.

    """
    n = len(time_series)

    if lag < 1:
        raise ValueError("Lag should be greater than or equal to 1.")

    if lag * dim > n:
        raise ValueError("Result matrix size limit exceeded. Adjust the lag or dim value.")

    pattern_space = np.zeros((n - lag * (dim - 1), dim))
    for i in range(dim):
        pattern_space[:, i] = time_series[i * lag : i * lag + n - lag * (dim - 1)]

    return pattern_space


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

    """
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
    time_series: np.ndarray | list[str]
        Vector or string of the sample data

    Returns
    -------
    ent: float
        The Shannon Entropy as float value

    """
    if isinstance(time_series, str):
        # Calculate frequency counts
        counter = Counter(time_series)
        total_count = len(time_series)

        # Calculate frequencies and Shannon entropy
        ent = 0.0
        for count in counter.values():
            freq = count / total_count
            ent += freq * np.log2(freq)

        ent = -ent
        return ent

    # Calculate frequency counts
    _, counts = np.unique(time_series, return_counts=True)
    total_count = len(time_series)

    # Calculate frequencies and Shannon entropy
    frequencies = counts / total_count
    ent = -np.sum(frequencies * np.log2(frequencies))

    return ent


def _count_template_matches(time_series, sample_length, tolerance):
    """Count template matches of lengths 0 to sample_length using Chebyshev norm.

    Parameters
    ----------
    time_series : np.ndarray
        Time series, 1-d vector.
    sample_length : int
        Length of longest template vector.
    tolerance : float
        Tolerance for matching.

    Returns
    -------
    N_temp : np.ndarray
        Array of length sample_length + 1, where N_temp[0] = n * (n - 1) / 2
        and N_temp[k] contains the number of matching template pairs of length k.

    """
    if not isinstance(time_series, np.ndarray):
        time_series = np.asarray(time_series)

    m = sample_length - 1
    n = len(time_series)

    N_temp = np.zeros(sample_length + 1)
    N_temp[0] = n * (n - 1) / 2

    for i in range(n - m - 1):
        template = time_series[i : (i + m + 1)]  # We have `sample_length` elements in the template
        rem_time_series = time_series[i + 1 :]

        search_list = np.arange(len(rem_time_series) - m, dtype=np.int32)
        for length in range(1, len(template) + 1):
            hit_list = np.abs(rem_time_series[search_list] - template[length - 1]) < tolerance
            N_temp[length] += np.sum(hit_list)
            search_list = search_list[hit_list] + 1

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
    tolerance : float
        tolerance (defaults to 0.1 * std(time_series)))

    Returns
    -------
    sampen: np.ndarray
        Array of Sample Entropies SE.
        SE[k] is the ratio `#templates of length k+1` / `#templates of length k`
        where `#templates of length 0` = n*(n - 1) / 2, by definition

    Notes
    -----
    The parameter 'sample_length' is equal to m + 1 in Ref[1].

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Sample_Entropy
    .. [2] http://physionet.incor.usp.br/physiotools/sampen/
    .. [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis of biological signals

    """
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    N_temp = _count_template_matches(time_series, sample_length, tolerance)
    sampen = -np.log(N_temp[1:] / N_temp[:-1])
    return sampen


def multiscale_entropy(time_series, sample_length, tolerance=None, maxscale=None):
    """Calculate Multiscale Entropy considering different time-scales of the time series.

    Parameters
    ----------
    time_series : np.ndarray
        Input time series for analysis.
    sample_length : int
        Bandwidth or group of points
    tolerance : float
        Tolerance value (default is 0.1 times the standard deviation of the `time_series`)
    maxscale : int
        Maximum timescale (default is the length of the `time_series`)

    Returns
    -------
    mse : np.ndarray
        Array of Multiscale Entropies

    References
    ----------
    .. [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
            Can be viewed at https://web.archive.org/web/20170207221539/http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html

    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    if maxscale is None:
        maxscale = len(time_series)

    mse = np.zeros(maxscale)
    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i + 1)
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
    return mse


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
        >>> print(permutation_entropy(x, order=2))
            0.918

    2. Normalized permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutation_entropy(x, order=3, normalize=True))
            0.589

    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = time_delay_embedding(x, embedding_dimension=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        factorial = math.factorial(order)
        pe /= np.log2(factorial)
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
        Array of Multiscale Permutation Entropies

    References
    ----------
    .. [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer`s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
    .. [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m

    """
    mspe = np.empty(scale)
    for i in range(scale):
        coarse_time_series = util_granulate_time_series(time_series, i + 1)
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
        >>> print(permutation_entropy(x, order=2))
            0.912

    2. Normalized weighted permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutation_entropy(x, order=3, normalize=True))
            0.547

    """
    x = time_delay_embedding(time_series, embedding_dimension=order, delay=delay)

    weights = np.var(x, axis=1)
    sorted_idx = x.argsort(kind="quicksort", axis=1)

    motif_weights = {}
    for weight, indices in zip(weights, sorted_idx, strict=True):
        motif = tuple(indices)
        if motif in motif_weights:
            motif_weights[motif] += weight
        else:
            motif_weights[motif] = weight

    pw = np.array(list(motif_weights.values()))
    pw /= weights.sum()

    b = np.log2(pw)
    wpe = -np.dot(pw, b)

    if normalize:
        wpe /= np.log2(math.factorial(order))

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
        Tolerance (default = 0.1 * std(time_series))

    Returns
    -------
    cmse : np.ndarray
        Array of Composite Multiscale Entropies

    References
    ----------
    .. [1] Wu, Shuen-De, et al. "Time series analysis using
        composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.

    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    cmse = np.zeros(scale)

    for i in range(scale):
        for j in range(i + 1):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            se = sample_entropy(tmp, sample_length, tolerance)[-1]
            cmse[i] += se / (i + 1)
    return cmse


def rcmse(time_series, sample_length, scale, tolerance=None):
    """Calculate Refined Composite Multiscale Entropy (RCMSE).

    Parameters
    ----------
    time_series : np.ndarray | list
        Time series for analysis.
    sample_length : int
        Number of sequential points of the time series.
    scale : int
        Scale factor.
    tolerance : float, optional
        Tolerance (default = 0.1 * std(time_series)).

    Returns
    -------
    rcmse : np.ndarray
        Array of Refined Composite Multiscale Entropies.

    Raises
    ------
    ValueError
        If scale < 1, sample_length < 1, time_series is not 1D,
        or time_series length is shorter than sample_length + 1.

    References
    ----------
    .. [1] Wu, S. D., et al. "Refined composite multiscale entropy: a novel measure
           for time series complexity analysis." IEEE Signal Processing Letters (2014).

    """
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array.")

    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")

    if sample_length < 1:
        raise ValueError("sample_length must be an integer >= 1.")

    if len(time_series) < sample_length + 1:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least sample_length + 1 ({sample_length + 1})."
        )

    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

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
            rcmse_values[i] = -np.log(n_m_plus_1 / n_m)

    return rcmse_values


def fuzzy_entropy(time_series, sample_length=2, tolerance=None, n=2):
    """Calculate Fuzzy Entropy (FuzzyEn) using exponential membership function.

    Fuzzy entropy characterizes the regularity of a time series by replacing
    the rigid Heaviside step boundary of sample entropy with a continuous
    exponential membership function and centering vectors around their local means.

    Parameters
    ----------
    time_series : np.ndarray or list
        Input time series for analysis, 1-d vector.
    sample_length : int, optional
        Embedding dimension (length of template vector m). Defaults to 2.
    tolerance : float, optional
        Tolerance radius r. Defaults to 0.2 * std(time_series).
    n : int or float, optional
        Weight of the fuzzy boundary (power of the exponential function).
        Defaults to 2.

    Returns
    -------
    fuzzyen : float
        Calculated Fuzzy Entropy.

    Raises
    ------
    ValueError
        If sample_length < 1, n <= 0, tolerance <= 0, time_series is not 1D,
        or time_series length is shorter than sample_length + 2.

    References
    ----------
    .. [1] Chen, W., et al. "Characterization of surface EMG signal based on fuzzy entropy."
           IEEE Transactions on Neural Systems and Rehabilitation Engineering 15.2 (2007): 266-272.

    """
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array.")

    if not isinstance(sample_length, (int, np.integer)) or sample_length < 1:
        raise ValueError("sample_length must be an integer >= 1.")

    if n <= 0:
        raise ValueError("n must be greater than 0.")

    if len(time_series) < sample_length + 2:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least sample_length + 2 ({sample_length + 2})."
        )

    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be greater than 0.")

    if tolerance is None:
        std = np.std(time_series)
        if std == 0:
            return 0.0
        tolerance = 0.2 * std

    n_vectors = len(time_series) - sample_length
    i_idx, j_idx = np.triu_indices(n_vectors, k=1)
    n_pairs = len(i_idx)

    # Embedding vectors for dimension m
    vec_m = sliding_window_view(time_series[: n_vectors + sample_length - 1], sample_length)
    cent_m = vec_m - np.mean(vec_m, axis=1, keepdims=True)

    # Embedding vectors for dimension m + 1
    vec_m1 = sliding_window_view(time_series[: n_vectors + sample_length], sample_length + 1)
    cent_m1 = vec_m1 - np.mean(vec_m1, axis=1, keepdims=True)

    chunk_size = 500_000
    if n_pairs <= chunk_size:
        dist_m = np.max(np.abs(cent_m[i_idx] - cent_m[j_idx]), axis=1)
        sim_m = np.exp(-((dist_m / tolerance) ** n))
        phi_m = np.mean(sim_m)

        dist_m1 = np.max(np.abs(cent_m1[i_idx] - cent_m1[j_idx]), axis=1)
        sim_m1 = np.exp(-((dist_m1 / tolerance) ** n))
        phi_m1 = np.mean(sim_m1)
    else:
        sum_sim_m = 0.0
        sum_sim_m1 = 0.0
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            sel_i = i_idx[start:end]
            sel_j = j_idx[start:end]

            d_m = np.max(np.abs(cent_m[sel_i] - cent_m[sel_j]), axis=1)
            sum_sim_m += np.sum(np.exp(-((d_m / tolerance) ** n)))

            d_m1 = np.max(np.abs(cent_m1[sel_i] - cent_m1[sel_j]), axis=1)
            sum_sim_m1 += np.sum(np.exp(-((d_m1 / tolerance) ** n)))

        phi_m = sum_sim_m / n_pairs
        phi_m1 = sum_sim_m1 / n_pairs

    if phi_m1 == 0 or phi_m == 0:
        warnings.warn(
            "Fuzzy similarity degrees vanished to zero. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(np.nan)

    return float(-np.log(phi_m1 / phi_m))


MIN_DISPERSION_CLASSES = 2
MAX_SAFE_INT64_BITS = 62
ENTROPY_EPSILON = 1e-15


def _validate_dispersion_params(  # noqa: PLR0913
    time_series,
    classes,
    order,
    delay,
    mapping,
    normalize,
):
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array.")

    if isinstance(classes, bool) or not isinstance(classes, (int, np.integer)) or classes < MIN_DISPERSION_CLASSES:
        raise ValueError("classes must be an integer >= 2.")

    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be an integer >= 1.")

    if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay must be an integer >= 1.")

    if not isinstance(mapping, str) or mapping.lower() not in ("ncdf", "linear"):
        raise ValueError("mapping must be either 'ncdf' or 'linear'.")

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean.")

    min_length = (order - 1) * delay + 1
    if len(time_series) < min_length:
        raise ValueError(
            f"Time series length ({len(time_series)}) must be at least (order - 1) * delay + 1 ({min_length})."
        )

    return time_series


def _map_to_classes(time_series, classes, mapping):
    mapping_lower = mapping.lower()
    if mapping_lower == "ncdf":
        mu = np.mean(time_series)
        sigma = np.std(time_series)
        if sigma == 0:
            return None
        v_erf = np.vectorize(math.erf, otypes=[float])
        y = 0.5 * (1.0 + v_erf((time_series - mu) / (sigma * np.sqrt(2))))
    else:
        min_val = np.min(time_series)
        max_val = np.max(time_series)
        if min_val == max_val:
            return None
        y = (time_series - min_val) / (max_val - min_val)

    return np.clip(np.rint(classes * y + 0.5), 1, classes).astype(np.int64)


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
        Time series data.
    classes : int, default=3
        Number of classes c used to discretize the time series (c >= 2).
    order : int, default=3
        Embedding dimension / pattern length (m >= 1).
    delay : int, default=1
        Time delay between consecutive pattern points (d >= 1).
    mapping : {"ncdf", "linear"}, default="ncdf"
        Mapping function to project the signal into the [0, 1] range:
        - "ncdf": Normal cumulative distribution function mapping.
        - "linear": Linear min-max normalization.
    normalize : bool, default=False
        If True, divide the entropy by log2(classes**order) to normalize
        the metric between 0 and 1.

    Returns
    -------
    de : float
        Dispersion Entropy in bits (or normalized between 0 and 1).

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
    time_series = _validate_dispersion_params(time_series, classes, order, delay, mapping, normalize)

    z = _map_to_classes(time_series, classes, mapping)
    if z is None:
        return 0.0

    z_zero = z - 1
    patterns = time_delay_embedding(z_zero, embedding_dimension=order, delay=delay).astype(np.int64)

    if order * math.log2(classes) >= MAX_SAFE_INT64_BITS:
        hashmult = np.array([classes**i for i in range(order)], dtype=object)
    else:
        hashmult = np.power(classes, np.arange(order, dtype=np.int64))

    hashval = np.dot(patterns, hashmult)
    _, counts = np.unique(hashval, return_counts=True)
    p = counts / np.sum(counts)
    de = float(-np.sum(p * np.log2(p)))

    if de == 0.0 or abs(de) < ENTROPY_EPSILON:
        return 0.0

    if normalize:
        de /= float(order * math.log2(classes))

    return float(de)
