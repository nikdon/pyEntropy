import math
from collections import Counter

import numpy as np


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

    # The code below follows the sample length convention of Ref [1] so:
    sample_length = sample_length - 1
    n = len(time_series)

    # N_temp is a vector that holds the number of matches
    # N_temp[k] holds matches templates of length k
    N_temp = np.zeros(sample_length + 2)

    # Templates of length 0 match by definition:
    N_temp[0] = n * (n - 1) / 2

    for i in range(n - sample_length - 1):
        template = time_series[i : (i + sample_length + 1)]  # We have `sample_length+1` elements in the template
        rem_time_series = time_series[i + 1 :]

        search_list = np.arange(len(rem_time_series) - sample_length, dtype=np.int32)
        for length in range(1, len(template) + 1):
            hit_list = np.abs(rem_time_series[search_list] - template[length - 1]) < tolerance
            N_temp[length] += np.sum(hit_list)
            search_list = search_list[hit_list] + 1

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
    for weight, indices in zip(weights, sorted_idx):  # noqa: B905
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
