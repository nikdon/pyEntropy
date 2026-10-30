"""Multiscale, composite multiscale and refined composite multiscale entropy for any measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ._validation import check_integer, validate_series, warn

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Measure:
    """How to compute an entropy measure in steps that the multiscale methods can reuse.

    Every function takes the parameters returned by ``prepare`` as keywords.

    Attributes
    ----------
    prepare : callable
        ``prepare(time_series, **params) -> (x, p)``: validate the series and the parameters
        and resolve data-dependent defaults (such as the tolerance) on the original series.
    stats : callable
        ``stats(x, **p)``: statistics of one series that ``pool`` can combine.
    value : callable
        ``value(stats, **p) -> float``: the entropy from the statistics.
    min_length : callable
        ``min_length(**p) -> int``: the shortest series that gives a value.
    pool : callable or None
        ``pool(list of stats) -> stats``: combine the statistics of several series, for the
        refined composite method.

    """

    prepare: Callable
    stats: Callable
    value: Callable
    min_length: Callable
    pool: Callable | None = None

    def __call__(self, time_series, **params):
        """Compute the measure of a single series."""
        x, p = self.prepare(time_series, **params)
        return self.value(self.stats(x, **p), **p)


# public measure function -> its Measure
MEASURES = {}


def register(func, measure):
    """Make `func` usable as the measure of the multiscale methods."""
    MEASURES[func] = measure


def sum_stats(stats):
    """Pool statistics that add up, such as match counts."""
    return np.sum(stats, axis=0)


def coarse_grain(x, scale, method="mean"):
    """Summarize non-overlapping blocks of `scale` samples, dropping an incomplete last block.

    method="mean" averages each block (scale 1 returns `x` itself, so integer input stays exact);
    method="variance" takes the unbiased variance of each block (scale >= 2).

    """
    if scale == 1 and method == "mean":
        return x
    b = len(x) // scale
    blocks = x[: b * scale].reshape(b, scale)
    if method == "variance":
        return np.var(blocks, axis=1, ddof=1)
    return np.mean(blocks, axis=1)


def check_coarse_graining(method):
    """Validate the coarse-graining method and return the first scale it defines."""
    if not isinstance(method, str):
        raise TypeError(f"coarse_graining must be a string, got {type(method).__name__}.")
    if method not in ("mean", "variance"):
        raise ValueError("coarse_graining must be 'mean' or 'variance'.")
    # the variance of a single sample is undefined, so variance coarse-graining starts at scale 2
    return 2 if method == "variance" else 1


def util_granulate_time_series(time_series: ArrayLike, scale: int) -> NDArray[np.float64]:
    """Extract coarse-grained time series.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    scale : int
        Scale factor: the length of the averaged blocks.

    Returns
    -------
    cts : np.ndarray
        The means of the non-overlapping blocks of `scale` samples, as a new float array;
        an incomplete last block is dropped.

    Raises
    ------
    TypeError
        If scale is not an integer (booleans excluded).
    ValueError
        If scale is below 1, or time_series is not 1D or contains non-finite values.

    """
    x = validate_series(time_series)
    check_integer("scale", scale, 1)
    return x.copy() if scale == 1 else coarse_grain(x, scale)


def bind(time_series, measure, params, method):
    """Validate the input once and return (Measure, series, resolved parameters, minimum length)."""
    spec = MEASURES.get(measure)
    if spec is None:
        if not callable(measure):
            raise TypeError(f"measure must be a callable, got {type(measure).__name__}.")
        spec = Measure(
            prepare=lambda ts, **p: (validate_series(ts), p),
            stats=measure,
            value=lambda v, **_: float(v),
            min_length=lambda **_: 1,
        )
    if method == "refined_composite_multiscale" and spec.pool is None:
        poolable = sorted(f.__name__ for f, m in MEASURES.items() if m.pool is not None)
        raise TypeError(
            "refined_composite_multiscale pools the internal statistics of a measure, so it needs one of "
            f"{', '.join(poolable)}."
        )
    x, p = spec.prepare(time_series, **params)
    min_length = spec.min_length(**p)
    # the other methods give NaN, with a warning, for scales that are too short, scale 1 included
    if method == "refined_composite_multiscale" and len(x) < min_length:
        raise ValueError(f"Time series length ({len(x)}) must be at least {min_length} for this measure.")
    return spec, x, p, min_length


def warn_too_short(scale, min_length, unit="scale"):
    """Warn that the coarse-grained (or decomposed) series are too short from `scale` on."""
    warn(
        f"Coarse-grained series are shorter than {min_length} samples from {unit} {scale} on. "
        f"Returning NaN for those {unit}s.",
        RuntimeWarning,
    )


def multiscale(
    time_series: ArrayLike,
    measure: Callable[..., Any],
    scale: int = 20,
    *,
    coarse_graining: Literal["mean", "variance"] = "mean",
    **params: Any,
) -> NDArray[np.float64]:
    """Compute an entropy measure of the coarse-grained series at scales 1 .. scale (Costa et al.).

    At scale k the series is averaged over non-overlapping blocks of k samples, and the
    measure is applied to the result.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    measure : callable
        One of sample_entropy (which contributes SampEn(m, r), its last entry),
        fuzzy_entropy, permutation_entropy, weighted_permutation_entropy and
        dispersion_entropy, or any function f(series, **params) -> float.
    scale : int, default=20
        Maximum scale, inclusive.
    coarse_graining : {"mean", "variance"}, default="mean"
        Keyword-only. Summarize each block by its mean (Costa et al. 2002) or by its unbiased variance
        (MSE_sigma2, Costa & Goldberger 2015), which follows the dynamics of the volatility. The
        variance of one sample is undefined, so scale 1 is NaN with "variance". Tolerances computed
        from the original series do not suit variances: pass `tolerance` explicitly (Costa &
        Goldberger used 0.5% of the original standard deviation).
    **params
        Parameters of the measure, e.g. ``m=2`` or ``order=3``. They are checked once, and
        settings that depend on the data are computed from the original series and kept
        across scales: the default tolerance of sample and fuzzy entropy, and the mean and
        std (or minimum and range) that dispersion entropy maps classes with.

    Returns
    -------
    values : np.ndarray
        Array of length scale; index i corresponds to scale i + 1. Units are those of
        the measure. An entry is NaN when the coarse-grained series is too short for the
        measure.

    Raises
    ------
    TypeError
        If measure is not callable, scale is not an integer (booleans excluded), or a
        parameter has the wrong type.
    ValueError
        If scale < 1 or a parameter is invalid (including, for some measures, a series too
        short for scale 1).

    Warns
    -----
    RuntimeWarning
        When the coarse-grained series are too short from some scale on.

    References
    ----------
    .. [1] Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of
           complex physiologic time series. Physical Review Letters, 89(6), 068102.

    Examples
    --------
    >>> import numpy as np
    >>> from pyentrp import multiscale, permutation_entropy
    >>> x = np.random.default_rng(0).standard_normal(1000)
    >>> multiscale(x, permutation_entropy, scale=3, order=3).round(2)
    array([2.58, 2.58, 2.58])

    """
    spec, x, p, min_length = bind(time_series, measure, params, "multiscale")
    check_integer("scale", scale, 1)
    first = check_coarse_graining(coarse_graining)

    values = np.full(scale, np.nan)
    for k in range(first, scale + 1):
        y = coarse_grain(x, k, coarse_graining)
        if len(y) < min_length:
            # coarse series only get shorter with the scale
            warn_too_short(k, min_length)
            break
        values[k - 1] = spec.value(spec.stats(y, **p), **p)
    return values


def composite_multiscale(
    time_series: ArrayLike,
    measure: Callable[..., Any],
    scale: int = 20,
    *,
    coarse_graining: Literal["mean", "variance"] = "mean",
    **params: Any,
) -> NDArray[np.float64]:
    """Compute an entropy measure at scales 1 .. scale, averaged over all k coarse-grainings at scale k (Wu et al.).

    At scale k the series is coarse-grained starting at each of its first k samples,
    and the k values of the measure are averaged. This uses every sample at every
    scale and lowers the variance of the estimate.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    measure : callable
        As in `multiscale`.
    scale : int, default=20
        Maximum scale, inclusive.
    coarse_graining : {"mean", "variance"}, default="mean"
        Keyword-only. Summarize each block by its mean (Costa et al. 2002) or by its unbiased variance
        (MSE_sigma2, Costa & Goldberger 2015), which follows the dynamics of the volatility. The
        variance of one sample is undefined, so scale 1 is NaN with "variance". Tolerances computed
        from the original series do not suit variances: pass `tolerance` explicitly (Costa &
        Goldberger used 0.5% of the original standard deviation).
    **params
        Parameters of the measure, as in `multiscale`.

    Returns
    -------
    values : np.ndarray
        Array of length scale; index i corresponds to scale i + 1. An entry is NaN
        when any coarse-grained series at that scale is too short for the measure.

    Raises
    ------
    TypeError
        As in `multiscale`.
    ValueError
        As in `multiscale`.

    Warns
    -----
    RuntimeWarning
        When the coarse-grained series are too short from some scale on.

    References
    ----------
    .. [1] Wu, S.-D., Wu, C.-W., Lin, S.-G., Wang, C.-C., & Lee, K.-Y. (2013). Time series
           analysis using composite multiscale entropy. Entropy, 15(3), 1069-1084.

    """
    spec, x, p, min_length = bind(time_series, measure, params, "composite_multiscale")
    check_integer("scale", scale, 1)
    first = check_coarse_graining(coarse_graining)

    values = np.full(scale, np.nan)
    for k in range(first, scale + 1):
        total = 0.0
        for j in range(k):
            y = coarse_grain(x[j:], k, coarse_graining)
            if len(y) < min_length:
                break
            total += spec.value(spec.stats(y, **p), **p) / k
        else:
            values[k - 1] = total
            continue
        # the last shift is the shortest, and it only gets shorter with the scale
        warn_too_short(k, min_length)
        break
    return values


def refined_composite_multiscale(
    time_series: ArrayLike,
    measure: Callable[..., Any],
    scale: int = 20,
    *,
    coarse_graining: Literal["mean", "variance"] = "mean",
    **params: Any,
) -> NDArray[np.float64]:
    """Compute an entropy measure at scales 1 .. scale, pooling all k coarse-grainings at scale k.

    Like `composite_multiscale`, but the k coarse-grained series are combined before the
    entropy is taken, which avoids undefined values on short series:

    - sample entropy sums the match counts (RCMSE, Wu et al. 2014);
    - fuzzy entropy sums the mean similarities (RCMFE, Azami & Escudero 2017);
    - permutation, weighted permutation and dispersion entropy average the pattern
      probabilities (RCMPE, Humeau-Heurtier et al. 2015; RCMDE, Azami et al. 2017).

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    measure : callable
        One of sample_entropy, fuzzy_entropy, permutation_entropy,
        weighted_permutation_entropy and dispersion_entropy.
    scale : int, default=20
        Maximum scale, inclusive.
    coarse_graining : {"mean", "variance"}, default="mean"
        Keyword-only. Summarize each block by its mean (Costa et al. 2002) or by its unbiased variance
        (MSE_sigma2, Costa & Goldberger 2015), which follows the dynamics of the volatility. The
        variance of one sample is undefined, so scale 1 is NaN with "variance". Tolerances computed
        from the original series do not suit variances: pass `tolerance` explicitly (Costa &
        Goldberger used 0.5% of the original standard deviation).
    **params
        Parameters of the measure, as in `multiscale`.

    Returns
    -------
    values : np.ndarray
        Array of length scale; index i corresponds to scale i + 1. Coarse-grained series
        that are too short for the measure are left out; an entry is NaN when all of them are.

    Raises
    ------
    TypeError
        If measure is not one of the functions above, or as in `multiscale`.
    ValueError
        As in `multiscale`, and if the series is too short for the measure at scale 1.

    Warns
    -----
    RuntimeWarning
        When all coarse-grained series are too short from some scale on.

    References
    ----------
    .. [1] Wu, S.-D., Wu, C.-W., Lin, S.-G., Lee, K.-Y., & Peng, C.-K. (2014). Analysis of
           complex time series using refined composite multiscale entropy. Physics Letters A,
           378(20), 1369-1374.
    .. [2] Humeau-Heurtier, A., Wu, C.-W., & Wu, S.-D. (2015). Refined composite multiscale
           permutation entropy to overcome multiscale permutation entropy length dependence.
           IEEE Signal Processing Letters, 22(12), 2364-2367.
    .. [3] Azami, H., & Escudero, J. (2017). Refined composite multivariate generalized
           multiscale fuzzy entropy: A tool for complexity analysis of multichannel signals.
           Physica A, 465, 261-276 (the univariate, mean-based case).
    .. [4] Azami, H., Rostaghi, M., Abásolo, D., & Escudero, J. (2017). Refined composite
           multiscale dispersion entropy and its application to biomedical signals. IEEE
           Transactions on Biomedical Engineering, 64(12), 2872-2879.

    """
    spec, x, p, min_length = bind(time_series, measure, params, "refined_composite_multiscale")
    check_integer("scale", scale, 1)
    first = check_coarse_graining(coarse_graining)

    values = np.full(scale, np.nan)
    for k in range(first, scale + 1):
        shifts = (coarse_grain(x[j:], k, coarse_graining) for j in range(k))
        stats = [spec.stats(y, **p) for y in shifts if len(y) >= min_length]
        if not stats:
            # the first shift is the longest, and it only gets shorter with the scale
            warn_too_short(k, min_length)
            break
        pool = cast("Callable", spec.pool)  # bind() rejects measures that cannot be pooled
        values[k - 1] = spec.value(pool(stats), **p)
    return values


def hierarchical_nodes(x, levels):
    """Return the series of the hierarchical decomposition tree, breadth first (2 ** levels - 1 nodes).

    The root is x cut to the largest power-of-two length; each node has two children of half its
    length: the average (x[2i] + x[2i+1]) / 2 (low frequencies) and the half difference
    (x[2i] - x[2i+1]) / 2 (high frequencies).

    """
    n = 1 << (len(x).bit_length() - 1) if len(x) else 0  # largest power of two <= len(x)
    nodes = [np.asarray(x[:n], dtype=float)]
    for k in range(2 ** (levels - 1) - 1):
        parent = nodes[k]
        even, odd = parent[0::2], parent[1::2]
        nodes.append((even + odd) / 2.0)
        nodes.append((even - odd) / 2.0)
    return nodes


def hierarchical(
    time_series: ArrayLike, measure: Callable[..., Any], levels: int = 3, **params: Any
) -> NDArray[np.float64]:
    """Compute an entropy measure on every node of the hierarchical decomposition of a series (Jiang et al.).

    Unlike coarse-graining, which keeps only the low-frequency (averaged) part, the hierarchical
    decomposition also keeps the high-frequency part at every level: each node splits into the averages
    and the half differences of its consecutive pairs of samples.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values. It is cut to the largest power-of-two length.
    measure : callable
        As in `multiscale`.
    levels : int, default=3
        Depth of the tree (>= 1): 2 ** levels - 1 nodes, the root being the series itself.
    **params
        Parameters of the measure, as in `multiscale` (data-dependent settings come from the original
        series).

    Returns
    -------
    values : np.ndarray
        Array of length 2 ** levels - 1 in breadth-first order: index 0 is the root, and the children of
        node i are 2i + 1 (averages) and 2i + 2 (half differences). An entry is NaN when the node is too
        short for the measure.

    Raises
    ------
    TypeError
        If measure is not callable, levels is not an integer (booleans excluded), or a parameter has the
        wrong type.
    ValueError
        If levels < 1 or a parameter is invalid.

    Warns
    -----
    RuntimeWarning
        When the nodes are too short for the measure from some level on.

    References
    ----------
    .. [1] Jiang, Y., Peng, C.-K., & Xu, Y. (2011). Hierarchical entropy analysis for biological
           signals. Journal of Computational and Applied Mathematics, 236(5), 728-742.

    Examples
    --------
    >>> import numpy as np
    >>> from pyentrp import hierarchical, sample_entropy
    >>> x = np.random.default_rng(0).standard_normal(1024)
    >>> hierarchical(x, sample_entropy, levels=2, m=2, tolerance=0.2).round(2)
    array([2.18, 1.71, 1.82])

    """
    spec, x, p, min_length = bind(time_series, measure, params, "hierarchical")
    check_integer("levels", levels, 1)
    nodes = hierarchical_nodes(x, levels)
    values = np.full(len(nodes), np.nan)
    for i, node in enumerate(nodes):
        if len(node) < min_length:
            # nodes are ordered by depth, and deeper nodes are shorter
            warn_too_short((i + 1).bit_length(), min_length, unit="level")
            break
        values[i] = spec.value(spec.stats(node, **p), **p)
    return values
