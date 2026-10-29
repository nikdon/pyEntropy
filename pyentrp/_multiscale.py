"""Coarse-graining and the measure records that the multiscale methods work with."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._validation import check_integer, validate_series


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
    pool : callable
        ``pool(list of stats) -> stats``: combine the statistics of several series, for the
        refined composite method.

    """

    prepare: Callable
    stats: Callable
    value: Callable
    min_length: Callable
    pool: Callable

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


def coarse_grain(x, scale):
    """Average non-overlapping blocks of `scale` samples, dropping an incomplete last block.

    Scale 1 returns `x` itself, so integer input stays exact.

    """
    if scale == 1:
        return x
    b = len(x) // scale
    return np.mean(x[: b * scale].reshape(b, scale), axis=1)


def util_granulate_time_series(time_series, scale):
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
