"""Benchmark inputs and registered pyentrp cases.

Add a benchmark by decorating a function with `@bench("label")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyentrp import entropy as ent

if TYPE_CHECKING:
    from collections.abc import Callable


SHANNON_SYMBOL_COUNT = 10


@dataclass(frozen=True, eq=False)
class BenchmarkData:
    """Store generated inputs used by benchmarks for one series length."""

    series: np.ndarray  # white-noise series
    tolerance: float  # tolerance: 0.2 * std(series)
    symbols: np.ndarray  # quantile-bin indices from 0 to SHANNON_SYMBOL_COUNT - 1


def make_data(n: int, seed: int = 0) -> BenchmarkData:
    """Generate white noise and derived inputs for a series of length `n`."""
    series = np.random.default_rng(seed).standard_normal(n)
    quantile_levels = np.arange(1, SHANNON_SYMBOL_COUNT) / SHANNON_SYMBOL_COUNT
    bin_edges = np.quantile(series, quantile_levels)
    return BenchmarkData(
        series=series,
        tolerance=0.2 * float(np.std(series)),
        symbols=np.digitize(series, bin_edges),
    )


@dataclass(frozen=True)
class Case:
    """Describe a benchmark and the computation it runs."""

    label: str
    run: Callable[[BenchmarkData], object]


CASES: list[Case] = []


def bench(label: str):
    """Register a benchmark whose function receives `BenchmarkData` and runs the computation."""
    if any(c.label == label for c in CASES):
        raise ValueError(f"bench({label!r}): label already used")

    def register(fn: Callable[[BenchmarkData], object]) -> Callable[[BenchmarkData], object]:
        CASES.append(Case(label, fn))
        return fn

    return register


@bench(f"shannon_entropy ({SHANNON_SYMBOL_COUNT} quantile bins)")
def _benchmark_shannon_entropy(data: BenchmarkData):
    return ent.shannon_entropy(data.symbols)


@bench("permutation_entropy (order 3)")
def _benchmark_permutation_entropy_order_3(data: BenchmarkData):
    return ent.permutation_entropy(data.series, order=3)


@bench("permutation_entropy (order 6)")
def _benchmark_permutation_entropy_order_6(data: BenchmarkData):
    return ent.permutation_entropy(data.series, order=6)


@bench("weighted_permutation_entropy (order 3)")
def _benchmark_weighted_permutation_entropy_order_3(data: BenchmarkData):
    return ent.weighted_permutation_entropy(data.series, order=3)


@bench("multiscale_permutation_entropy (order 3, 5 scales)")
def _benchmark_multiscale_permutation_entropy(data: BenchmarkData):
    return ent.multiscale_permutation_entropy(data.series, 3, 1, 5)


@bench("dispersion_entropy (6 classes, order 3)")
def _benchmark_dispersion_entropy(data: BenchmarkData):
    return ent.dispersion_entropy(data.series, classes=6, order=3)


@bench("sample_entropy (m=2)")
def _benchmark_sample_entropy(data: BenchmarkData):
    return ent.sample_entropy(data.series, m=2, tolerance=data.tolerance)


@bench("multiscale_entropy (m=2, 5 scales)")
def _benchmark_multiscale_entropy(data: BenchmarkData):
    return ent.multiscale_entropy(data.series, m=2, tolerance=data.tolerance, scale=5)


@bench("composite_multiscale_entropy (m=2, scale 5)")
def _benchmark_composite_multiscale_entropy(data: BenchmarkData):
    return ent.composite_multiscale_entropy(data.series, m=2, scale=5, tolerance=data.tolerance)


@bench("rcmse (m=2, scale 5)")
def _benchmark_rcmse(data: BenchmarkData):
    return ent.rcmse(data.series, m=2, scale=5, tolerance=data.tolerance)


@bench("fuzzy_entropy (m=2)")
def _benchmark_fuzzy_entropy(data: BenchmarkData):
    return ent.fuzzy_entropy(data.series, 2, data.tolerance, 2)
