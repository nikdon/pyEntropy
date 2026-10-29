"""Benchmark inputs and registered pyentrp cases.

Add a benchmark by decorating a function with `@bench("label")`. Coverage is inferred from public entropy functions
named in the function body; pass `covers=` when a case calls a measure indirectly.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyentrp import entropy as ent

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, eq=False)
class Data:
    """Store generated inputs used by benchmarks for one series length."""

    x: np.ndarray  # white-noise series
    r: float  # tolerance: 0.2 * std(x)
    sym: np.ndarray  # x quantised into 10 integer symbols (for Shannon entropy)

    @classmethod
    def make(cls, n: int, seed: int = 0) -> Data:
        """Generate white noise and derived inputs for a series of length `n`."""
        x = np.random.default_rng(seed).standard_normal(n)
        edges = np.quantile(x, np.linspace(0, 1, 11)[1:-1])
        return cls(x=x, r=0.2 * float(np.std(x)), sym=np.digitize(x, edges))


@dataclass(frozen=True)
class Case:
    """Describe a benchmark and the public entropy measures it exercises."""

    label: str
    run: Callable[[Data], object]
    covers: frozenset[str]  # public pyentrp.entropy functions this case exercises (for --check)


CASES: list[Case] = []


def _is_measure(name: str, fn) -> bool:
    """Return whether a function is public rather than a helper such as util_* or time_delay_embedding."""
    helper = name.startswith(("_", "util_")) or name == "time_delay_embedding"
    return inspect.isfunction(fn) and fn.__module__ == ent.__name__ and not helper


def _called_measures(fn: Callable) -> frozenset[str]:
    """Public pyentrp.entropy functions a benchmark refers to, read from the names in its code."""
    return frozenset(n for n in fn.__code__.co_names if _is_measure(n, getattr(ent, n, None)))


def bench(label: str, covers: str | None = None):
    """Register a benchmark. The decorated function receives a `Data` and runs the computation.

    Which pyentrp function it exercises is read from the function body. Pass `covers="name"` only when the
    benchmark calls pyentrp indirectly (through a helper of your own), so the body doesn't mention it.
    """
    if covers is not None and not _is_measure(covers, getattr(ent, covers, None)):
        raise ValueError(f"bench({label!r}): {covers!r} is not a public function of pyentrp.entropy")
    if any(c.label == label for c in CASES):
        raise ValueError(f"bench({label!r}): label already used")

    def register(fn: Callable[[Data], object]) -> Callable[[Data], object]:
        names = _called_measures(fn) | ({covers} if covers else set())
        if not names:
            raise ValueError(f"bench({label!r}): calls no public pyentrp.entropy function; pass covers=... if indirect")
        CASES.append(Case(label, fn, frozenset(names)))
        return fn

    return register


@bench("shannon_entropy (10 symbols)")
def _(d):
    return ent.shannon_entropy(d.sym)


@bench("permutation_entropy (order 3)")
def _(d):
    return ent.permutation_entropy(d.x, order=3)


@bench("permutation_entropy (order 6)")
def _(d):
    return ent.permutation_entropy(d.x, order=6)


@bench("weighted_permutation_entropy (order 3)")
def _(d):
    return ent.weighted_permutation_entropy(d.x, order=3)


@bench("multiscale_permutation_entropy (m=3, 5 scales)")
def _(d):
    return ent.multiscale_permutation_entropy(d.x, 3, 1, 5)


@bench("dispersion_entropy (c=6, m=3)")
def _(d):
    return ent.dispersion_entropy(d.x, classes=6, order=3)


@bench("sample_entropy (m=2)")
def _(d):
    return ent.sample_entropy(d.x, 3, d.r)  # sample_length = m + 1


@bench("multiscale_entropy (m=2, 5 scales)")
def _(d):
    return ent.multiscale_entropy(d.x, 3, d.r, maxscale=5)


@bench("composite_multiscale_entropy (m=2, scale 5)")
def _(d):
    return ent.composite_multiscale_entropy(d.x, 3, 5, d.r)


@bench("rcmse (m=2, scale 5)")
def _(d):
    return ent.rcmse(d.x, 3, 5, d.r)


@bench("fuzzy_entropy (m=2)")
def _(d):
    return ent.fuzzy_entropy(d.x, 2, d.r, 2)


def uncovered() -> list[str]:
    """Return public entropy functions that lack a registered benchmark."""
    covered = {getattr(ent, n) for case in CASES for n in case.covers}
    return sorted(n for n, f in inspect.getmembers(ent) if _is_measure(n, f) and f not in covered)
