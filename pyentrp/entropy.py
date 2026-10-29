"""Entropy measures for time series.

This module is the public namespace of pyentrp (``from pyentrp import entropy as ent``).
The implementations live in the private modules of the package.
"""

from ._core import (
    _BINCOUNT_MAX_PATTERNS,
    _count_ids,
    _entropy_bits,
    _granulate,
    shannon_entropy,
    time_delay_embedding,
    util_granulate_time_series,
    util_pattern_space,
)
from ._dispersion import _map_to_classes, _validate_dispersion_params, dispersion_entropy
from ._fuzzy import _compute_phi, _prepare_centered_T, fuzzy_entropy
from ._ordinal import (
    _MAX_PERMUTATION_ORDER,
    _motif_lehmer_ids,
    _permutation_entropy,
    _window_weights,
    multiscale_permutation_entropy,
    permutation_entropy,
    weighted_permutation_entropy,
)
from ._sample import (
    _count_template_matches,
    _sampen,
    _sampen_last,
    composite_multiscale_entropy,
    multiscale_entropy,
    rcmse,
    refined_composite_multiscale_entropy,
    sample_entropy,
)
from ._validation import (
    _check_embedding_length,
    _check_integer,
    _check_tolerance,
    _default_tolerance,
    _resolve_tolerance,
    _validate_series,
)

__all__ = [
    "composite_multiscale_entropy",
    "dispersion_entropy",
    "fuzzy_entropy",
    "multiscale_entropy",
    "multiscale_permutation_entropy",
    "permutation_entropy",
    "rcmse",
    "refined_composite_multiscale_entropy",
    "sample_entropy",
    "shannon_entropy",
    "time_delay_embedding",
    "util_granulate_time_series",
    "util_pattern_space",
    "weighted_permutation_entropy",
]

# private helpers stay importable from here for the tests
_PRIVATE = (
    _BINCOUNT_MAX_PATTERNS,
    _MAX_PERMUTATION_ORDER,
    _check_embedding_length,
    _check_integer,
    _check_tolerance,
    _compute_phi,
    _count_ids,
    _count_template_matches,
    _default_tolerance,
    _entropy_bits,
    _granulate,
    _map_to_classes,
    _motif_lehmer_ids,
    _permutation_entropy,
    _prepare_centered_T,
    _resolve_tolerance,
    _sampen,
    _sampen_last,
    _validate_dispersion_params,
    _validate_series,
    _window_weights,
)
