# pyentrp: entropy measures for time series in Python

[![PyPI](https://img.shields.io/pypi/v/pyentrp)](https://pypi.org/project/pyentrp/)
[![Python](https://img.shields.io/pypi/pyversions/pyentrp)](https://pypi.org/project/pyentrp/)
[![CI](https://github.com/nikdon/pyEntropy/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/nikdon/pyEntropy/actions/workflows/python-package.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/pyentrp)](https://pypistats.org/packages/pyentrp)
[![License](https://img.shields.io/github/license/nikdon/pyEntropy)](https://github.com/nikdon/pyEntropy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17727789.svg)](https://doi.org/10.5281/zenodo.17727789)

**pyEntropy** (PyPI and import name: `pyentrp`) computes twenty entropy and complexity measures of
one-dimensional time series: regularity measures such as **sample**, **approximate** and **fuzzy** entropy, ordinal
measures such as **permutation** entropy and **statistical complexity**, symbolic measures such as **dispersion**
entropy and **Lempel-Ziv** complexity, and **spectral**, **SVD** and **wavelet** entropy, with multiscale,
composite, refined composite and hierarchical versions. Typical uses are EEG/ECG and heart-rate variability,
vibration and condition monitoring, and finance.

- **Accurate.** Agrees with EntropyHub, NeuroKit2, antropy, ordpy and PyWavelets to 1e-12 wherever the definitions match
  ([details](#agreement-with-other-libraries)), and documents every place where they differ.
- **Fast.** Vectorized NumPy throughout; sample entropy compares only nearby templates ([timings](#performance)).
- **One dependency: NumPy.**
- **Typed.** Type hints for the whole public API (PEP 561), checked with [pyrefly](https://pyrefly.org) in CI.
- **Documented.** Each function lists its parameters, units and the paper it implements.

Upgrading from 2.x? See [Migrating from 2.x](#migrating-from-2x).

## Install

```bash
pip install pyentrp
```

Python 3.10+ and NumPy 1.26+.

## Quick start

```python
import numpy as np
import pyentrp as pe

rng = np.random.default_rng(42)
t = np.arange(1000)
noise = rng.standard_normal(1000)
sine = np.sin(2 * np.pi * t / 50)

for name, x in [("white noise", noise), ("sine wave", sine)]:
    r = 0.2 * np.std(x)  # tolerance, in signal units
    sampen = pe.sample_entropy(x, m=2, tolerance=r)[-1]
    permen = pe.permutation_entropy(x, order=3, normalize=True)
    print(f"{name:12s} SampEn={sampen:.3f}  PermEn={permen:.3f}")
```

```text
white noise  SampEn=2.202  PermEn=0.998
sine wave    SampEn=0.248  PermEn=0.491
```

Lower values mean a more regular signal. `sample_entropy` returns SampEn for every template length up to m,
so `[-1]` is SampEn(m). Multiscale entropy returns one value per scale:

```python
mse = pe.multiscale_entropy(noise, m=2, tolerance=0.2 * np.std(noise), scale=5)
for scale, value in enumerate(mse, start=1):
    print(f"scale {scale}: {value:.3f}")
```

```text
scale 1: 2.202
scale 2: 1.965
scale 3: 1.602
scale 4: 1.655
scale 5: 1.502
```

## Supported measures

| Measure | Function | Unit | Multiscale | Reference |
|---|---|---|---|---|
| **Regularity (template matching)** | | | | |
| Sample entropy | `sample_entropy` | nats | all; shortcuts `multiscale_entropy` (MSE), `composite_multiscale_entropy` (CMSE), `rcmse` (RCMSE, alias `refined_composite_multiscale_entropy`) | [Richman & Moorman (2000)](https://doi.org/10.1152/ajpheart.2000.278.6.H2039) |
| Approximate entropy | `approximate_entropy` | nats | not refined | [Pincus (1991)](https://doi.org/10.1073/pnas.88.6.2297) |
| Fuzzy entropy | `fuzzy_entropy` | nats | all | [Chen et al. (2007)](https://doi.org/10.1109/TNSRE.2007.897025) |
| Range entropy | `range_entropy` | nats | as a function | [Omidvarnia et al. (2018)](https://doi.org/10.3390/e20120962) |
| Cosine similarity entropy | `cosine_similarity_entropy` | bits | as a function | [Chanwimalueang & Mandic (2017)](https://doi.org/10.3390/e19120652) |
| Distribution entropy | `distribution_entropy` | 0-1 (bits with `normalize=False`) | as a function | [Li et al. (2015)](https://doi.org/10.1007/s11517-014-1216-0) |
| **Ordinal patterns** | | | | |
| Permutation entropy (Shannon, Renyi, Tsallis) | `permutation_entropy` | bits, or 0-1 with `normalize=True` | all; shortcut `multiscale_permutation_entropy` | [Bandt & Pompe (2002)](https://doi.org/10.1103/PhysRevLett.88.174102) |
| Weighted permutation entropy | `weighted_permutation_entropy` | bits, or 0-1 | all | [Fadlallah et al. (2013)](https://doi.org/10.1103/PhysRevE.87.022911) |
| Amplitude-aware permutation entropy | `amplitude_aware_permutation_entropy` | bits, or 0-1 | all | [Azami & Escudero (2016)](https://doi.org/10.1016/j.cmpb.2016.02.008) |
| Statistical complexity | `statistical_complexity` | 0-1 | all | [Rosso et al. (2007)](https://doi.org/10.1103/PhysRevLett.99.154102) |
| Bubble entropy | `bubble_entropy` | no unit | as a function | [Manis et al. (2017)](https://doi.org/10.1109/TBME.2017.2664105) |
| **Symbolic** | | | | |
| Dispersion entropy (plain or fluctuation-based; Shannon, Renyi, Tsallis) | `dispersion_entropy` | bits, or 0-1 | all | [Rostaghi & Azami (2016)](https://doi.org/10.1109/LSP.2016.2542881), [Azami & Escudero (2018)](https://doi.org/10.3390/e20030210) |
| Reverse dispersion entropy | `reverse_dispersion_entropy` | no unit, or 0-1 | all | [Li et al. (2019)](https://doi.org/10.3390/s19235203) |
| Slope entropy | `slope_entropy` | bits | as a function | [Cuesta-Frau (2019)](https://doi.org/10.3390/e21121167) |
| Corrected conditional entropy | `corrected_conditional_entropy` | nats, or 0-1 | as a function | [Porta et al. (1998)](https://doi.org/10.1007/s004220050414) |
| Lempel-Ziv complexity | `lempel_ziv_complexity` | normalized, or phrases | as a function | [Lempel & Ziv (1976)](https://doi.org/10.1109/TIT.1976.1055501) |
| **Spectral and decomposition** | | | | |
| Spectral entropy | `spectral_entropy` | bits, or 0-1 | as a function | [Inouye et al. (1991)](https://doi.org/10.1016/0013-4694(91)90138-T) |
| SVD entropy | `svd_entropy` | bits, or 0-1 | as a function | [Roberts et al. (1999)](https://doi.org/10.1007/BF02513272) |
| Wavelet entropy | `wavelet_entropy` | bits, or 0-1 | as a function | [Rosso et al. (2001)](https://doi.org/10.1016/S0165-0270(00)00356-3) |
| **Distribution of values** | | | | |
| Shannon entropy | `shannon_entropy` | bits | — | [Shannon (1948)](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) |

"All" means `multiscale`, `composite_multiscale`, `refined_composite_multiscale` and `hierarchical` (see
[Multiscale analysis](#multiscale-analysis)). "Not refined": every method but the refined composite one, which has no
pooled form for this measure. "As a function": every method but refined composite, with the measure applied to each
coarse-grained series as an ordinary function, so data-dependent defaults are computed per series.

Utilities: `time_delay_embedding(x, order, delay)` returns the delay-embedded windows, and
`util_granulate_time_series(x, scale)` the coarse-grained series used by the multiscale methods.

## Choosing a measure

| Measure | Cost | Looks at | Good for | Watch out for |
|---|---|---|---|---|
| Sample entropy | O(n²), less with the sorted band search | amplitude patterns within a tolerance | the standard regularity measure in physiology | inf or NaN when no templates match (short or very regular series) |
| Approximate entropy | O(n²), as sample entropy | the same, counting self-matches | comparison with older studies; always defined | biased towards regularity on short series |
| Fuzzy entropy | O(n²), the slowest | the same, with a smooth similarity | short or noisy series; varies smoothly with the tolerance | cost on long series |
| Range entropy | O(n²) | the spread of the differences between templates | amplitude-invariant regularity, tolerance in [0, 1] | cost on long series |
| Cosine similarity entropy | O(n²) | the angle between templates | signals with spikes or changing gain | cost on long series |
| Distribution entropy | O(n²) | the whole distribution of template distances | very short series; no tolerance | cost on long series |
| Permutation entropy | O(n) | the order of values only | long series, speed; unchanged by any monotonic transform of the data | ignores amplitude; needs n much larger than `order!` |
| Weighted / amplitude-aware permutation entropy | O(n) | order of values, weighted by variance or amplitude | like permutation entropy, when amplitude or spikes matter | as permutation entropy |
| Statistical complexity | O(n) | how far the ordinal patterns are from uniform | telling chaos from noise (with normalized permutation entropy) | as permutation entropy |
| Bubble entropy | O(n m²) | the number of swaps to sort each window | a measure with no tolerance and weak dependence on m | m has no default |
| Dispersion entropy | O(n) | amplitude classes and their order | fast amplitude-aware alternative to sample entropy | needs n much larger than `classes ** order` |
| Slope entropy | O(n) | the size and sign of consecutive steps | amplitude-aware symbolic dynamics | thresholds are in signal units |
| Corrected conditional entropy | O(n log n) per pattern length | the predictability of the next quantized value | short, quantized series (cardiovascular variability) | depends on the number of levels |
| Lempel-Ziv complexity | O(n) | the number of new patterns in a binarized series | randomness of a sequence, EEG and genomics | binarization discards amplitude |
| Spectral, SVD, wavelet entropy | O(n log n) / O(n m²) / O(n) | the spread of power over frequencies, components or scales | rhythmic and oscillatory signals (EEG, vibration) | blind to nonlinear structure |
| Shannon entropy | O(n log n) | the distribution of values, not their order | symbols and discrete data | no time structure |

Units differ between measures (nats, bits, or none; see the table above), so compare values of the same measure and
parameters only. The default tolerance of sample entropy is `0.1 * std(x)`, while most of the literature uses
`0.2 * std(x)`; pass `tolerance=` explicitly when you compare with published values.

## Multiscale analysis

Four functions take a measure and compute it over scales; the measure's own parameters follow as keywords:

```python
pe.multiscale(noise, pe.sample_entropy, scale=5, m=2)  # the same as multiscale_entropy
pe.composite_multiscale(noise, pe.permutation_entropy, scale=5, order=3, normalize=True)
pe.refined_composite_multiscale(noise, pe.dispersion_entropy, scale=5, classes=6, order=2, normalize=True)
pe.hierarchical(noise, pe.sample_entropy, levels=3, m=2)  # 7 nodes: the series, its averages and differences
```

| Function | At scale k | Reference |
|---|---|---|
| `multiscale` | the measure of the series averaged over blocks of k samples | [Costa et al. (2002)](https://doi.org/10.1103/PhysRevLett.89.068102) |
| `composite_multiscale` | the mean of the measure over the k coarse-grainings that start at samples 1..k | [Wu et al. (2013)](https://doi.org/10.3390/e15031069) |
| `refined_composite_multiscale` | the measure of the k coarse-grainings pooled together: summed match counts or similarities, averaged pattern probabilities | [Wu et al. (2014)](https://doi.org/10.1016/j.physleta.2014.03.034), [Humeau-Heurtier et al. (2015)](https://doi.org/10.1109/LSP.2015.2482603), [Azami & Escudero (2017)](https://doi.org/10.1016/j.physa.2016.07.077), [Azami et al. (2017)](https://doi.org/10.1109/TBME.2017.2679136) |
| `hierarchical` | the measure on every node of the decomposition tree: averages and half differences of consecutive pairs | [Jiang et al. (2011)](https://doi.org/10.1016/j.cam.2011.06.007) |

- Any measure in the table above works, and so does any function `f(series, **params) -> float`;
  `refined_composite_multiscale` needs a measure marked "all".
- `coarse_graining="variance"` summarizes each block by its unbiased variance instead of its mean (MSE_σ², [Costa &
  Goldberger (2015)](https://doi.org/10.3390/e17031197)), to follow the volatility of a signal. Scale 1 is then NaN,
  and the tolerance should be given explicitly (the authors used 0.5% of the original standard deviation).
- Parameters are checked once. Settings that depend on the data come from the original series and stay fixed across
  scales: the default tolerance, and the mean and std (or range) that dispersion entropy maps classes with.
- A scale whose coarse-grained series is too short for the measure gives NaN, with one `RuntimeWarning`.
  `refined_composite_multiscale` leaves out only the shifts that are too short, so it still gives a value while at
  least one shift at that scale is long enough.
- `multiscale_entropy`, `composite_multiscale_entropy`, `rcmse` and `multiscale_permutation_entropy` remain as
  shortcuts for these functions with sample and permutation entropy.

## Parameters

- `m` is the embedding dimension for sample, multiscale and fuzzy entropy: templates of length m are compared with
  templates of length m + 1. `sample_entropy(x, m=2)` returns SampEn for m = 0, 1, 2, so `[-1]` is m = 2.
- `order` is the pattern length for permutation, weighted permutation, dispersion entropy and `time_delay_embedding`.
  Permutation entropy orders go up to 20.
- `tolerance` is absolute, in the units of your signal. Default: `0.1 * std(x)` (population std) for sample entropy
  and its multiscale variants; `0.2 * std(x)` for fuzzy entropy. For the common r = 0.2 SD pass
  `tolerance=0.2 * np.std(x)`.
- `scale` is the largest scale of the multiscale functions. All of them default to 20, as in Costa et al. (2002),
  except `multiscale_permutation_entropy`, where it is required.
- Fuzzy entropy uses the membership function `exp(-(d / r)**exponent)` [Chen et al. (2007)](https://doi.org/10.1109/TNSRE.2007.897025).
- Ties in permutation entropy are ranked by position (stable ordering).

## Migrating from 2.x

The old names still work in 3.x with a `DeprecationWarning` and will be removed in 4.0.

| Function | 2.x | 3.x |
|---|---|---|
| `sample_entropy`, `multiscale_entropy`, `composite_multiscale_entropy`, `rcmse` | `sample_length=3` (or positional `3`) | `m=2` (`m = sample_length - 1`) |
| `multiscale_entropy` | `maxscale=` | `scale=` |
| `fuzzy_entropy` | `sample_length=`, `n=` | `m=`, `exponent=` (same values) |
| `multiscale_permutation_entropy` | `m=` | `order=` (same value) |
| `time_delay_embedding` | `embedding_dimension=` | `order=` |
| `util_pattern_space(x, lag, dim)` | | `time_delay_embedding(np.asarray(x, dtype=float), dim, lag).copy()` |

## Agreement with other libraries

[`benchmarks/crosscheck.py`](benchmarks/crosscheck.py) compares pyentrp with EntropyHub 2.0, NeuroKit2 0.2.13,
antropy 0.2.3, ordpy and PyWavelets on three signals: 120 checks of 30 cases, all within 1e-12 (most
within 1e-15). Where results differ, the definitions differ, and each case was checked by reproducing the other
library exactly:

- Composite MSE and MPE: EntropyHub cuts every shifted series to the same length; pyentrp keeps every complete block.
- Multiscale dispersion entropy: EntropyHub maps each coarse-grained series with its own mean and std; pyentrp keeps
  those of the original series, as Azami's reference code does.
- Composite and refined composite MSE in NeuroKit2 downsample (time-shift MSE) instead of averaging.
- Dispersion entropy in NeuroKit2 uses the sample std (ddof=1) and divides an entropy in bits by ln(c^m).
- Fuzzy entropy in NeuroKit2 counts each template as matching itself.
- Reverse dispersion entropy in EntropyHub sums (p - 1/K)² over the observed patterns only; the paper sums over all K.
- Cosine similarity entropy in EntropyHub rounds angles to 6 digits and tests angle < r; the paper uses angle <= r.
- Range entropy in EntropyHub never matches identical templates (0/0); pyentrp counts them as matches.
- Corrected conditional entropy in EntropyHub counts single patterns of length L - 1 and divides by N.
- Variance coarse-graining in EntropyHub uses the biased variance; the paper uses the unbiased one.

```bash
pip install EntropyHub neurokit2 antropy ordpy PyWavelets
python -m benchmarks.crosscheck
```

## Performance

pyentrp 3.0.0, Python 3.13.18, NumPy 2.2.6, Apple M4 Pro. White noise, best of up to 3 timeit repeats (calls over 2 s
are timed once), each benchmark in its own single-threaded process. Template-matching measures use r = 0.2 std.

| Function | N = 1,000 | N = 10,000 | N = 100,000 |
|---|---|---|---|
| `shannon_entropy (10 quantile bins)` | 10 µs | 39 µs | 313 µs |
| `permutation_entropy (order 3)` | 11 µs | 35 µs | 264 µs |
| `permutation_entropy (order 6)` | 24 µs | 83 µs | 649 µs |
| `weighted_permutation_entropy (order 3)` | 15 µs | 57 µs | 452 µs |
| `multiscale_permutation_entropy (order 3, 5 scales)` | 55 µs | 172 µs | 1.3 ms |
| `dispersion_entropy (6 classes, order 3)` | 29 µs | 90 µs | 624 µs |
| `composite_multiscale permutation_entropy (order 3, scale 5)` | 160 µs | 480 µs | 3.5 ms |
| `refined_composite_multiscale dispersion_entropy (6 classes, order 3, scale 5)` | 256 µs | 751 µs | 5.3 ms |
| `sample_entropy (m=2)` | 456 µs | 15.5 ms | 1.4 s |
| `multiscale_entropy (m=2, 5 scales)` | 964 µs | 31.2 ms | 2.4 s |
| `composite_multiscale_entropy (m=2, scale 5)` | 2.0 ms | 64.0 ms | 4.4 s |
| `rcmse (m=2, scale 5)` | 2.0 ms | 64.0 ms | 4.4 s |
| `fuzzy_entropy (m=2)` | 4.1 ms | 343.6 ms | 29.5 s |
| `approximate_entropy (m=2)` | 841 µs | 25.7 ms | 2.1 s |
| `range_entropy (m=2, r=0.2)` | 2.7 ms | 224.6 ms | 18.2 s |
| `cosine_similarity_entropy (m=2)` | 801 µs | 63.1 ms | 4.3 s |
| `distribution_entropy (m=2, 512 bins)` | 4.6 ms | 271.1 ms | 23.8 s |
| `bubble_entropy (m=10)` | 69 µs | 416 µs | 3.4 ms |
| `amplitude_aware_permutation_entropy (order 3)` | 17 µs | 63 µs | 481 µs |
| `statistical_complexity (order 4)` | 15 µs | 45 µs | 333 µs |
| `reverse_dispersion_entropy (6 classes, order 3)` | 27 µs | 87 µs | 605 µs |
| `slope_entropy (m=3)` | 22 µs | 121 µs | 1.4 ms |
| `lempel_ziv_complexity (median)` | 480 µs | 5.2 ms | 71.1 ms |
| `corrected_conditional_entropy (6 levels)` | 291 µs | 5.4 ms | 93.5 ms |
| `spectral_entropy (welch)` | 26 µs | 96 µs | 728 µs |
| `svd_entropy (order 3)` | 27 µs | 104 µs | 785 µs |
| `wavelet_entropy (db4)` | 132 µs | 359 µs | 1.9 ms |

Reproduce with `python -m benchmarks.bench --sizes 1000 10000 100000`.

## Tests

```bash
python -m unittest discover
```

The tests cover known values and input validation, constant, sinusoidal and noisy signals, mathematical equivalences,
brute-force reference counts, and the docstring examples.

## Citing

If pyentrp contributes to your research, please cite it. GitHub's "Cite this repository" button reads
[`CITATION.cff`](CITATION.cff), or use:

```bibtex
@software{pyentrp,
  author = {Donets, Nikolay},
  title  = {pyEntropy (pyentrp): entropy measures for time series in Python},
  doi    = {10.5281/zenodo.17727789},
  url    = {https://github.com/nikdon/pyEntropy}
}
```

Please also cite the original paper of each measure you use (links in [Supported measures](#supported-measures)).

## Acknowledgements

Raphael Vallat sped up permutation entropy and added base-2 logarithms and normalization (2018).
Samuel Dotson contributed weighted permutation entropy (2021).

## Contributing

Bug reports, fixes and new measures are welcome: open an [issue](https://github.com/nikdon/pyEntropy/issues) or a
pull request. To set up and check a change:

```bash
poetry install
poetry run pre-commit install        # ruff, pyrefly and commit message checks on commit
poetry run python -m unittest discover
poetry run ruff check . && poetry run ruff format --check . && poetry run pyrefly check
python -m benchmarks.bench --sizes 1000 10000   # for changes that can affect speed
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); releases are cut with
commitizen.

## License

Apache-2.0. See [LICENSE](LICENSE).
