# pyentrp: entropy measures for time series in Python

[![PyPI](https://img.shields.io/pypi/v/pyentrp)](https://pypi.org/project/pyentrp/)
[![Python](https://img.shields.io/pypi/pyversions/pyentrp)](https://pypi.org/project/pyentrp/)
[![CI](https://github.com/nikdon/pyEntropy/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/nikdon/pyEntropy/actions/workflows/python-package.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/pyentrp)](https://pypistats.org/packages/pyentrp)
[![License](https://img.shields.io/github/license/nikdon/pyEntropy)](https://github.com/nikdon/pyEntropy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17727789.svg)](https://doi.org/10.5281/zenodo.17727789)

**pyEntropy** (PyPI and import name: `pyentrp`) computes **sample entropy**, **permutation entropy**,
**fuzzy entropy**, **dispersion entropy** and **Shannon entropy** of one-dimensional time series. Every measure
except Shannon entropy also has multiscale, composite multiscale and refined composite multiscale versions. Typical uses are EEG/ECG and heart-rate variability, vibration and condition
monitoring, and finance.

- **One dependency: NumPy.**
- **Fast.** See ([timings](#performance)).
- **Tested.** See the [test suite](#tests) for known values, edge cases and mathematical equivalences.
- **Documented.** Each function lists its parameters, units and the paper it implements.

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

Lower values mean a more regular signal. Multiscale entropy returns one value per scale:

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

## Multiscale analysis

Three functions take any measure and compute it at scales 1 to `scale`. The measure's own parameters follow as
keywords:

```python
pe.multiscale(noise, pe.sample_entropy, scale=5, m=2)  # the same as multiscale_entropy
pe.composite_multiscale(noise, pe.permutation_entropy, scale=5, order=3, normalize=True)
pe.refined_composite_multiscale(noise, pe.dispersion_entropy, scale=5, classes=6, order=2, normalize=True)
```

| Function | At scale k | Reference |
|---|---|---|
| `multiscale` | the measure of the series averaged over blocks of k samples | [Costa et al. (2002)](https://doi.org/10.1103/PhysRevLett.89.068102) |
| `composite_multiscale` | the mean of the measure over the k coarse-grainings that start at samples 1..k | [Wu et al. (2013)](https://doi.org/10.3390/e15031069) |
| `refined_composite_multiscale` | the measure of the k coarse-grainings pooled together: summed match counts or similarities, averaged pattern probabilities | [Wu et al. (2014)](https://doi.org/10.1016/j.physleta.2014.03.034), [Humeau-Heurtier et al. (2015)](https://doi.org/10.1109/LSP.2015.2482603), [Azami et al. (2017)](https://doi.org/10.1109/TBME.2017.2679136) |

- The measure can be `sample_entropy`, `fuzzy_entropy`, `permutation_entropy`, `weighted_permutation_entropy` or
  `dispersion_entropy`. `multiscale` and `composite_multiscale` also accept any function `f(series, **params) -> float`.
- Parameters are checked once. Settings that depend on the data come from the original series and stay fixed across
  scales: the default tolerance, and the mean and std (or range) that dispersion entropy maps classes with.
- A scale whose coarse-grained series is too short for the measure gives NaN, with one `RuntimeWarning`.
  `refined_composite_multiscale` leaves out only the shifts that are too short, so it still gives a value while at
  least one shift at that scale is long enough.
- `multiscale_entropy`, `composite_multiscale_entropy`, `rcmse` and `multiscale_permutation_entropy` remain as
  shortcuts for these functions with sample and permutation entropy.

## Supported measures

| Measure | Function | Unit | Reference |
|---|---|---|---|
| Shannon entropy | `shannon_entropy` | bits | [Shannon (1948)](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) |
| Sample entropy | `sample_entropy` | nats | [Richman & Moorman (2000)](https://doi.org/10.1152/ajpheart.2000.278.6.H2039) |
| Multiscale entropy (MSE) | `multiscale_entropy` | nats | [Costa et al. (2002)](https://doi.org/10.1103/PhysRevLett.89.068102) |
| Composite MSE (CMSE) | `composite_multiscale_entropy` | nats | [Wu et al. (2013)](https://doi.org/10.3390/e15031069) |
| Refined composite MSE (RCMSE) | `rcmse` (alias `refined_composite_multiscale_entropy`) | nats | [Wu et al. (2014)](https://doi.org/10.1016/j.physleta.2014.03.034) |
| Permutation entropy | `permutation_entropy` | bits, or 0-1 with `normalize=True` | [Bandt & Pompe (2002)](https://doi.org/10.1103/PhysRevLett.88.174102) |
| Multiscale permutation entropy | `multiscale_permutation_entropy` | bits | [Morabito et al. (2012)](https://doi.org/10.3390/e14071186) |
| Weighted permutation entropy | `weighted_permutation_entropy` | bits, or 0-1 with `normalize=True` | [Fadlallah et al. (2013)](https://doi.org/10.1103/PhysRevE.87.022911) |
| Fuzzy entropy | `fuzzy_entropy` | nats | [Chen et al. (2007)](https://doi.org/10.1109/TNSRE.2007.897025) |
| Dispersion entropy | `dispersion_entropy` | bits, or 0-1 with `normalize=True` | [Rostaghi & Azami (2016)](https://doi.org/10.1109/LSP.2016.2542881) |

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

### Migrating from 2.x

The old names still work in 3.x with a `DeprecationWarning` and will be removed in 4.0.

| Function | 2.x | 3.x |
|---|---|---|
| `sample_entropy`, `multiscale_entropy`, `composite_multiscale_entropy`, `rcmse` | `sample_length=3` (or positional `3`) | `m=2` (`m = sample_length - 1`) |
| `multiscale_entropy` | `maxscale=` | `scale=` |
| `fuzzy_entropy` | `sample_length=`, `n=` | `m=`, `exponent=` (same values) |
| `multiscale_permutation_entropy` | `m=` | `order=` (same value) |
| `time_delay_embedding` | `embedding_dimension=` | `order=` |
| `util_pattern_space(x, lag, dim)` | | `time_delay_embedding(np.asarray(x, dtype=float), dim, lag).copy()` |

## Performance

pyentrp 2.3.0, Python 3.13.18, NumPy 2.2.6, Apple M4 Pro. White noise, best of up to 3 timeit repeats (calls over 2s are timed once), each benchmark in its own single-threaded process.

| Function | N = 1,000 | N = 10,000 | N = 100,000 |
|---|---|---|---|
| `shannon_entropy (10 quantile bins)` | 10 µs | 40 µs | 319 µs |
| `permutation_entropy (order 3)` | 8 µs | 32 µs | 256 µs |
| `permutation_entropy (order 6)` | 21 µs | 79 µs | 639 µs |
| `weighted_permutation_entropy (order 3)` | 12 µs | 53 µs | 441 µs |
| `multiscale_permutation_entropy (order 3, 5 scales)` | 52 µs | 172 µs | 1.3 ms |
| `dispersion_entropy (6 classes, order 3)` | 29 µs | 90 µs | 619 µs |
| `composite_multiscale permutation_entropy (order 3, scale 5)` | 153 µs | 469 µs | 3.5 ms |
| `refined_composite_multiscale dispersion_entropy (6 classes, order 3, scale 5)` | 251 µs | 742 µs | 5.3 ms |
| `sample_entropy (m=2)` | 656 µs | 43.3 ms | 2.9 s |
| `multiscale_entropy (m=2, 5 scales)` | 1.2 ms | 67.8 ms | 4.3 s |
| `composite_multiscale_entropy (m=2, scale 5)` | 2.1 ms | 111.2 ms | 6.7 s |
| `rcmse (m=2, scale 5)` | 2.1 ms | 111.8 ms | 6.9 s |
| `fuzzy_entropy (m=2)` | 4.2 ms | 351.4 ms | 29.8 s |

## Tests

Run the test suite from the repository root:

```bash
python -m unittest discover
```

The tests cover known values and input validation, constant, sinusoidal and noisy signals, mathematical equivalences,
and brute-force or manually counted reference cases for template matching, RCMSE, fuzzy entropy and dispersion entropy.

## Citing

If pyentrp contributes to your research, please cite it. GitHub's "Cite this repository" button reads
[`CITATION.cff`](CITATION.cff), or use:

```bibtex
@software{pyentrp,
  author  = {Donets, Nikolay},
  title   = {pyEntropy (pyentrp): entropy measures for time series in Python},
  version = {2.3.0},
  year    = {2026},
  doi     = {10.5281/zenodo.17727789},
  url     = {https://github.com/nikdon/pyEntropy}
}
```

Please also cite the original paper of each measure you use (links in the table above).

## Acknowledgements

Raphael Vallat sped up permutation entropy and added base-2 logarithms and normalization (2018).
Samuel Dotson contributed weighted permutation entropy (2021).

## Contributing

Bug reports, fixes and new measures are welcome: open an [issue](https://github.com/nikdon/pyEntropy/issues) or a
pull request.

## License

Apache-2.0. See [LICENSE](LICENSE).
