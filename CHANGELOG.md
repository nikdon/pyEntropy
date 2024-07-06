# Change Log

## 1.0.0 (2024-07-06)

### BREAKING CHANGE

- remove deprecated function `_embed`

### Refactor

- update ruff lint rules and fix tests
- use `math.factorial` instead of `np.math.factorial`

## 0.9.0 (2023-12-20)

### BREAKING CHANGE

- remove support of python 3.8

## 0.8.5 (2023-09-27)

- build: add support for Python 3.11
- build: add support for Python 3.12

## 0.8.4 (2023-06-18)

### Refactor

- **entropy.py**: refactoring of all functions and docstrings

## 0.8.3 (2023-06-09)

### Fix

- use 1d vector for cmse output (#21)

### Perf

- **entropy.py**: improvements to calculation of shannon_entropy

## 0.8.2 (2023-06-05)

## 0.8.1 (2023-06-05)

## 0.8.0 (2023-06-04)

### Refactor

- major cleanup of the repository with new ci/cd and updates to build and dependencies

## 0.7.1

- Minor changes to `setup.py` to make the project's page nice

## 0.7.0

- [#19](https://github.com/nikdon/pyEntropy/pull/19) - `weighted_permutation_entropy`

## 0.6.0

- [#15](https://github.com/nikdon/pyEntropy/pull/15) - Sample entropy ignores last `M` values (thanks @CSchoel)

## 0.5.0

Due to changes to `permutation_entropy` this release might break compatibility with previous versions.

This release introduces improvements to `permutation_entropy` (thanks @raphaelvallat)

- Changed signature of the `def permutation_entropy(time_series, m, delay)` to `def permutation_entropy(time_series, order=3, delay=1, normalize=False)`
- Increased speed of the `permutation_entropy`
- Changed the log base from 10 to 2 in `permutation_entropy` (as per Band and Pompe 2002)
- Added normalization to the `permutation_entropy`
- Default value for the `permutation_entropy` are changed
- More tests
- Cleanup of docs with PEP8 and NumpyDoc
