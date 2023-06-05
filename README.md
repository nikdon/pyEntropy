# pyEntropy (pyEntrp)

[![codecov](https://codecov.io/gh/nikdon/pyEntropy/branch/master/graph/badge.svg)](https://codecov.io/gh/nikdon/pyEntropy)
![py38 status](https://img.shields.io/badge/python3.8-supported-green.svg)
![py39 status](https://img.shields.io/badge/python3.9-supported-green.svg)
![py310 status](https://img.shields.io/badge/python3.10-supported-green.svg)

1. [Quick start](#quick-start)
2. [Usage](#usage)
3. [Contributors and participation](#contributors-and-participation)

This is a small set of functions on top of NumPy that help to compute different types of entropy for time series analysis.

The library provides functions for computing the following types of entropy:

+ Shannon Entropy ```shannon_entropy```
+ Sample Entropy ```sample_entropy```
+ Multiscale Entropy ```multiscale_entropy```
+ Composite Multiscale Entropy ```composite_multiscale_entropy```
+ Permutation Entropy ```permutation_entropy```
+ Multiscale Permutation Entropy ```multiscale_permutation_entropy```
+ Weighted Permutation Entropy ```weighted_permutation_entropy```

## Quick start

`pip install pyentrp`

`poetry add pyentrp`

## Usage

```python
from pyentrp import entropy as ent
import numpy as np


ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
std_ts = np.std(ts)
sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
```

## Contributors and participation

* [Nikolay Donets](https://github.com/nikdon)
* [Jakob Dreyer](https://github.com/jakobdreyer)
* [Raphael Vallat](https://github.com/raphaelvallat)
* [Christopher Schölzel](https://github.com/CSchoel)
* [Sam Dotson](https://github.com/samgdotson)

Contributions are very welcome, documentation improvements/corrections, bug reports, even feature requests :)
