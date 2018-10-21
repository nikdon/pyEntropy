# pyEntropy (pyEntrp)

[![pypi](https://img.shields.io/badge/pypi-0.5.0-green.svg)](https://pypi.python.org/pypi/pyentrp/0.5.0)
[![Build Status](https://travis-ci.org/nikdon/pyEntropy.svg?branch=master)](https://travis-ci.org/nikdon/pyEntropy)
[![codecov](https://codecov.io/gh/nikdon/pyEntropy/branch/master/graph/badge.svg)](https://codecov.io/gh/nikdon/pyEntropy)
![py27 status](https://img.shields.io/badge/python2.7-supported-green.svg)
![py36 status](https://img.shields.io/badge/python3.6.1-supported-green.svg)

1. [Quick start](#quick-start)
2. [Usage](#usage)
3. [Contributors and participation](#contribution-and-participation)

This is a small set of functions on top of NumPy that help to compute different types of entropy for time series analysis.

Currently available:

+ Shannon Entropy ```shannon_entropy```
+ Sample Entropy ```sample_entropy```
+ Multiscale Entropy ```multiscale_entropy```
+ Composite Multiscale Entropy ```composite_multiscale_entropy```
+ Permutation Entropy ```permutation_entropy```
+ Multiscale Permutation Entropy ```multiscale_permutation_entropy```

## Quick start

`pip install pyentrp`

## Usage

```python
from pyentrp import entropy as ent
import numpy as np


ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
std_ts = np.std(ts)
sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
```

## Requirements:

+ ```>numpy-1.7.0```

## Contributors and participation

* [Nikolay Donets](https://github.com/nikdon)
* [Jakob Dreyer](https://github.com/jakobdreyer)
* [Raphael Vallat](https://github.com/raphaelvallat)

Contributions are very welcome, documentation improvements/corrections, bug reports, even feature requests :)

[![Analytics](https://ga-beacon.appspot.com/UA-91956314-1/pyEntropy/readme?pixel)](https://github.com/igrigorik/ga-beacon)
