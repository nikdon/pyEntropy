# pyEntropy (pyEntrp)

![py38 status](https://img.shields.io/badge/python3.8-supported-green.svg)
![py39 status](https://img.shields.io/badge/python3.9-supported-green.svg)
![py310 status](https://img.shields.io/badge/python3.10-supported-green.svg)
![py311 status](https://img.shields.io/badge/python3.11-supported-green.svg)

1. [Quick start](#quick-start)
2. [Usage](#usage)
3. [Contributors and participation](#contributors-and-participation)

pyEntropy is a lightweight library built on top of NumPy
that provides functions for computing various types of entropy for time series analysis.

The library currently supports the following types of entropy computation:

+ Shannon Entropy ```shannon_entropy```
+ Sample Entropy ```sample_entropy```
+ Multiscale Entropy ```multiscale_entropy```
+ Composite Multiscale Entropy ```composite_multiscale_entropy```
+ Permutation Entropy ```permutation_entropy```
+ Multiscale Permutation Entropy ```multiscale_permutation_entropy```
+ Weighted Permutation Entropy ```weighted_permutation_entropy```

## Quick start

Install [pyEntropy](https://github.com/nikdon/pyEntropy) using pip:

```
pip install pyentrp
```

Install [pyEntropy](https://github.com/nikdon/pyEntropy) using poetry:

```
poetry add pyentrp
```

## Usage

```python
from pyentrp import entropy as ent
import numpy as np

ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
std_ts = np.std(ts)
sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
```

## Contributors and participation

[pyEntropy](https://github.com/nikdon/pyEntropy) is an open-source project, and contributions are highly encouraged.
If you would like to contribute, you can:

- [Fork the repository](https://github.com/nikdon/pyEntropy/fork) and submit pull requests with your improvements, bug
  fixes, or new features.
- Report any issues or bugs you encounter on the [issue tracker](https://github.com/nikdon/pyEntropy/issues).
- Help improve the documentation by
  submitting [documentation improvements or corrections](https://github.com/nikdon/pyEntropy/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation).
- Participate in discussions and share your ideas.

The following contributors have made significant contributions to pyEntropy:

* [Nikolay Donets](https://github.com/nikdon)
* [Jakob Dreyer](https://github.com/jakobdreyer)
* [Raphael Vallat](https://github.com/raphaelvallat)
* [Christopher Schölzel](https://github.com/CSchoel)
* [Sam Dotson](https://github.com/samgdotson)

Contributions are very welcome, documentation improvements/corrections, bug reports, even feature requests :)

If you find [pyEntropy](https://github.com/nikdon/pyEntropy) useful, please consider giving it a star.

Your support is greatly appreciated!
