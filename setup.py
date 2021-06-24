from distutils.core import setup

long_description = """pyEntropy (pyEntrp)
===================

|pypi| |Build Status| |codecov| |py27 status| |py36 status| |image1|
|image2| |image3|

1. `Quick start`_
2. `Usage`_
3. `Contributors and participation`_

This is a small set of functions on top of NumPy that help to compute
different types of entropy for time series analysis.

-  Shannon Entropy ``shannon_entropy``
-  Sample Entropy ``sample_entropy``
-  Multiscale Entropy ``multiscale_entropy``
-  Composite Multiscale Entropy ``composite_multiscale_entropy``
-  Permutation Entropy ``permutation_entropy``
-  Multiscale Permutation Entropy ``multiscale_permutation_entropy``
-  Weighted Permutation Entropy ``weighted_permutation_entropy``

Quick start
-----------

``pip install pyentrp``

Usage
-----

.. code:: python

   from pyentrp import entropy as ent
   import numpy as np


   ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3]
   std_ts = np.std(ts)
   sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)

Requirements:
-------------

-  ``>numpy-1.7.0``

Contributors and participation
------------------------------

-  `Nikolay Donets`_
-  `Jakob Dreyer`_
-  `Raphael Vallat`_
-  `Christopher Schölzel`_
-  `Sam Dotson`_

Contributions are very welcome, documentation improvements/corrections,
bug reports, even feature requests :)

.. _Quick start: #quick-start
.. _Usage: #usage
.. _Contributors and participation: #contribution-and-participation
.. _Nikolay Donets: https://github.com/nikdon
.. _Jakob Dreyer: https://github.com/jakobdreyer
.. _Raphael Vallat: https://github.com/raphaelvallat
.. _Christopher Schölzel: https://github.com/CSchoel
.. _Sam Dotson: https://github.com/samgdotson

.. |pypi| image:: https://img.shields.io/badge/pypi-0.7.1-green.svg
   :target: https://pypi.python.org/pypi/pyentrp/0.7.1
.. |Build Status| image:: https://travis-ci.org/nikdon/pyEntropy.svg?branch=master
   :target: https://travis-ci.org/nikdon/pyEntropy
.. |codecov| image:: https://codecov.io/gh/nikdon/pyEntropy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nikdon/pyEntropy
.. |py27 status| image:: https://img.shields.io/badge/python2.7-supported-green.svg
.. |py36 status| image:: https://img.shields.io/badge/python3.6-supported-green.svg
.. |image1| image:: https://img.shields.io/badge/python3.7-supported-green.svg
.. |image2| image:: https://img.shields.io/badge/python3.8-supported-green.svg
.. |image3| image:: https://img.shields.io/badge/python3.9-supported-green.svg
"""

setup(
    name='pyentrp',
    version='0.7.1',
    description='Functions on top of NumPy for computing different types of entropy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nikdon/pyEntropy',
    download_url='https://github.com/nikdon/pyEntropy/archive/0.7.1.tar.gz',
    author='Nikolay Donets',
    author_email='nd.startup@gmail.com',
    maintainer='Nikolay Donets',
    maintainer_email='nd.startup@gmail.com',
    license='Apache-2.0',
    packages=['pyentrp'],

    install_requires=[
        'numpy>=1.7.0 ',
    ],
    test_suite="tests.test_entropy",

    keywords=[
        'python',
        'entropy',
        'sample entropy',
        'multiscale entropy',
        'permutation entropy',
        'composite multiscale entropy',
        'multiscale permutation entropy'
    ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
