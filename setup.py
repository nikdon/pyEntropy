from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyentrp",
    version="0.8.4",
    description="Functions on top of NumPy for computing different types of entropy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikdon/pyEntropy",
    author="Nikolay Donets",
    author_email="nd@donets.org",
    maintainer="Nikolay Donets",
    maintainer_email="nd@donets.org",
    license="Apache-2.0",
    packages=["pyentrp"],
    test_suite="tests.test_entropy",
    keywords=[
        "python",
        "entropy",
        "sample entropy",
        "multiscale entropy",
        "permutation entropy",
        "composite multiscale entropy",
        "multiscale permutation entropy",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
