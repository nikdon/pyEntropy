from distutils.core import setup

setup(
    name='pyentrp',
    version='0.1.0',
    description='Functions on top of NumPy for computing different types of entropy',
    long_description=open("README.md").read(),
    url='https://github.com/nikdon/pyEntropy',
    download_url='github.com/nikdon/pyEntropy/tarball/0.1.0.tar.gz',
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

    keywords=['entropy', 'python', 'sample entropy', 'multiscale entropy', 'permutation entropy',
              'composite multiscale entropy'],

    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
