"""Check pyentrp against EntropyHub, NeuroKit2 and antropy where the definitions are the same.

Usage (the reference libraries are not dependencies of pyentrp)::

    pip install EntropyHub neurokit2 antropy
    python -m benchmarks.crosscheck

Every check must agree to 1e-12. Known differences, all from conventions rather than errors:

- Composite MSE / MPE vs EntropyHub: EntropyHub cuts every shifted series to the same length; pyentrp keeps every
  complete block.
- Multiscale dispersion entropy vs EntropyHub: EntropyHub maps each coarse series with its own mean and std; pyentrp
  uses those of the original series, as Azami's reference code does.
- Refined fuzzy entropy vs EntropyHub: EntropyHub implements RMFE-sigma (coarse-graining by standard deviation).
- Composite and refined MSE vs NeuroKit2: NeuroKit2 downsamples (time-shift MSE) instead of averaging.
- Dispersion entropy vs NeuroKit2: NeuroKit2 maps classes with the sample std (ddof=1) and divides an entropy in
  bits by ln(c ** m).
- Fuzzy entropy vs NeuroKit2: NeuroKit2 counts self-matches.
"""

import sys
import warnings

import antropy as ant
import EntropyHub as EH
import neurokit2 as nk
import numpy as np

import pyentrp as pe

TOLERANCE = 1e-12


def signals():
    """Return the fixed test signals."""
    rng = np.random.default_rng(42)
    ar1 = np.zeros(1000)
    for i in range(1, 1000):
        ar1[i] = 0.9 * ar1[i - 1] + rng.standard_normal()
    return {
        "noise": rng.standard_normal(1000),
        "sine+noise": np.sin(np.linspace(0, 40 * np.pi, 1000)) + 0.3 * rng.standard_normal(1000),
        "ar1": ar1,
    }


def nk_value(result):
    """Return the value of a NeuroKit2 (value, info) result."""
    return result[0] if isinstance(result, tuple) else result


def cases(x):
    """Yield (name, pyentrp value, {library: reference function}) for one signal."""
    r = 0.2 * np.std(x)  # population std, as every library below uses
    yield (
        "SampEn m=2",
        pe.sample_entropy(x, m=2, tolerance=r)[-1],
        {
            "EntropyHub": lambda: EH.SampEn(x, m=2, r=r)[0][-1],
            "antropy": lambda: ant.sample_entropy(x, order=2),
            "NeuroKit2": lambda: nk_value(nk.entropy_sample(x, dimension=2, tolerance=r)),
        },
    )
    for order, delay in ((3, 1), (4, 2), (6, 1)):
        yield (
            f"PermEn order={order} delay={delay}",
            pe.permutation_entropy(x, order, delay),
            {
                "EntropyHub": lambda o=order, d=delay: EH.PermEn(x, m=o, tau=d, Logx=2)[0][-1],
                "antropy": lambda o=order, d=delay: ant.perm_entropy(x, order=o, delay=d),
                "NeuroKit2": lambda o=order, d=delay: nk_value(
                    nk.entropy_permutation(x, dimension=o, delay=d, corrected=False)
                ),
            },
        )
        yield (
            f"PermEn order={order} delay={delay} normalized",
            pe.permutation_entropy(x, order, delay, normalize=True),
            {"antropy": lambda o=order, d=delay: ant.perm_entropy(x, order=o, delay=d, normalize=True)},
        )
    for order in (3, 4):
        yield (
            f"WPE order={order}",
            pe.weighted_permutation_entropy(x, order, 1),
            {
                "EntropyHub": lambda o=order: EH.PermEn(x, m=o, tau=1, Logx=2, Typex="weighted")[0][-1],
                "NeuroKit2": lambda o=order: nk_value(
                    nk.entropy_permutation(x, dimension=o, delay=1, weighted=True, corrected=False)
                ),
            },
        )
    for classes, order in ((6, 2), (6, 3), (3, 2)):
        yield (
            f"DispEn c={classes} order={order}",
            pe.dispersion_entropy(x, classes, order),
            {"EntropyHub": lambda c=classes, o=order: EH.DispEn(x, m=o, tau=1, c=c, Typex="ncdf", Logx=2)[0]},
        )
    # pyentrp's exp(-(d / r) ** 2) is EntropyHub's default exp(-d ** 2 / r') with r' = r ** 2
    yield (
        "FuzzEn m=2",
        pe.fuzzy_entropy(x, m=2, tolerance=r, exponent=2),
        {"EntropyHub": lambda: EH.FuzzEn(x, m=2, r=(r**2, 2.0))[0][-1]},
    )
    yield (
        "MSE m=2, scales 1-5",
        pe.multiscale(x, pe.sample_entropy, 5, m=2, tolerance=r),
        {
            "EntropyHub": lambda: EH.MSEn(x, EH.MSobject("SampEn", m=2, r=r), Scales=5)[0],
            "NeuroKit2": lambda: nk.entropy_multiscale(
                x, scale=[1, 2, 3, 4, 5], dimension=2, tolerance=r, method="MSEn"
            )[1]["Value"],
        },
    )
    yield (
        "MPE order=3, scales 1-5",
        pe.multiscale(x, pe.permutation_entropy, 5, order=3),
        {"EntropyHub": lambda: EH.MSEn(x, EH.MSobject("PermEn", m=3, tau=1, Logx=2), Scales=5)[0]},
    )


def main():
    """Run every check and return the number of mismatches."""
    warnings.filterwarnings("ignore")
    failures = 0
    for signal, x in signals().items():
        for name, ours, refs in cases(x):
            for library, reference in refs.items():
                diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(reference(), dtype=float))))
                ok = diff <= TOLERANCE
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {name:34s} {signal:11s} {library:11s} max |diff| = {diff:.1e}")
    print(f"\n{failures} mismatches")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
