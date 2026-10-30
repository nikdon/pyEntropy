"""Spectral and decomposition measures: spectral, singular value decomposition and wavelet entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._validation import check_embedding_length, check_integer, validate_series

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Daubechies decomposition low-pass filters (orthonormal), db1 = Haar
DAUBECHIES = {
    "db1": (
        0.7071067811865476,
        0.7071067811865476,
    ),
    "db2": (
        -0.12940952255126037,
        0.2241438680420134,
        0.8365163037378079,
        0.48296291314453416,
    ),
    "db3": (
        0.03522629188570953,
        -0.08544127388202666,
        -0.13501102001025458,
        0.45987750211849154,
        0.8068915093110925,
        0.33267055295008263,
    ),
    "db4": (
        -0.010597401785069032,
        0.0328830116668852,
        0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
        0.6308807679298589,
        0.7148465705529157,
        0.2303778133088965,
    ),
    "db5": (
        0.0033357252854737712,
        -0.012580751999081999,
        -0.006241490212798274,
        0.07757149384004572,
        -0.032244869584638375,
        -0.24229488706638203,
        0.13842814590132074,
        0.7243085284377729,
        0.6038292697971896,
        0.16010239797419293,
    ),
    "db6": (
        -0.0010773010853084796,
        0.004777257510945511,
        0.0005538422011614961,
        -0.03158203931748603,
        0.027522865530305727,
        0.09750160558732304,
        -0.12976686756726194,
        -0.22626469396543983,
        0.31525035170919763,
        0.7511339080210954,
        0.49462389039845306,
        0.11154074335010947,
    ),
    "db7": (
        0.00035371379997452024,
        -0.0018016407040474908,
        0.0004295779729213665,
        0.01255099855609984,
        -0.01657454163066688,
        -0.03802993693501441,
        0.08061260915108308,
        0.07130921926683026,
        -0.22403618499387498,
        -0.14390600392856498,
        0.4697822874051931,
        0.7291320908462351,
        0.3965393194819173,
        0.07785205408500918,
    ),
    "db8": (
        -0.00011747678412476953,
        0.0006754494064505693,
        -0.00039174037337694705,
        -0.004870352993451574,
        0.008746094047405777,
        0.013981027917398282,
        -0.044088253930794755,
        -0.017369301001807547,
        0.12874742662047847,
        0.0004724845739132828,
        -0.2840155429615469,
        -0.015829105256349306,
        0.5853546836542067,
        0.6756307362972898,
        0.31287159091429995,
        0.05441584224310401,
    ),
    "db9": (
        3.93473203162716e-05,
        -0.0002519631889427101,
        0.00023038576352319597,
        0.0018476468830562265,
        -0.00428150368246343,
        -0.004723204757751397,
        0.022361662123679096,
        0.00025094711483145197,
        -0.06763282906132997,
        0.03072568147933338,
        0.14854074933810638,
        -0.09684078322297646,
        -0.2932737832791749,
        0.13319738582500756,
        0.6572880780513005,
        0.6048231236901112,
        0.24383467461259034,
        0.038077947363878345,
    ),
    "db10": (
        -1.3264202894521244e-05,
        9.358867032006959e-05,
        -0.00011646685512928545,
        -0.0006858566949597116,
        0.001992405295185056,
        0.001395351747052901,
        -0.010733175483330575,
        0.0036065535669561697,
        0.033212674059341,
        -0.029457536821875813,
        -0.07139414716639708,
        0.09305736460357235,
        0.12736934033579325,
        -0.19594627437737705,
        -0.24984642432731538,
        0.2811723436605775,
        0.6884590394536035,
        0.5272011889317256,
        0.1881768000776915,
        0.026670057900555554,
    ),
}


def check_bool(name, value):
    """Validate a boolean flag (NumPy booleans included)."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")


def entropy_of_spectrum(power, normalize):
    """Return the Shannon entropy in bits of a non-negative spectrum, normalized by log2 of its length if asked."""
    total = float(np.sum(power))
    if total <= 0.0 or len(power) < 2:
        return 0.0
    p = power[power > 0] / total
    h = float(-np.dot(p, np.log2(p)))
    if normalize:
        h /= math.log2(len(power))
    return h if h > 0.0 else 0.0


def one_sided_density(segments, window, sf):
    """Return the one-sided power spectral density of detrended segments (rows), averaged over the rows.

    Matches scipy.signal.periodogram and welch with detrend="constant" and scaling="density".

    """
    n = segments.shape[1]
    centered = segments - segments.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    power *= 1.0 / (sf * float(np.dot(window, window)))
    # every frequency but 0 (and the Nyquist frequency for even n) stands for its negative twin too
    power[:, 1 : None if n % 2 else -1] *= 2.0
    return power.mean(axis=0), np.fft.rfftfreq(n, 1.0 / sf)


def spectral_entropy(
    time_series: ArrayLike,
    sf: float = 1.0,
    method: Literal["periodogram", "welch"] = "periodogram",
    nperseg: int | None = None,
    normalize: bool = False,
    *,
    band: tuple[float, float] | None = None,
) -> float:
    """Calculate spectral entropy: the Shannon entropy of the normalized power spectral density.

    A flat spectrum (white noise) gives the largest value and a single sinusoid the smallest. The
    spectrum is the periodogram or Welch's estimate, computed as scipy.signal.periodogram and
    scipy.signal.welch do by default (constant detrending, one-sided density, Hann window for Welch).

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values, with at least two samples.
    sf : float, default=1.0
        Sampling frequency, in Hz. Only used to express `band`; the entropy itself does not depend on it.
    method : {"periodogram", "welch"}, default="periodogram"
        Spectrum estimate.
    nperseg : int, optional
        Segment length for Welch's method (half-overlapping Hann windows). Defaults to 256, or the
        series length if shorter.
    normalize : bool, default=False
        Divide by log2 of the number of frequencies, so the result is between 0 and 1.
    band : tuple of float, optional
        Keyword-only. Keep only the frequencies f with band[0] <= f <= band[1] (in the units of sf).

    Returns
    -------
    spec : float
        Spectral entropy in bits (or normalized). 0.0 when the spectrum has no power, e.g. a constant
        series.

    Raises
    ------
    TypeError
        If nperseg is not an integer (booleans excluded), or method is not a string.
    ValueError
        If sf is not positive, method is unknown, nperseg < 2, band is not an increasing pair within
        [0, sf / 2] or holds fewer than two frequencies, or time_series is not 1D, contains non-finite
        values or has fewer than two samples.

    References
    ----------
    .. [1] Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A., Katsuda, Y., & Hirano,
           M. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum.
           Electroencephalography and Clinical Neurophysiology, 79(3), 204-210.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1000) / 100
    >>> round(spectral_entropy(np.sin(2 * np.pi * 5 * t), sf=100, normalize=True), 3)
    0.0

    """
    x = validate_series(time_series)
    if not (sf > 0 and np.isfinite(sf)):
        raise ValueError("sf must be a positive finite number.")
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    if method not in ("periodogram", "welch"):
        raise ValueError("method must be 'periodogram' or 'welch'.")
    check_bool("normalize", normalize)
    if len(x) < 2:
        raise ValueError("time_series must have at least two samples.")

    if method == "periodogram":
        power, freqs = one_sided_density(x[None, :], np.ones(len(x)), sf)
    else:
        if nperseg is None:
            nperseg = min(256, len(x))
        check_integer("nperseg", nperseg, 2)
        nperseg = min(nperseg, len(x))
        step = nperseg - nperseg // 2
        n_seg = (len(x) - nperseg // 2) // step
        starts = np.arange(n_seg) * step
        segments = x[starts[:, None] + np.arange(nperseg)]
        # periodic Hann window, as scipy.signal.get_window("hann", nperseg)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg)
        power, freqs = one_sided_density(segments, window, sf)

    if band is not None:
        low, high = band
        if not (0 <= low < high <= sf / 2):
            raise ValueError("band must be (low, high) with 0 <= low < high <= sf / 2.")
        keep = (freqs >= low) & (freqs <= high)
        if np.count_nonzero(keep) < 2:
            raise ValueError("band holds fewer than two frequencies of the spectrum.")
        power = power[keep]
    return entropy_of_spectrum(power, normalize)


def svd_entropy(time_series: ArrayLike, order: int = 3, delay: int = 1, normalize: bool = False) -> float:
    """Calculate singular value decomposition (SVD) entropy.

    The Shannon entropy of the normalized singular values of the delay-embedding matrix: few dominant
    singular values (a low-dimensional, regular signal) give a low value.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    order : int, default=3
        Embedding dimension (number of columns of the embedding matrix, >= 1).
    delay : int, default=1
        Time delay between the columns.
    normalize : bool, default=False
        Divide by log2(order), so the result is between 0 and 1.

    Returns
    -------
    svd_e : float
        SVD entropy in bits (or normalized).

    Raises
    ------
    TypeError
        If order or delay is not an integer (booleans excluded), or normalize is not a bool.
    ValueError
        If order or delay is below 1, or time_series is not 1D, contains non-finite values or has fewer
        than (order - 1) * delay + 1 samples.

    References
    ----------
    .. [1] Roberts, S. J., Penny, W., & Rezek, I. (1999). Temporal and spatial complexity measures for
           electroencephalogram based brain-computer interfacing. Medical & Biological Engineering &
           Computing, 37(1), 93-98.

    Examples
    --------
    >>> import numpy as np
    >>> round(svd_entropy(np.sin(np.arange(500) / 5), order=3), 3)
    0.585

    """
    x = validate_series(time_series)
    check_integer("order", order, 1)
    check_integer("delay", delay, 1)
    check_bool("normalize", normalize)
    check_embedding_length(x, order, delay)
    n_win = len(x) - (order - 1) * delay
    embedded = np.stack([x[k * delay : k * delay + n_win] for k in range(order)], axis=1)
    singular = np.linalg.svd(embedded, compute_uv=False)
    # singular values below the rank tolerance of np.linalg.matrix_rank are rounding noise
    singular[singular <= singular[0] * max(embedded.shape) * np.finfo(float).eps] = 0.0
    return entropy_of_spectrum(singular, normalize and order > 1)


def dwt_step(x, low, high):
    """One level of the periodized discrete wavelet transform (pywt.dwt with mode="periodization")."""
    if len(x) % 2:
        x = np.append(x, x[-1])  # an odd length is extended by its last sample
    n, taps = len(x), len(low)
    # circular convolution, kept at every second sample, offset by half the filter length
    positions = 2 * np.arange(n // 2) + taps // 2
    approx, detail = np.zeros(n // 2), np.zeros(n // 2)
    for j in range(taps):
        shifted = x[(positions - j) % n]
        approx += low[j] * shifted
        detail += high[j] * shifted
    return approx, detail


def wavelet_entropy(
    time_series: ArrayLike,
    wavelet: Literal["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10"] = "db4",
    levels: int | None = None,
    normalize: bool = False,
) -> float:
    """Calculate wavelet entropy: the Shannon entropy of the relative wavelet energy of each level.

    The series is decomposed with the periodized orthogonal discrete wavelet transform. The energy of
    the detail coefficients of each level, relative to the total detail energy, gives a distribution
    over levels (frequency bands), and its entropy is low when one band dominates.

    Parameters
    ----------
    time_series : np.ndarray | list
        1D series of finite values.
    wavelet : {"db1", ..., "db10"}, default="db4"
        Daubechies wavelet; db1 is the Haar wavelet.
    levels : int, optional
        Number of decomposition levels (>= 2). Defaults to the largest useful number,
        floor(log2(n / (filter length - 1))), as pywt.dwt_max_level.
    normalize : bool, default=False
        Divide by log2(levels), so the result is between 0 and 1.

    Returns
    -------
    wavelet_e : float
        Wavelet entropy in bits (or normalized). 0.0 when the detail coefficients have no energy.

    Notes
    -----
    The final approximation coefficients are left out of the energy distribution, as in Rosso et al.,
    whose relative energies run over the resolution (detail) levels.

    Raises
    ------
    TypeError
        If levels is not an integer (booleans excluded), or wavelet is not a string.
    ValueError
        If wavelet is unknown, levels is below 2 or above the largest useful number, or time_series is
        not 1D, contains non-finite values or is too short for two levels.

    References
    ----------
    .. [1] Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A., Schurmann, M., & Basar, E.
           (2001). Wavelet entropy: A new tool for analysis of short duration brain electrical signals.
           Journal of Neuroscience Methods, 105(1), 65-75.

    Examples
    --------
    >>> import numpy as np
    >>> round(wavelet_entropy(np.random.default_rng(0).standard_normal(1024)), 3)
    1.926

    """
    x = validate_series(time_series)
    if not isinstance(wavelet, str):
        raise TypeError(f"wavelet must be a string, got {type(wavelet).__name__}.")
    if wavelet not in DAUBECHIES:
        raise ValueError(f"wavelet must be one of {', '.join(DAUBECHIES)}.")
    check_bool("normalize", normalize)
    low = np.array(DAUBECHIES[wavelet])
    taps = len(low)
    # the decomposition high-pass filter is the alternating flip of the low-pass one
    high = np.array([(-1) ** (k + 1) * low[taps - 1 - k] for k in range(taps)])
    max_levels = math.floor(math.log2(len(x) / (taps - 1))) if len(x) >= taps - 1 else 0
    if levels is None:
        levels = max_levels
    else:
        check_integer("levels", levels, 2)
    if max_levels < 2 or levels > max_levels:
        raise ValueError(f"time_series is too short for {levels} levels with {wavelet} (at most {max_levels} levels).")

    energies = np.zeros(levels)
    approx = x
    for j in range(levels):
        approx, detail = dwt_step(approx, low, high)
        energies[j] = float(np.dot(detail, detail))
    return entropy_of_spectrum(energies, normalize)
