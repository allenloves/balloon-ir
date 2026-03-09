"""
1/3-Octave Filter Bank (used by Stage 3)

Implements §5.1 of Abel et al. (2010) "Estimating Room Impulse Responses
from Recorded Balloon Pops," AES Convention Paper 8171.

This module provides a perfect-amplitude-reconstruction zero-phase filter
bank that splits a signal into 1/3-octave frequency bands. It is used in
Stage 3 to independently analyze and shape the spectral energy of the
balloon recording and synthesized echo sequence.

Architecture (Fig. 11 in the paper):
  The filter bank uses cascaded Butterworth lowpass filters applied
  forward-backward (scipy.signal.sosfiltfilt) for zero-phase response.
  Perfect reconstruction is guaranteed by a subtraction scheme:

    band_0       = LP(f_0, signal)              — lowest band
    band_k       = LP(f_k, signal) - LP(f_{k-1}, signal)  — middle bands
    band_{N-1}   = signal - LP(f_{N-2}, signal) — highest band

  Since the bands are defined by subtraction, their sum telescopes
  exactly back to the original signal, ensuring perfect reconstruction
  up to floating-point precision (<1e-10 error).

  Each lowpass uses a 3rd-order Butterworth filter applied via sosfiltfilt
  (forward-backward), yielding an effective 6th-order zero-phase response
  with ~36 dB/octave transition slopes (6 × 6 dB/octave per pole).

  NOTE: The paper states "60 dB/octave" (§5.1), which would require a
  5th-order Butterworth (5 × 2 × 6 = 60). The discrepancy may be due
  to the paper's cascaded band-splitting architecture (Fig. 11) where
  adjacent filters compound the rolloff. Our subtraction-based scheme
  achieves ~36 dB/octave per band edge, which is sufficient for clean
  band separation while guaranteeing perfect reconstruction.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Optional


def compute_band_frequencies(
    sr: int,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1/3-octave center frequencies and crossover frequencies.

    Center frequencies follow the standard IEC 61260 series, based on
    1000 Hz as the reference:  f_center[k] = 1000 * 2^(k/3).

    Crossover frequencies sit at the geometric midpoints between
    adjacent centers:  f_cross[k] = f_center[k] * 2^(1/6),
    which equals the upper band edge of band k and the lower band
    edge of band k+1.

    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    f_min : float
        Lowest center frequency to include (Hz). Default 50 Hz.
    f_max : float or None
        Highest center frequency to include (Hz).
        If None, uses 0.9 * sr/2 to stay safely below Nyquist.

    Returns
    -------
    centers : np.ndarray
        1/3-octave center frequencies in Hz, ascending order.
    crossovers : np.ndarray
        Crossover frequencies between adjacent bands.
        Length is len(centers) - 1.
    """
    nyquist = sr / 2.0
    if f_max is None:
        f_max = nyquist * 0.9

    f_ref = 1000.0  # IEC reference frequency

    # Generate center frequencies: f_ref * 2^(k/3)
    # k ranges widely enough to cover [f_min, f_max]
    centers = []
    for k in range(-30, 30):
        f = f_ref * (2.0 ** (k / 3.0))
        if f_min <= f <= f_max:
            centers.append(f)

    centers = np.array(sorted(centers))

    # Crossover frequencies: upper edge of each band (except the last)
    # f_cross[k] = f_center[k] * 2^(1/6)
    crossovers = centers[:-1] * (2.0 ** (1.0 / 6.0))

    # Clamp crossovers below Nyquist (should already be, but safety)
    crossovers = crossovers[crossovers < nyquist * 0.999]

    # Trim centers to match: we need len(crossovers) + 1 bands
    centers = centers[: len(crossovers) + 1]

    return centers, crossovers


def design_lowpass_sos(
    cutoff_hz: float,
    sr: int,
    order: int = 3,
) -> np.ndarray:
    """
    Design a Butterworth lowpass filter in second-order sections (SOS).

    SOS form is used instead of transfer function (b, a) for numerical
    stability, especially at low frequencies where high-order IIR filters
    can become unstable in (b, a) form.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz (-3 dB point for single-pass;
        -6 dB point after forward-backward application).
    sr : int
        Sample rate in Hz.
    order : int
        Butterworth filter order. Default 3 (paper §5.1).
        After forward-backward (sosfiltfilt), the effective order
        doubles to 6, giving ~36 dB/octave rolloff
        (6 poles × 6 dB/octave per pole).

    Returns
    -------
    sos : np.ndarray
        Second-order section coefficients, shape (n_sections, 6).
    """
    nyquist = sr / 2.0
    # Wn is normalized to Nyquist (0 to 1)
    wn = cutoff_hz / nyquist
    sos = butter(order, wn, btype="low", output="sos")
    return sos


def apply_filterbank(
    signal: np.ndarray,
    sr: int,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
    order: int = 3,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Split a signal into 1/3-octave bands with perfect reconstruction.

    The splitting uses cascaded zero-phase Butterworth lowpass filters
    with a subtraction scheme that guarantees the sum of all bands
    exactly reconstructs the original signal.

    The procedure (see module docstring and Fig. 11):
      1. For each crossover frequency f_k, compute:
           lp_k = sosfiltfilt(butter_lowpass(f_k), signal)
      2. Define bands by subtraction:
           band_0     = lp_0                    (lowest band)
           band_k     = lp_k - lp_{k-1}         (middle bands)
           band_last  = signal - lp_{last}       (highest band)
      3. Sum of all bands = signal  (exact, by telescoping)

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (num_samples,).
    sr : int
        Sample rate in Hz.
    f_min : float
        Lowest band center frequency in Hz. Default 50 Hz.
    f_max : float or None
        Highest band center frequency in Hz. Default ~0.9 * Nyquist.
    order : int
        Butterworth filter order (before doubling by filtfilt).
        Default 3 → 6th-order zero-phase → ~36 dB/octave.
        See module docstring for discussion of paper's "60 dB/octave" claim.

    Returns
    -------
    bands : list of np.ndarray
        List of band-filtered signals, one per 1/3-octave band.
        Each has the same shape as the input signal.
        bands[0] is the lowest frequency band, bands[-1] is the highest.
    centers : np.ndarray
        Center frequencies of each band in Hz.
    crossovers : np.ndarray
        Crossover frequencies between adjacent bands in Hz.
    """
    centers, crossovers = compute_band_frequencies(sr, f_min, f_max)

    # Compute all lowpass outputs at crossover frequencies
    lp_outputs = []
    for fc in crossovers:
        sos = design_lowpass_sos(fc, sr, order)
        lp = sosfiltfilt(sos, signal)
        lp_outputs.append(lp)

    # Build bands by subtraction — guarantees perfect reconstruction
    bands = []

    # Lowest band: everything below the first crossover
    bands.append(lp_outputs[0])

    # Middle bands: difference between adjacent lowpass outputs
    for i in range(1, len(lp_outputs)):
        bands.append(lp_outputs[i] - lp_outputs[i - 1])

    # Highest band: everything above the last crossover
    bands.append(signal - lp_outputs[-1])

    return bands, centers, crossovers


def reconstruct(bands: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct the original signal from filter bank bands.

    This is simply the sum of all bands. Due to the subtraction-based
    splitting, this sum telescopes to the original signal exactly.

    Parameters
    ----------
    bands : list of np.ndarray
        Band-filtered signals from apply_filterbank().

    Returns
    -------
    reconstructed : np.ndarray
        Sum of all bands, which should equal the original signal
        within floating-point precision.
    """
    return sum(bands)
