"""
Time-Frequency Energy Analysis & Synthesis (Stage 3)

Implements §5 of Abel et al. (2010) "Estimating Room Impulse Responses
from Recorded Balloon Pops," AES Convention Paper 8171.

This module shapes the spectral content of the synthetic echo sequence
(from Stage 1) to match the room's frequency-dependent decay characteristics
as measured in the balloon pop recording. The process has four steps:

  3b. Estimate smoothed energy profiles β²_k(t) and ν²_k(t) for each
      1/3-octave band of the balloon recording and echo sequence (Eq. 12-13).
  3c. Extrapolate the balloon's band energies below the noise floor so
      the synthesized IR decays naturally instead of hitting a flat floor.
  3d. Compute gain functions γ_k(t) = β_k(t)/ν_k(t) and apply them to
      the echo sequence bands, imprinting the balloon's energy envelope
      onto the synthetic pulses (Eq. 14).
  3e. Equalize the direct path arrival to be spectrally flat by computing
      per-band inverse gains α_k, then sum all shaped bands to produce
      the final IR (Eq. 16).

IMPORTANT: The filter bank (Step 3a) operates on the ORIGINAL balloon
recording b(t) and synthesized echo sequence p(t) — NOT on the integrated
signal from Stage 1. The integration was only used for NED estimation.
"""

import numpy as np
from scipy.signal.windows import hann
from typing import Optional

from core.filterbank import apply_filterbank, reconstruct


# ---------------------------------------------------------------------------
# 3b. Band Energy Estimation
# ---------------------------------------------------------------------------

def estimate_band_energy(
    band_signal: np.ndarray,
    sr: int,
    window_ms: float = 10.0,
) -> np.ndarray:
    """
    Compute smoothed energy profile for a single frequency band.

    Implements Equations (12) and (13) from Abel et al. (2010), §5.2:
        β²_k(t) = b_k(t)² * w(t)     — for balloon band k
        ν²_k(t) = p_k(t)² * w(t)     — for echo sequence band k

    where * denotes convolution and w(t) is a Hanning window with unit sum.

    The paper (§5.2) specifies "a 10ms-long Hanning window is used so as
    to reveal short-duration features in the original balloon pop band
    energies."

    Parameters
    ----------
    band_signal : np.ndarray
        Band-filtered signal b_k(t) or p_k(t), shape (num_samples,).
    sr : int
        Sample rate in Hz.
    window_ms : float
        Smoothing window length in ms. Default 10ms (paper §5.2).

    Returns
    -------
    energy : np.ndarray
        Smoothed energy profile β²_k(t) or ν²_k(t), same length as input.
        Always non-negative.
    """
    # --- Equations (12)/(13): Smoothed squared signal ---
    # Square the band signal, then convolve with a Hanning window
    # having unit sum. This gives a running weighted average of
    # the instantaneous energy (squared amplitude).
    window_len = max(1, int(sr * window_ms / 1000.0))
    if window_len % 2 == 0:
        window_len += 1  # ensure odd length for symmetric window

    # Hanning window normalized to unit sum
    w = hann(window_len, sym=True)
    w /= np.sum(w)

    squared = band_signal ** 2

    # Convolve with 'same' mode to preserve signal length
    energy = np.convolve(squared, w, mode="same")

    return energy


def estimate_all_band_energies(
    bands: list[np.ndarray],
    sr: int,
    window_ms: float = 10.0,
) -> list[np.ndarray]:
    """
    Compute smoothed energy profiles for all frequency bands.

    Parameters
    ----------
    bands : list of np.ndarray
        Band-filtered signals from apply_filterbank().
    sr : int
        Sample rate in Hz.
    window_ms : float
        Smoothing window length in ms.

    Returns
    -------
    energies : list of np.ndarray
        Smoothed energy profile for each band.
    """
    return [estimate_band_energy(b, sr, window_ms) for b in bands]


# ---------------------------------------------------------------------------
# 3c. Energy Extrapolation (Below Noise Floor)
# ---------------------------------------------------------------------------

def extrapolate_energy(
    energy: np.ndarray,
    sr: int,
    noise_floor_db: float = -40.0,
    fit_range_db: float = 10.0,
) -> np.ndarray:
    """
    Extrapolate a band energy curve below its noise floor.

    ENGINEERING DECISION: The paper (§5.2) cites Bryan & Abel [13]
    for energy extrapolation below the noise floor, but does not
    detail the algorithm. We use a simplified approach: fit a linear
    regression (in dB) to the energy curve in a region above the
    estimated noise floor, then extend that line.

    This produces a more natural decay tail instead of the energy
    hitting a flat noise floor (compare Fig. 13 upper vs. lower panels).

    Parameters
    ----------
    energy : np.ndarray
        Smoothed band energy profile β²_k(t) from Step 3b.
    sr : int
        Sample rate in Hz.
    noise_floor_db : float
        Threshold below peak energy (in dB) at which the noise floor
        is assumed to begin. Default -40 dB.
    fit_range_db : float
        Width of the region (in dB) above the noise floor used for
        fitting the linear decay slope. Default 10 dB.
        The fit region is [noise_floor_db, noise_floor_db + fit_range_db]
        relative to peak, e.g., [-40, -30] dB.

    Returns
    -------
    energy_ext : np.ndarray
        Energy profile with values below the noise floor replaced by
        the extrapolated decay. Same shape as input.
    """
    energy_ext = energy.copy()

    # Convert to dB (relative to peak)
    peak_energy = np.max(energy)
    if peak_energy <= 0:
        return energy_ext

    energy_db = 10.0 * np.log10(energy / peak_energy + 1e-30)

    # Estimate noise floor: median energy in the last 500ms
    tail_samples = min(len(energy), int(sr * 0.5))
    if tail_samples > 0:
        tail_median_db = np.median(energy_db[-tail_samples:])
    else:
        tail_median_db = noise_floor_db

    # Use the lower of specified threshold and estimated floor
    effective_floor_db = max(tail_median_db, noise_floor_db)

    # Find the time where energy first drops below the noise floor + 3dB
    floor_threshold_db = effective_floor_db + 3.0
    below_floor = np.where(energy_db < floor_threshold_db)[0]

    if len(below_floor) == 0:
        # Energy never reaches the noise floor — nothing to extrapolate
        return energy_ext

    floor_onset = below_floor[0]

    # Fit region: from (floor - fit_range_db) to floor_threshold_db
    fit_upper_db = floor_threshold_db + fit_range_db
    fit_region = np.where(
        (energy_db >= floor_threshold_db) & (energy_db <= fit_upper_db)
    )[0]

    if len(fit_region) < 10:
        # Not enough points for a reliable fit — skip extrapolation
        return energy_ext

    # Linear regression in dB domain: energy_db ≈ slope * t + intercept
    t_fit = fit_region.astype(np.float64)
    db_fit = energy_db[fit_region]
    # Use numpy polyfit (degree 1 = linear)
    slope, intercept = np.polyfit(t_fit, db_fit, 1)

    if slope >= 0:
        # Energy is not decaying in the fit region — skip
        return energy_ext

    # Extrapolate: replace everything from floor_onset onward
    t_extrap = np.arange(floor_onset, len(energy), dtype=np.float64)
    extrap_db = slope * t_extrap + intercept

    # Convert back to linear energy
    extrap_linear = peak_energy * 10.0 ** (extrap_db / 10.0)

    # Only replace where the extrapolated value is below the original
    # (don't boost energy that's already above the extrapolated line)
    for i, t in enumerate(range(floor_onset, len(energy))):
        energy_ext[t] = min(energy_ext[t], extrap_linear[i])

    return energy_ext


# ---------------------------------------------------------------------------
# 3d. Band Energy Imprinting
# ---------------------------------------------------------------------------

def compute_gain(
    balloon_energy: np.ndarray,
    echo_energy: np.ndarray,
    smoothing_samples: int = 0,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute the per-band gain function γ_k(t).

    Implements Equation (14) from Abel et al. (2010), §5.3:
        γ_k(t) = β_k(t) / ν_k(t)

    where β_k and ν_k are AMPLITUDES (square roots of the energy profiles
    from Eq. 12-13):
        γ_k(t) = sqrt(balloon_energy_k(t)) / sqrt(echo_energy_k(t))
               = sqrt(balloon_energy_k(t) / echo_energy_k(t))

    Parameters
    ----------
    balloon_energy : np.ndarray
        Smoothed energy β²_k(t) of the balloon recording's band k.
    echo_energy : np.ndarray
        Smoothed energy ν²_k(t) of the echo sequence's band k.
    smoothing_samples : int
        If > 0, apply a moving-average smoothing to γ_k(t) to avoid
        rapid gain fluctuations (see spec §11 item 6). Default 0 (no smoothing).
    epsilon : float
        Small value to prevent division by zero. Where echo energy is
        below epsilon, γ is set to 0.

    Returns
    -------
    gamma : np.ndarray
        Gain function γ_k(t). Same length as inputs.
    """
    # --- Equation (14): γ_k(t) = β_k(t) / ν_k(t) ---
    # = sqrt(β²_k(t)) / sqrt(ν²_k(t)) = sqrt(β²_k(t) / ν²_k(t))
    safe_echo = np.maximum(echo_energy, epsilon)
    gamma = np.sqrt(balloon_energy / safe_echo)

    # Zero out gain where echo energy is negligible
    gamma[echo_energy < epsilon] = 0.0

    # Optional smoothing to prevent rapid gain fluctuations
    if smoothing_samples > 1:
        kernel = np.ones(smoothing_samples) / smoothing_samples
        gamma = np.convolve(gamma, kernel, mode="same")

    return gamma


def imprint_energy(
    echo_bands: list[np.ndarray],
    balloon_energies: list[np.ndarray],
    echo_energies: list[np.ndarray],
    smoothing_samples: int = 0,
) -> list[np.ndarray]:
    """
    Apply the balloon's energy envelope to all bands of the echo sequence.

    For each band k, computes γ_k(t) and multiplies it onto the echo
    sequence band: shaped_k(t) = p_k(t) · γ_k(t).

    Parameters
    ----------
    echo_bands : list of np.ndarray
        Band-filtered echo sequence signals p_k(t).
    balloon_energies : list of np.ndarray
        Smoothed energy profiles β²_k(t) of the balloon bands.
    echo_energies : list of np.ndarray
        Smoothed energy profiles ν²_k(t) of the echo bands.
    smoothing_samples : int
        Gain smoothing window length in samples.

    Returns
    -------
    shaped_bands : list of np.ndarray
        Energy-shaped echo sequence bands: p_k(t) · γ_k(t).
    """
    shaped = []
    for p_k, beta_sq_k, nu_sq_k in zip(
        echo_bands, balloon_energies, echo_energies
    ):
        gamma_k = compute_gain(beta_sq_k, nu_sq_k, smoothing_samples)
        shaped.append(p_k * gamma_k)
    return shaped


# ---------------------------------------------------------------------------
# 3e. Direct Path Equalization & Final Summation
# ---------------------------------------------------------------------------

def estimate_direct_path_gains(
    balloon_bands: list[np.ndarray],
    onset_sample: int,
    sr: int,
    window_ms: float = 5.0,
) -> np.ndarray:
    """
    Estimate per-band gains of the direct path arrival from the balloon recording.

    For each band k, measures the RMS energy of the balloon's band-filtered
    signal in a short window around the onset. These gains reflect the
    spectral shape of the N-wave direct arrival — typically stronger at
    low frequencies and weaker at high frequencies.

    Parameters
    ----------
    balloon_bands : list of np.ndarray
        Band-filtered balloon recording signals b_k(t).
    onset_sample : int
        Onset position in the signal.
    sr : int
        Sample rate in Hz.
    window_ms : float
        Analysis window length after onset, in ms. Default 5ms.

    Returns
    -------
    gains : np.ndarray
        Per-band direct path RMS amplitudes. Shape (num_bands,).
    """
    window_len = int(sr * window_ms / 1000.0)
    gains = np.zeros(len(balloon_bands))

    for k, b_k in enumerate(balloon_bands):
        start = onset_sample
        end = min(len(b_k), onset_sample + window_len)
        if end <= start:
            continue
        # RMS of the direct path window
        gains[k] = np.sqrt(np.mean(b_k[start:end] ** 2))

    return gains


def equalize_and_sum(
    shaped_bands: list[np.ndarray],
    balloon_bands: list[np.ndarray],
    onset_sample: int,
    sr: int,
    window_ms: float = 5.0,
) -> np.ndarray:
    """
    Equalize shaped bands so the direct path is spectrally flat, then sum.

    Implements Equation (16) from Abel et al. (2010), §5.3:
        h̃(t) = Σ_k p_k(t) · γ_k(t) · α_k

    where α_k is the inverse of the direct path arrival's per-band gain,
    estimated from the BALLOON recording's band-filtered signal. This
    "whitens" the direct path: if the balloon's N-wave was louder at
    1kHz than 10kHz (due to the N-wave spectral shape), α_k compensates
    so the final IR's direct path is spectrally flat.

    Parameters
    ----------
    shaped_bands : list of np.ndarray
        Energy-shaped echo bands p_k(t) · γ_k(t) from Step 3d.
    balloon_bands : list of np.ndarray
        Band-filtered BALLOON recording b_k(t) — used to measure
        the direct path spectral shape.
    onset_sample : int
        Onset position.
    sr : int
        Sample rate in Hz.
    window_ms : float
        Window for direct path gain estimation.

    Returns
    -------
    ir : np.ndarray
        Final synthesized impulse response h̃(t).
    """
    # --- Estimate direct path per-band gains from balloon ---
    direct_gains = estimate_direct_path_gains(
        balloon_bands, onset_sample, sr, window_ms
    )

    # --- Equation (16): α_k = 1 / direct_gain_k ---
    # Normalize so the median α_k ≈ 1 (preserves overall level)
    valid = direct_gains > 0
    if np.any(valid):
        median_gain = np.median(direct_gains[valid])
    else:
        median_gain = 1.0

    alpha = np.ones(len(shaped_bands))
    for k in range(len(shaped_bands)):
        if direct_gains[k] > 0:
            # α_k = median_gain / gain_k
            # This inverts the spectral shape while keeping overall level
            alpha[k] = median_gain / direct_gains[k]

    # --- Sum all equalized bands ---
    ir = np.zeros_like(shaped_bands[0])
    for k, band in enumerate(shaped_bands):
        ir += band * alpha[k]

    return ir


# ---------------------------------------------------------------------------
# Main entry point for Stage 3
# ---------------------------------------------------------------------------

def shape_energy(
    balloon_mono: np.ndarray,
    echo_sequence: np.ndarray,
    sr: int,
    onset_sample: int,
    energy_window_ms: float = 10.0,
    extrapolate: bool = True,
    noise_floor_db: float = -40.0,
    gain_smoothing_ms: float = 0.0,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """
    Complete Stage 3 pipeline: shape the echo sequence's spectral energy
    to match the balloon recording's frequency-dependent decay.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Original (non-integrated) mono balloon recording from Stage 0.
    echo_sequence : np.ndarray
        Synthetic pulse sequence from Stage 1, same length as balloon_mono.
    sr : int
        Sample rate in Hz.
    onset_sample : int
        Onset position from Stage 0.
    energy_window_ms : float
        Band energy smoothing window in ms. Default 10ms (paper §5.2).
    extrapolate : bool
        Whether to extrapolate energy below noise floor. Default True.
    noise_floor_db : float
        Noise floor threshold for extrapolation. Default -40 dB.
    gain_smoothing_ms : float
        Gain function γ_k(t) smoothing window in ms. Default 0 (no smoothing).
    f_min : float
        Lowest filter bank center frequency in Hz.
    f_max : float or None
        Highest filter bank center frequency in Hz.

    Returns
    -------
    ir : np.ndarray
        Synthesized impulse response h̃(t), same length as input.
    """
    smoothing_samples = int(sr * gain_smoothing_ms / 1000.0)

    # --- 3a. Split both signals into 1/3-octave bands ---
    balloon_bands, centers, crossovers = apply_filterbank(
        balloon_mono, sr, f_min, f_max
    )
    echo_bands, _, _ = apply_filterbank(
        echo_sequence, sr, f_min, f_max
    )

    # --- 3b. Estimate band energies ---
    balloon_energies = estimate_all_band_energies(
        balloon_bands, sr, energy_window_ms
    )
    echo_energies = estimate_all_band_energies(
        echo_bands, sr, energy_window_ms
    )

    # --- 3c. Energy extrapolation (optional) ---
    if extrapolate:
        balloon_energies = [
            extrapolate_energy(e, sr, noise_floor_db)
            for e in balloon_energies
        ]

    # --- 3d. Energy imprinting ---
    shaped_bands = imprint_energy(
        echo_bands, balloon_energies, echo_energies, smoothing_samples
    )

    # --- 3e. Direct path equalization & summation ---
    ir = equalize_and_sum(
        shaped_bands, balloon_bands, onset_sample, sr
    )

    return ir
