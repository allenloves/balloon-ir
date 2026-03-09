"""
Preprocessing (Stage 0)

Implements the input preparation stage described in §6 of Abel et al. (2010)
"Estimating Room Impulse Responses from Recorded Balloon Pops,"
AES Convention Paper 8171.

This module reads a balloon pop WAV file, normalizes the amplitude,
detects the onset of the pop, and trims pre-onset silence. These
operations ensure that downstream stages (echo density analysis,
energy shaping, etc.) receive a clean, consistently formatted signal
with a known onset position.

The onset detection uses a short-window RMS energy threshold: the first
sample where the running RMS exceeds a fraction of the peak RMS is
identified as the onset. This is robust to low-level background noise
that precedes the balloon burst.
"""

import numpy as np
import soundfile as sf
from typing import Optional


def read_and_normalize(
    file_path: str,
    target_peak: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and normalize its amplitude.

    Parameters
    ----------
    file_path : str
        Path to the WAV file.
    target_peak : float
        Peak amplitude after normalization. Default 1.0.

    Returns
    -------
    audio : np.ndarray
        Audio data, shape (num_samples,) for mono or (num_samples, 2) for stereo.
        Normalized so that max(abs(audio)) == target_peak.
    sr : int
        Sample rate in Hz.
    """
    audio, sr = sf.read(file_path, dtype="float64")
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio, sr


def detect_onset(
    audio_mono: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    window_ms: float = 1.0,
) -> int:
    """
    Detect the onset of the balloon pop using short-window RMS energy.

    The algorithm computes a running RMS in a short window (default 1ms)
    and finds the first sample where the RMS exceeds the threshold
    (specified in dB relative to the peak RMS of the entire signal).

    Parameters
    ----------
    audio_mono : np.ndarray
        Mono audio signal, shape (num_samples,).
    sr : int
        Sample rate in Hz.
    threshold_db : float
        Onset threshold in dB relative to peak RMS. Default -40 dB.
        A more negative value detects earlier (more sensitive).
    window_ms : float
        RMS window length in milliseconds. Default 1.0 ms.

    Returns
    -------
    onset_sample : int
        Index of the detected onset sample.
    """
    window_len = max(1, int(sr * window_ms / 1000.0))
    n = len(audio_mono)

    # Handle signals shorter than window
    if n <= window_len:
        peak = np.max(np.abs(audio_mono))
        if peak == 0:
            return 0
        threshold_linear = peak * 10.0 ** (threshold_db / 20.0)
        indices = np.where(np.abs(audio_mono) >= threshold_linear)[0]
        return int(indices[0]) if len(indices) > 0 else 0

    # Compute running RMS using a cumulative sum trick for efficiency.
    # Use a causal window (looking back) so the detected index aligns
    # with the onset rather than lagging behind it.
    sq = audio_mono ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(sq)))

    # rms[i] = RMS of samples [i : i + window_len]
    # This is a forward-looking window, so the onset is detected
    # at the start of the energy burst, not the center.
    rms_sq = (cumsum[window_len:] - cumsum[:-window_len]) / window_len
    rms = np.sqrt(rms_sq)

    peak_rms = np.max(rms)
    if peak_rms == 0:
        return 0

    threshold_linear = peak_rms * 10.0 ** (threshold_db / 20.0)
    indices = np.where(rms >= threshold_linear)[0]

    if len(indices) == 0:
        return 0

    # Coarse onset from RMS (may be up to window_len samples early
    # because the window that first contains the onset extends backward)
    coarse_onset = int(indices[0])

    # Refine: the coarse onset may be up to window_len samples early,
    # because the RMS window that first contains the true onset extends
    # backward. Search within [coarse_onset, coarse_onset + window_len]
    # for the first sample whose absolute amplitude exceeds 10% of the
    # local peak — this skips low-level background noise and lands on
    # the actual first energetic sample.
    refine_start = coarse_onset
    refine_end = min(n, coarse_onset + window_len)
    abs_segment = np.abs(audio_mono[refine_start:refine_end])
    local_peak = np.max(abs_segment)

    if local_peak > 0:
        amp_threshold = local_peak * 0.1
        refine_indices = np.where(abs_segment >= amp_threshold)[0]
        if len(refine_indices) > 0:
            return refine_start + int(refine_indices[0])

    return coarse_onset


def preprocess(
    file_path: str,
    target_sr: Optional[int] = None,
    pre_onset_ms: float = 10.0,
    onset_threshold_db: float = -40.0,
) -> dict:
    """
    Full preprocessing pipeline for a balloon pop recording.

    Reads the WAV file, normalizes, detects onset, and trims.
    Optionally resamples to a target sample rate.

    Parameters
    ----------
    file_path : str
        Path to the balloon pop WAV file.
    target_sr : int or None
        If specified, resample to this sample rate. If None, keep original.
    pre_onset_ms : float
        Milliseconds of audio to keep before the detected onset.
        Default 10 ms (safety margin).
    onset_threshold_db : float
        Onset detection threshold in dB relative to peak RMS.

    Returns
    -------
    result : dict
        'balloon_mono'    : np.ndarray — mono signal, trimmed
        'balloon_stereo'  : tuple[np.ndarray, np.ndarray] or None — (L, R) if stereo
        'sr'              : int — sample rate
        'onset_sample'    : int — onset position in the trimmed signal
    """
    audio, sr = read_and_normalize(file_path)

    # Resample if requested
    if target_sr is not None and target_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g, axis=0)
        sr = target_sr

    # Create mono mixdown
    if audio.ndim == 2 and audio.shape[1] == 2:
        mono = (audio[:, 0] + audio[:, 1]) / 2.0
        is_stereo = True
    else:
        mono = audio.ravel()
        is_stereo = False

    # Detect onset on mono signal
    onset = detect_onset(mono, sr, threshold_db=onset_threshold_db)

    # Trim: keep pre_onset_ms before onset
    pre_onset_samples = int(sr * pre_onset_ms / 1000.0)
    trim_start = max(0, onset - pre_onset_samples)

    mono = mono[trim_start:]

    # Adjust onset position relative to trimmed signal
    onset_in_trimmed = onset - trim_start

    result = {
        "balloon_mono": mono,
        "balloon_stereo": None,
        "sr": sr,
        "onset_sample": onset_in_trimmed,
    }

    if is_stereo:
        left = audio[trim_start:, 0]
        right = audio[trim_start:, 1]
        result["balloon_stereo"] = (left, right)

    return result
