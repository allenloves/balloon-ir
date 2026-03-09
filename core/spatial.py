"""
Spatial Character Analysis & Synthesis (Stage 2)

Implements §4 of Abel et al. (2010) "Estimating Room Impulse Responses
from Recorded Balloon Pops," AES Convention Paper 8171.

This module preserves the room's spatial character (envelopment vs.
directional energy) when synthesizing a stereo impulse response. It
estimates the inter-channel cross-correlation (ICCC) from the original
stereo balloon recording and imposes that correlation profile onto
two independently generated echo sequences from Stage 1.

Key concept: Cross-correlation near 0 indicates energy arriving
independently at each ear → sense of envelopment. Cross-correlation
near 1 indicates coherent energy from a specific direction → focused
image. Rooms typically start with high correlation (direct path) and
transition toward lower correlation as the sound field becomes diffuse
(see Fig. 9 and Fig. 10 in the paper).

This stage is SKIPPED entirely for mono recordings.

The paper (§4) describes two approaches for stereo synthesis:
  1. Three-sequence method: more accurate lag-dependent correlation,
     but "computationally cumbersome" (paper's words).
  2. Two-sequence rotation: simpler, uses a 2×2 rotation matrix M
     (Eq. 10-11) to impose zero-lag correlation.

We implement approach 2 (rotation matrix), as recommended by the paper
for practical use. The perceptual difference is minimal.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# 2a. Inter-Channel Cross-Correlation (ICCC) Estimation
# ---------------------------------------------------------------------------

def estimate_iccc(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    window_ms: float = 50.0,
) -> np.ndarray:
    """
    Compute running zero-lag cross-correlation between L and R channels.

    Implements Equation (9) from Abel et al. (2010), §4, using a
    running unit-sum window w(τ):

        C(t) = Σ w(τ) b₁(τ) b₂(τ)
               ─────────────────────────────────────
               [Σ w(τ) b₁²(τ)]^½ · [Σ w(τ) b₂²(τ)]^½

    where sums run from τ = t-Δ to t+Δ, and w is a rectangular window
    with unit sum. The paper uses a 50ms running window (§4, Fig. 9).

    We use zero-lag only (l=0), as the paper suggests this is sufficient
    for practical purposes. The full method would search lags in [-1,1]ms
    and take the maximum, but "the perceptual difference is minimal."

    Parameters
    ----------
    left : np.ndarray
        Left channel of the stereo balloon recording.
    right : np.ndarray
        Right channel of the stereo balloon recording.
    sr : int
        Sample rate in Hz.
    window_ms : float
        Running window length in ms. Default 50ms (paper §4).

    Returns
    -------
    iccc : np.ndarray
        Inter-channel cross-correlation profile C(t).
        Values in [-1, 1]. Same length as input signals.
        +1 = channels identical (mono), 0 = independent, -1 = inverted.
    """
    n = len(left)
    half_window = int(sr * window_ms / 1000.0 / 2.0)
    half_window = max(1, half_window)

    iccc = np.zeros(n)

    # Precompute cumulative sums for efficient windowed computation
    lr = left * right
    l_sq = left ** 2
    r_sq = right ** 2

    cum_lr = np.concatenate(([0.0], np.cumsum(lr)))
    cum_l_sq = np.concatenate(([0.0], np.cumsum(l_sq)))
    cum_r_sq = np.concatenate(([0.0], np.cumsum(r_sq)))

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)

        # --- Equation (9): Windowed cross-correlation ---
        sum_lr = cum_lr[hi] - cum_lr[lo]
        sum_l_sq = cum_l_sq[hi] - cum_l_sq[lo]
        sum_r_sq = cum_r_sq[hi] - cum_r_sq[lo]

        denom = np.sqrt(sum_l_sq * sum_r_sq)
        if denom > 0:
            iccc[i] = sum_lr / denom
        else:
            iccc[i] = 0.0

    # Clamp to [-1, 1] for numerical safety
    iccc = np.clip(iccc, -1.0, 1.0)

    return iccc


# ---------------------------------------------------------------------------
# 2b. Stereo Echo Sequence Synthesis
# ---------------------------------------------------------------------------

def impose_correlation(
    seq_left: np.ndarray,
    seq_right: np.ndarray,
    correlation_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a rotation matrix to impose prescribed cross-correlation.

    Implements Equations (10) and (11) from Abel et al. (2010), §4.

    Starting with two statistically independent echo sequences (from
    Stage 1, generated with different random seeds), this function
    applies a time-varying rotation matrix M to create the desired
    inter-channel correlation:

        M = [[cos θ, sin θ],     — Equation (10)
             [sin θ, cos θ]]

        θ(t) = arcsin(C(t)) / 2  — Equation (11)

    Properties of the rotation:
      - C = 0  →  θ = 0     →  M = I          →  channels unchanged (independent)
      - C = 1  →  θ = π/4   →  M = 1/√2·[[1,1],[1,1]]  →  channels identical (mono)
      - C = -1 →  θ = -π/4  →  channels become inversely correlated

    Parameters
    ----------
    seq_left : np.ndarray
        First independent echo sequence (assigned to left channel).
    seq_right : np.ndarray
        Second independent echo sequence (assigned to right channel).
        Must be the same length as seq_left.
    correlation_profile : np.ndarray
        Target cross-correlation C(t) to impose, values in [-1, 1].
        Same length as the sequences.

    Returns
    -------
    new_left : np.ndarray
        Left channel with imposed correlation.
    new_right : np.ndarray
        Right channel with imposed correlation.
    """
    # --- Equation (11): θ(t) = arcsin(C(t)) / 2 ---
    # Clamp correlation to valid range for arcsin
    C = np.clip(correlation_profile, -1.0, 1.0)
    theta = np.arcsin(C) / 2.0

    # --- Equation (10): Rotation matrix M ---
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    new_left = seq_left * cos_theta + seq_right * sin_theta
    new_right = seq_left * sin_theta + seq_right * cos_theta

    return new_left, new_right


# ---------------------------------------------------------------------------
# Main entry point for Stage 2
# ---------------------------------------------------------------------------

def analyze_and_synthesize_spatial(
    balloon_stereo: tuple[np.ndarray, np.ndarray],
    echo_sequences: list[np.ndarray],
    sr: int,
    window_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete Stage 2 pipeline: estimate spatial character from the
    stereo balloon recording and impose it on the echo sequences.

    Parameters
    ----------
    balloon_stereo : tuple of (np.ndarray, np.ndarray)
        (left, right) channels of the stereo balloon recording.
    echo_sequences : list of np.ndarray
        Two independently generated echo sequences from Stage 1.
        Must have length >= 2.
    sr : int
        Sample rate in Hz.
    window_ms : float
        ICCC estimation window in ms. Default 50ms (paper §4).

    Returns
    -------
    echo_left : np.ndarray
        Left channel echo sequence with imposed correlation.
    echo_right : np.ndarray
        Right channel echo sequence with imposed correlation.
    iccc_profile : np.ndarray
        Measured ICCC profile from the balloon recording.
    """
    left, right = balloon_stereo

    # --- 2a. Estimate ICCC from balloon recording ---
    iccc_profile = estimate_iccc(left, right, sr, window_ms)

    # --- 2b. Impose correlation on independent echo sequences ---
    echo_left, echo_right = impose_correlation(
        echo_sequences[0], echo_sequences[1], iccc_profile
    )

    return echo_left, echo_right, iccc_profile
