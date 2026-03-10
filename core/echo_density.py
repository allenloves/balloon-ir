"""
Echo Density Analysis & Synthesis (Stage 1)

Implements §3 of Abel et al. (2010) "Estimating Room Impulse Responses
from Recorded Balloon Pops," AES Convention Paper 8171.

This module analyzes the temporal structure of reflections in a balloon
pop recording and generates a synthetic pulse sequence with matching
perceptual echo density. The key insight (from psychoacoustic research
by Huang et al., 2008) is that human perception of echo density depends
on the overall density of arrivals, not their precise timing — allowing
us to synthesize a statistically equivalent sequence rather than
reproducing each reflection exactly.

Pipeline within this stage:
  1a. Integrate the balloon recording to convert N-wave doublets into
      single parabolic pulses (Eq. 7, Fig. 4-5).
  1b. Compute Normalized Echo Density (NED) η_b(t) on the integrated
      signal using a sliding window (Eq. 3-4).
  1c. Convert balloon NED to full-bandwidth NED η_h(t) using the
      duration ratio between N-wave and Dirac pulses (Eq. 8).
  1d. Clamp η_h(t): once it first reaches 0.995, hold it fixed
      for all subsequent samples (paper §3.2, below Eq. 8).
  1e. Convert η_h(t) to absolute echo density (AED) e(t) in
      echoes per second (Eq. 5 rearranged).
  1f. Synthesize a Poisson-distributed pulse sequence whose time-varying
      density follows e(t), with early reflections placed manually.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc
from typing import Optional


# ---------------------------------------------------------------------------
# 1a. Integration
# ---------------------------------------------------------------------------

def integrate_balloon(balloon_mono: np.ndarray) -> np.ndarray:
    """
    Integrate the balloon pop recording to convert N-wave doublets
    into single parabolic pulses.

    Implements the concept shown in Eq. (7) and Fig. 4-5 of Abel et al.
    (2010). The N-wave waveform from a balloon pop is a bipolar doublet
    (positive peak followed by negative peak). Its time integral is a
    single parabolic bump, which is easier to analyze for echo density
    because each reflection produces one peak instead of two.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Mono balloon pop recording, shape (num_samples,).
        This should be the raw (non-integrated) recording from Stage 0.

    Returns
    -------
    integrated : np.ndarray
        Cumulative integral of the input signal.
        Length is len(balloon_mono) - 1 due to trapezoidal integration.
    """
    # --- Equation (7): Time integral of N-wave → parabolic pulse ---
    # For an ideal N-wave n(t) of duration 2ρ/c, the integral is:
    #   ∫₀ᵗ n(τ)dτ = max{0, pρ/(2c) [1 - ((ct-R)/ρ)²]}
    # which is a parabola with peak pρ²/(4Rc) at the center.
    # For a recorded signal with many overlapping echoes, we simply
    # take the running cumulative integral.
    integrated = cumulative_trapezoid(balloon_mono, initial=0)
    return integrated


# ---------------------------------------------------------------------------
# 1b. Normalized Echo Density (NED) Estimation
# ---------------------------------------------------------------------------

def compute_ned(
    signal: np.ndarray,
    half_window: int,
    sr: int,
) -> np.ndarray:
    """
    Compute Normalized Echo Density (NED) over a sliding window.

    Implements Equation (3) from Abel et al. (2010), §3.1.

    NED measures how "diffuse" the signal is at each point in time,
    on a scale from 0 (sparse specular reflections) to ~1 (fully
    diffuse, statistically Gaussian). It works by counting what
    fraction of samples in a window exceed the window's standard
    deviation, then normalizing by the fraction expected for
    Gaussian noise (≈ 0.3173, from the complementary error function).

    Parameters
    ----------
    signal : np.ndarray
        The INTEGRATED balloon pop recording (not the raw recording).
        Integration converts N-wave doublets into single parabolic
        pulses, preventing each echo from being double-counted.
        See §3.2, Equation (7).
    half_window : int
        Half-window size Δ in samples. The full window is 2Δ+1.
        Paper uses ~43ms; at 48kHz this is Δ ≈ 1024.
    sr : int
        Sample rate in Hz (used only for documentation/validation).

    Returns
    -------
    ned : np.ndarray
        NED profile η(t), same length as input signal.
        Values typically range from 0 to ~1, but can slightly
        exceed 1 if the amplitude distribution is more uniform
        than Gaussian (see discussion in §3.1).
    """
    n = len(signal)
    ned = np.zeros(n)
    sq = signal ** 2

    # --- Gaussian reference value ---
    # erfc(1/√2) ≈ 0.3173 is the theoretical fraction of samples
    # outside ±1σ for a Gaussian distribution. Dividing by this
    # calibrates the scale so that Gaussian noise → NED ≈ 1.0.
    GAUSSIAN_REFERENCE = erfc(1.0 / np.sqrt(2.0))  # ≈ 0.3173

    for i in range(n):
        # Window bounds, clamped to signal edges
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window_sq = sq[lo:hi]
        window_length = hi - lo

        # --- Equation (4): Window variance σ²(t) ---
        # This is the mean squared value in the window, equivalent to
        # RMS² since we assume zero-mean signal (the integrated balloon
        # recording fluctuates around zero in the late-field).
        sigma_sq = np.mean(window_sq)

        if sigma_sq == 0:
            ned[i] = 0.0
            continue

        # --- Equation (3): Normalized Echo Density ---
        # Count samples where instantaneous energy h²(τ) exceeds σ²(t).
        # Sparse reflections: only a few large peaks exceed σ² → low count.
        # Gaussian noise: ~31.73% of samples exceed σ² → count/N ≈ 0.3173.
        exceeding_count = np.sum(window_sq > sigma_sq)

        # Normalize: fraction of exceeding samples / Gaussian reference
        fraction = exceeding_count / window_length
        ned[i] = fraction / GAUSSIAN_REFERENCE

    return ned


def compute_ned_fast(
    signal: np.ndarray,
    half_window: int,
    sr: int,
) -> np.ndarray:
    """
    Fast (vectorized) NED computation using stride tricks.

    Equivalent to compute_ned() but significantly faster for long signals.
    Uses the same algorithm (Eq. 3-4) but processes all windows at once
    via NumPy vectorization.

    Parameters
    ----------
    signal : np.ndarray
        The INTEGRATED balloon pop recording.
    half_window : int
        Half-window size Δ in samples.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    ned : np.ndarray
        NED profile η(t), same length as input signal.
    """
    n = len(signal)
    ned = np.zeros(n)
    sq = signal ** 2

    GAUSSIAN_REFERENCE = erfc(1.0 / np.sqrt(2.0))  # ≈ 0.3173

    # Compute running sum of sq using cumulative sum for O(1) per window
    cumsum_sq = np.concatenate(([0.0], np.cumsum(sq)))

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window_length = hi - lo

        # --- Equation (4): σ²(t) via cumulative sum ---
        sigma_sq = (cumsum_sq[hi] - cumsum_sq[lo]) / window_length

        if sigma_sq == 0:
            continue

        # --- Equation (3): Count exceeding samples ---
        # Still need per-window comparison; slice is fast in NumPy
        exceeding_count = np.sum(sq[lo:hi] > sigma_sq)
        ned[i] = (exceeding_count / window_length) / GAUSSIAN_REFERENCE

    return ned


# ---------------------------------------------------------------------------
# 1c. NED Conversion (Balloon → Full-Bandwidth)
# ---------------------------------------------------------------------------

def convert_ned_balloon_to_fullband(
    eta_b: np.ndarray,
    duration_ratio: float,
) -> np.ndarray:
    """
    Convert balloon NED to equivalent full-bandwidth NED.

    Implements Equation (8) from Abel et al. (2010), §3.2.

    The balloon's N-wave echoes have a finite duration (2ρ/c), which is
    much longer than a full-bandwidth Dirac pulse (δ ≈ 1/sr). Because
    the N-wave echoes are wider, they overlap more easily, making the
    balloon recording appear denser than the underlying room response
    would be with ideal impulses. This function corrects for that
    duration ratio.

    Parameters
    ----------
    eta_b : np.ndarray
        Balloon NED profile η_b(t). Values in [0, ~1].
    duration_ratio : float
        Ratio of N-wave duration to full-bandwidth pulse duration:
        (2ρ/c) / δ = (2ρ/c) * sr.
        Typically 10–20 for common balloon sizes at 48kHz.
        Example: balloon diameter 30cm → ρ=0.15m → 2ρ/c ≈ 0.87ms
                 at sr=48000 → duration_ratio ≈ 0.87ms * 48 ≈ 42

    Returns
    -------
    eta_h : np.ndarray
        Full-bandwidth NED profile η_h(t). Same shape as eta_b.
        η_h ≤ η_b always, because removing the temporal smearing
        of the N-wave reveals a sparser underlying density.
    """
    # --- Clamp η_b to [0, 0.999] to avoid division by zero ---
    # When η_b → 1, the denominator → 0. Clamping at 0.999 keeps
    # the conversion numerically stable (see spec §11, item 5).
    eta_b_clamped = np.clip(eta_b, 0.0, 0.999)

    # --- Equation (8): η_h = η_b / ((1 - η_b) · R + η_b) ---
    # where R = 2ρ/(δc) is the duration ratio.
    # When η_b = 0 → η_h = 0  (sparse stays sparse)
    # When η_b = 1 → η_h = 1  (fully diffuse stays fully diffuse)
    # For intermediate values, η_h < η_b.
    eta_h = eta_b_clamped / (
        (1.0 - eta_b_clamped) * duration_ratio + eta_b_clamped
    )

    return eta_h


# ---------------------------------------------------------------------------
# 1d. NED Clamping
# ---------------------------------------------------------------------------

def clamp_ned(
    eta_h: np.ndarray,
    threshold: float = 0.995,
) -> np.ndarray:
    """
    Clamp full-bandwidth NED: once it first reaches the threshold,
    hold it fixed for all subsequent samples.

    NOTE: Paper §3.2 (below Eq. 8) states that η_h is held fixed
    at 0.995 after first reaching that value, to prevent numerical
    instability in the fully diffuse late-field. We implement this
    as a forward pass: once the threshold is crossed, all subsequent
    values are clamped. See also Fig. 6 (dashed line plateaus).

    Parameters
    ----------
    eta_h : np.ndarray
        Full-bandwidth NED profile before clamping.
    threshold : float
        Clamping threshold. Default 0.995 (from paper).

    Returns
    -------
    eta_h_clamped : np.ndarray
        Clamped NED profile. Same shape as input.
    """
    eta_h_clamped = eta_h.copy()

    # Find first index where η_h reaches the threshold
    indices = np.where(eta_h_clamped >= threshold)[0]
    if len(indices) > 0:
        first_crossing = indices[0]
        eta_h_clamped[first_crossing:] = threshold

    return eta_h_clamped


# ---------------------------------------------------------------------------
# 1e. Absolute Echo Density (AED)
# ---------------------------------------------------------------------------

def compute_aed(
    eta_h: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Convert full-bandwidth NED to Absolute Echo Density (AED).

    Rearranges Equation (5) from Abel et al. (2010), §3.2:
        η(t) = e(t) / (e(t) + 1/δ)
    to solve for e(t):
        e(t) = η(t) · (1/δ) / (1 - η(t))
             = η(t) · sr / (1 - η(t))

    where δ ≈ 1/sr is the full-bandwidth pulse duration (one sample).

    This gives the number of full-bandwidth echoes per second at each
    time point. The AED drives the Poisson process in Step 1f.

    Parameters
    ----------
    eta_h : np.ndarray
        Full-bandwidth NED profile η_h(t). Should be clamped (Step 1d)
        so that values stay below 1.0 to avoid division by zero.
    sr : int
        Sample rate in Hz. Used as 1/δ (inverse pulse duration).

    Returns
    -------
    aed : np.ndarray
        Absolute echo density e(t) in echoes per second.
        Same shape as eta_h.
    """
    # --- Rearranged Equation (5) ---
    # e(t) = η_h(t) * sr / (1 - η_h(t))
    # With η_h clamped to max 0.995, maximum AED = 0.995 * sr / 0.005
    # = 199 * sr ≈ 9.6 million echoes/sec at 48kHz (very dense).
    aed = eta_h * sr / (1.0 - eta_h)

    return aed


# ---------------------------------------------------------------------------
# 1f. Echo Sequence Synthesis
# ---------------------------------------------------------------------------

def estimate_nwave_duration(
    balloon_mono: np.ndarray,
    onset_sample: int,
    sr: int,
    search_window_ms: float = 5.0,
) -> tuple[float, float]:
    """
    Estimate the N-wave duration from the direct-path arrival,
    and derive the balloon radius ρ.

    The N-wave duration equals 2ρ/c, where ρ is the balloon radius
    and c is the speed of sound. We measure this by finding the time
    between the first and last zero-crossings of the direct-path
    N-wave arrival.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Raw (non-integrated) mono balloon recording.
    onset_sample : int
        Detected onset sample index.
    sr : int
        Sample rate in Hz.
    search_window_ms : float
        Window after onset to search for the N-wave, in ms.

    Returns
    -------
    nwave_duration_s : float
        Estimated N-wave duration 2ρ/c in seconds.
    balloon_radius_m : float
        Estimated balloon radius ρ in meters.
    """
    c = 343.0  # Speed of sound in air, m/s

    search_len = int(sr * search_window_ms / 1000.0)
    segment = balloon_mono[onset_sample:onset_sample + search_len]

    # Find zero crossings in the direct-path N-wave
    sign_changes = np.where(np.diff(np.sign(segment)))[0]

    if len(sign_changes) >= 2:
        # N-wave duration = time between first and last zero crossing
        first_zc = sign_changes[0]
        last_zc = sign_changes[-1]
        nwave_duration_s = (last_zc - first_zc) / sr
    else:
        # ENGINEERING DECISION: If zero crossings are not clearly found,
        # fall back to a typical balloon diameter of 25cm.
        nwave_duration_s = 0.25 / c * 2  # 2ρ/c for ρ = 0.125m

    balloon_radius_m = nwave_duration_s * c / 2.0

    return nwave_duration_s, balloon_radius_m


def detect_early_reflections(
    integrated: np.ndarray,
    onset_sample: int,
    sr: int,
    num_reflections: int = 2,
    min_spacing_ms: float = 2.0,
    threshold_db: float = -12.0,
) -> list[dict]:
    """
    Detect early reflections (direct path, floor reflection, etc.)
    from the integrated balloon response.

    ENGINEERING DECISION: The paper (§3.2, below Eq. 8) says "the
    first few clear arrivals may be placed by hand." We automate this
    by detecting peaks in the integrated signal that exceed a threshold
    relative to the direct-path peak.

    Parameters
    ----------
    integrated : np.ndarray
        Integrated balloon recording (from Step 1a).
    onset_sample : int
        Onset position in the integrated signal.
    sr : int
        Sample rate in Hz.
    num_reflections : int
        Maximum number of early reflections to detect. Default 2
        (direct path + floor reflection).
    min_spacing_ms : float
        Minimum spacing between detected reflections in ms.
    threshold_db : float
        Detection threshold in dB relative to the direct-path peak.

    Returns
    -------
    reflections : list of dict
        Each dict has:
        - 'sample': int — sample index of the reflection
        - 'amplitude': float — amplitude of the reflection
        - 'time_ms': float — time in ms relative to onset
    """
    min_spacing_samples = int(sr * min_spacing_ms / 1000.0)

    # Search in the early part of the response (first 200ms after onset)
    search_end = onset_sample + int(sr * 0.2)
    search_end = min(search_end, len(integrated))
    segment = np.abs(integrated[onset_sample:search_end])

    if len(segment) == 0:
        return []

    peak_val = np.max(segment)
    if peak_val == 0:
        return []

    threshold_linear = peak_val * 10.0 ** (threshold_db / 20.0)

    reflections = []
    used_mask = np.zeros(len(segment), dtype=bool)

    for _ in range(num_reflections):
        # Mask out already-used regions
        masked = segment.copy()
        masked[used_mask] = 0.0

        if np.max(masked) < threshold_linear:
            break

        peak_idx = np.argmax(masked)
        amp = segment[peak_idx]

        reflections.append({
            "sample": onset_sample + peak_idx,
            "amplitude": amp,
            "time_ms": peak_idx / sr * 1000.0,
        })

        # Mask out a region around this peak to enforce minimum spacing
        mask_lo = max(0, peak_idx - min_spacing_samples)
        mask_hi = min(len(segment), peak_idx + min_spacing_samples + 1)
        used_mask[mask_lo:mask_hi] = True

    # Sort by time
    reflections.sort(key=lambda r: r["sample"])
    return reflections


def detect_early_reflections_ned_guided(
    integrated: np.ndarray,
    ned_fullband: np.ndarray,
    onset_sample: int,
    sr: int,
    ned_threshold: float = 0.3,
    min_early_ms: float = 10.0,
    min_spacing_ms: float = 1.0,
    peak_threshold_db: float = -20.0,
    max_reflections: int = 50,
) -> tuple[list[dict], int]:
    """
    Detect early reflections using the NED profile as a phase-transition
    boundary. Instead of detecting a fixed number of peaks, this function
    finds ALL discrete reflections in the sparse early region (where
    NED < ned_threshold), then hands off to Poisson synthesis once the
    field becomes statistically diffuse.

    This preserves the perceptually important "bounce" character of the
    early field — discrete, clearly separated wall reflections that are
    audible as distinct echoes before the late reverb tail takes over.

    ENGINEERING DECISION: The original detect_early_reflections() with
    num_reflections=2 only placed the direct path and floor reflection,
    then immediately started Poisson synthesis. This caused the sparse
    early region (typically 30–80ms) to be filled with statistically
    uniform noise, destroying the discrete bounce character. The NED-
    guided approach lets the measured echo density profile itself decide
    where the transition should occur.

    Parameters
    ----------
    integrated : np.ndarray
        Integrated balloon recording (from Step 1a).
    ned_fullband : np.ndarray
        Full-bandwidth NED profile η_h(t) (from Steps 1c-1d).
        Used to determine the sparse→dense transition point.
    onset_sample : int
        Onset position in the integrated signal.
    sr : int
        Sample rate in Hz.
    ned_threshold : float
        NED value at which we consider the field "dense enough" for
        Poisson synthesis. Default 0.3 means: once 30% of the way
        to fully diffuse, stop placing manual peaks and let the
        Poisson process take over. Typical rooms transition around
        30–80ms after onset.
    min_early_ms : float
        Minimum duration of the early region in ms, regardless of NED.
        Default 10ms. This is a physics-based floor: sound travels
        ~3.4 m in 10ms, so the earliest wall reflection in any room
        cannot arrive before the sound has made a round trip to the
        nearest surface. Even in very reverberant spaces where NED
        rises quickly, the first ~10ms after the direct path are
        physically guaranteed to contain only a few discrete arrivals.
        Set to 0 to disable (pure NED-guided).
    min_spacing_ms : float
        Minimum spacing between detected reflections in ms.
        Reduced from default 2.0 to 1.0 to catch closely spaced
        early reflections (e.g. flutter echoes between parallel walls).
    peak_threshold_db : float
        Detection threshold in dB relative to the strongest peak in
        the search region. Lowered from default -12 to -20 to catch
        weaker early reflections that still contribute to bounce.
    max_reflections : int
        Safety cap on number of detected reflections.

    Returns
    -------
    reflections : list of dict
        Each dict has:
        - 'sample': int — sample index of the reflection
        - 'amplitude': float — amplitude of the reflection
        - 'time_ms': float — time in ms relative to onset
    transition_sample : int
        Sample index where NED first exceeds ned_threshold.
        Poisson synthesis should start from this point.
    """
    # --- Find the transition point from NED profile ---
    # Search only from onset onwards
    ned_from_onset = ned_fullband[onset_sample:]
    transition_indices = np.where(ned_from_onset >= ned_threshold)[0]

    if len(transition_indices) > 0:
        # Transition point relative to full signal
        transition_sample = onset_sample + transition_indices[0]
    else:
        # NED never reaches threshold — use entire signal as "early"
        # (very dry room or very short recording)
        transition_sample = len(integrated)

    # --- Enforce minimum early region duration ---
    # Physics-based floor: the earliest possible wall reflection requires
    # a round trip to the nearest surface. Even in very dense/reverberant
    # spaces, the NED estimation window (43ms) can cause the NED to appear
    # high at onset because it averages in energy from later dense arrivals.
    # The min_early_ms floor prevents the transition from occurring before
    # the first few physical reflections have had time to arrive.
    min_early_samples = int(sr * min_early_ms / 1000.0)
    min_transition = onset_sample + min_early_samples
    if transition_sample < min_transition:
        transition_sample = min_transition

    transition_time_ms = (transition_sample - onset_sample) / sr * 1000.0

    # --- Detect peaks in the sparse region [onset, transition) ---
    search_end = min(transition_sample, len(integrated))
    segment = np.abs(integrated[onset_sample:search_end])

    if len(segment) == 0:
        return [], transition_sample

    peak_val = np.max(segment)
    if peak_val == 0:
        return [], transition_sample

    min_spacing_samples = int(sr * min_spacing_ms / 1000.0)
    threshold_linear = peak_val * 10.0 ** (peak_threshold_db / 20.0)

    reflections = []
    used_mask = np.zeros(len(segment), dtype=bool)

    for _ in range(max_reflections):
        masked = segment.copy()
        masked[used_mask] = 0.0

        if np.max(masked) < threshold_linear:
            break

        peak_idx = np.argmax(masked)
        amp = segment[peak_idx]

        reflections.append({
            "sample": onset_sample + peak_idx,
            "amplitude": amp,
            "time_ms": peak_idx / sr * 1000.0,
        })

        # Mask out region around this peak
        mask_lo = max(0, peak_idx - min_spacing_samples)
        mask_hi = min(len(segment), peak_idx + min_spacing_samples + 1)
        used_mask[mask_lo:mask_hi] = True

    # Sort by time
    reflections.sort(key=lambda r: r["sample"])

    return reflections, transition_sample


def synthesize_echo_sequence(
    aed_profile: np.ndarray,
    sr: int,
    duration_samples: int,
    early_reflections: Optional[list[dict]] = None,
    transition_sample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a synthetic pulse sequence with Poisson-distributed
    inter-arrival times following the AED profile.

    Implements the echo sequence synthesis described in §3.2 of Abel
    et al. (2010). Pulses are generated at Poisson-distributed time
    intervals according to the estimated echo density profile e(t).
    Pulse amplitudes are drawn from a Gaussian distribution and scaled
    according to the local echo density so as to give a roughly
    constant energy profile (paper p.6, above Fig. 8).

    The first few arrivals (direct path, floor reflection) are placed
    manually at positions detected in Step 1f, rather than generated
    randomly (paper §3.2, below Eq. 8).

    Parameters
    ----------
    aed_profile : np.ndarray
        Absolute echo density e(t) in echoes/second.
    sr : int
        Sample rate in Hz.
    duration_samples : int
        Length of the output sequence in samples.
    early_reflections : list of dict or None
        Early reflections to place manually. Each dict must have
        'sample' (int) and 'amplitude' (float) keys.
    transition_sample : int or None
        Sample index where Poisson synthesis should begin. If provided,
        this overrides the default behavior of starting immediately
        after the last early reflection. This is the NED-guided
        transition point from detect_early_reflections_ned_guided().
        Before this point, ONLY manually placed early reflections
        appear in the output (preserving the sparse, discrete bounce
        character of the early field).
        If None, falls back to original behavior: start Poisson
        synthesis right after the last early reflection.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
        If None, a new default generator is created.

    Returns
    -------
    echo_sequence : np.ndarray
        Synthetic pulse train, shape (duration_samples,).
        Contains Dirac-like pulses (single-sample spikes) at
        Poisson-distributed intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    echo_seq = np.zeros(duration_samples)

    # --- Place early reflections manually ---
    # Paper §3.2: "pulses representing the first few clear arrivals
    # (typically the direct path and floor reflection) may be placed
    # by hand."
    #
    # With NED-guided detection (method C), early_reflections may
    # contain all discrete peaks in the sparse region, not just 2.
    synthesis_start = 0
    if early_reflections:
        for ref in early_reflections:
            idx = ref["sample"]
            if 0 <= idx < duration_samples:
                echo_seq[idx] = ref["amplitude"]

    # --- Determine Poisson synthesis start point ---
    if transition_sample is not None:
        # NED-guided: start Poisson at the measured transition point
        synthesis_start = transition_sample
    elif early_reflections:
        # Legacy behavior: start right after last early reflection
        synthesis_start = max(r["sample"] for r in early_reflections) + 1

    # --- Poisson process synthesis ---
    # For each time step, the expected interval between pulses is
    # sr / e(t) samples. We draw actual intervals from an exponential
    # distribution with that mean (Poisson process property).
    current_sample = synthesis_start

    while current_sample < duration_samples:
        # Get local AED at current position
        aed_idx = min(current_sample, len(aed_profile) - 1)
        local_aed = aed_profile[aed_idx]

        if local_aed <= 0:
            # No echoes expected here; skip ahead
            current_sample += 1
            continue

        # Expected interval in samples: sr / e(t)
        expected_interval = sr / local_aed

        # Draw actual interval from exponential distribution
        # (memoryless property of Poisson process)
        actual_interval = rng.exponential(expected_interval)
        actual_interval = max(1.0, actual_interval)  # at least 1 sample

        current_sample += int(round(actual_interval))

        if current_sample >= duration_samples:
            break

        # Draw pulse amplitude from Gaussian distribution
        # NOTE ON ENERGY SCALING: The paper (§3.2, p.6 above Fig. 8) states
        # that amplitudes are "scaled according to the local echo density
        # so as to give a roughly constant energy profile." This would mean
        # multiplying by 1/√(local_aed) so that energy/time ∝ amplitude² × density
        # stays constant. We intentionally omit this scaling here because
        # Stage 3 (energy_shaping.py) applies γ_k(t) = β_k(t)/ν_k(t), which
        # overwrites the energy profile entirely — matching it to the balloon
        # recording's measured band energies. Any scaling done here would be
        # undone by γ_k(t).
        #
        # If you need to audition this echo sequence BEFORE Stage 3 (e.g.,
        # for teaching demonstrations), uncomment the following line:
        #   amplitude = rng.standard_normal() / np.sqrt(max(local_aed, 1.0))
        amplitude = rng.standard_normal()

        echo_seq[current_sample] = amplitude

    return echo_seq


# ---------------------------------------------------------------------------
# Main entry point for Stage 1
# ---------------------------------------------------------------------------

def analyze_and_synthesize_density(
    balloon_mono: np.ndarray,
    sr: int,
    onset_sample: int,
    ned_window_ms: float = 43.0,
    balloon_diameter_cm: Optional[float] = None,
    num_early_reflections: int = 2,
    ned_transition_threshold: Optional[float] = 0.3,
    min_early_ms: float = 10.0,
    num_sequences: int = 1,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Complete Stage 1 pipeline: analyze balloon echo density and
    synthesize matching pulse sequence(s).

    Parameters
    ----------
    balloon_mono : np.ndarray
        Mono balloon pop recording from Stage 0.
    sr : int
        Sample rate in Hz.
    onset_sample : int
        Onset position from Stage 0.
    ned_window_ms : float
        NED estimation window length in ms. Default 43ms (paper §3.1).
    balloon_diameter_cm : float or None
        Balloon diameter in cm. If None, auto-detect from direct path.
    num_early_reflections : int
        Number of early reflections to detect and place manually.
        Only used when ned_transition_threshold is None (legacy mode).
    ned_transition_threshold : float or None
        NED value for sparse→dense phase transition (Method C).
        When set (default 0.3), uses NED-guided early reflection
        detection: ALL discrete peaks in the sparse region (where
        η_h < threshold) are placed manually, and Poisson synthesis
        only begins after the transition point. This preserves the
        perceptually important "bounce" character of early reflections.
        Set to None to use legacy fixed-count detection.
    min_early_ms : float
        Minimum early region duration in ms, regardless of NED.
        Default 10ms. Physics-based floor: sound travels ~3.4m in
        10ms, so the earliest wall reflection cannot arrive before
        this. Prevents the NED estimation window from causing a
        premature transition in dense/reverberant spaces.
        Set to 0 to disable.
    num_sequences : int
        Number of independent echo sequences to generate.
        Use 2 for stereo processing (Stage 2).
    random_seed : int or None
        Base random seed for reproducibility.

    Returns
    -------
    result : dict
        'integrated'           : np.ndarray — integrated balloon recording
        'ned_balloon'          : np.ndarray — balloon NED η_b(t)
        'ned_fullband'         : np.ndarray — full-bandwidth NED η_h(t)
        'aed'                  : np.ndarray — absolute echo density e(t)
        'nwave_duration_s'     : float — estimated N-wave duration
        'balloon_radius_m'     : float — estimated balloon radius
        'early_reflections'    : list[dict] — detected early reflections
        'transition_sample'    : int or None — NED transition point
        'transition_time_ms'   : float or None — transition time in ms
        'echo_sequences'       : list[np.ndarray] — synthesized pulse sequences
    """
    # --- 1a. Integration ---
    integrated = integrate_balloon(balloon_mono)

    # --- Estimate N-wave duration ---
    if balloon_diameter_cm is not None:
        # User-provided balloon diameter
        balloon_radius_m = balloon_diameter_cm / 200.0  # cm → m
        c = 343.0
        nwave_duration_s = 2.0 * balloon_radius_m / c
    else:
        # Auto-detect from direct-path waveform
        nwave_duration_s, balloon_radius_m = estimate_nwave_duration(
            balloon_mono, onset_sample, sr
        )

    # --- 1b. NED estimation on integrated signal ---
    half_window = int(sr * ned_window_ms / 1000.0 / 2.0)  # Δ in samples
    half_window = max(1, half_window)
    ned_balloon = compute_ned_fast(integrated, half_window, sr)

    # --- 1c. NED conversion (balloon → full-bandwidth) ---
    # Duration ratio R = (2ρ/c) / (1/sr) = nwave_duration_s * sr
    duration_ratio = nwave_duration_s * sr
    ned_fullband = convert_ned_balloon_to_fullband(ned_balloon, duration_ratio)

    # --- 1d. Clamp η_h ---
    ned_fullband = clamp_ned(ned_fullband)

    # --- 1e. AED ---
    aed = compute_aed(ned_fullband, sr)

    # --- Detect early reflections ---
    transition_sample = None
    if ned_transition_threshold is not None:
        # Method C: NED-guided phase transition
        early_refs, transition_sample = detect_early_reflections_ned_guided(
            integrated, ned_fullband, onset_sample, sr,
            ned_threshold=ned_transition_threshold,
            min_early_ms=min_early_ms,
        )
        transition_time_ms = (transition_sample - onset_sample) / sr * 1000.0
    else:
        # Legacy: fixed-count detection
        early_refs = detect_early_reflections(
            integrated, onset_sample, sr,
            num_reflections=num_early_reflections,
        )
        transition_time_ms = None

    # --- 1f. Synthesize echo sequence(s) ---
    sequences = []
    for i in range(num_sequences):
        seed = (random_seed + i) if random_seed is not None else None
        rng = np.random.default_rng(seed)
        seq = synthesize_echo_sequence(
            aed, sr, len(balloon_mono),
            early_reflections=early_refs,
            transition_sample=transition_sample,
            rng=rng,
        )
        sequences.append(seq)

    return {
        "integrated": integrated,
        "ned_balloon": ned_balloon,
        "ned_fullband": ned_fullband,
        "aed": aed,
        "nwave_duration_s": nwave_duration_s,
        "balloon_radius_m": balloon_radius_m,
        "early_reflections": early_refs,
        "transition_sample": transition_sample,
        "transition_time_ms": transition_time_ms,
        "echo_sequences": sequences,
    }
