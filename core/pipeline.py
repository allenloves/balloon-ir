"""
Full Processing Pipeline

Orchestrates all stages (0–4) of the balloon pop → room impulse response
synthesis method described in Abel et al. (2010), AES Convention Paper 8171.

This module wires together:
  Stage 0: Preprocessing (onset detection, normalization, trimming)
  Stage 1: Echo density analysis & synthesis (NED, AED, Poisson pulses)
  Stage 2: Spatial character (stereo ICCC & correlation imposition)
  Stage 3: Time-frequency energy shaping (filter bank, band energy, γ_k, α_k)
  Stage 4: Post-processing (normalize, fade-out, trim, export)
"""

import numpy as np
from typing import Optional

from core.preprocessing import preprocess
from core.echo_density import analyze_and_synthesize_density
from core.spatial import analyze_and_synthesize_spatial
from core.energy_shaping import shape_energy
from core.postprocessing import postprocess, export_wav


def process_balloon(
    file_path: str,
    # Stage 0 params
    target_sr: Optional[int] = None,
    onset_threshold_db: float = -40.0,
    # Stage 1 params
    ned_window_ms: float = 43.0,
    balloon_diameter_cm: Optional[float] = None,
    num_early_reflections: int = 2,
    ned_transition_threshold: Optional[float] = 0.3,
    random_seed: Optional[int] = None,
    # Stage 2 params (stereo only)
    iccc_window_ms: float = 50.0,
    # Stage 3 params
    energy_window_ms: float = 10.0,
    extrapolate: bool = True,
    noise_floor_db: float = -40.0,
    gain_smoothing_ms: float = 0.0,
    pulse_halo_ms: float = 2.0,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
    # Stage 4 params
    target_dbfs: float = -1.0,
    fade_ms: float = 50.0,
    trim_threshold_db: float = -80.0,
    output_length_s: Optional[float] = None,
    output_bit_depth: int = 24,
    # Output
    output_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> dict:
    """
    Full pipeline: balloon pop WAV → synthesized room impulse response.

    Parameters
    ----------
    file_path : str
        Path to the input balloon pop WAV file.
    target_sr : int or None
        Resample to this rate. None = keep original.
    onset_threshold_db : float
        Onset detection threshold in dB.
    ned_window_ms : float
        NED estimation window in ms. Default 43ms.
    balloon_diameter_cm : float or None
        Balloon diameter in cm. None = auto-detect.
    num_early_reflections : int
        Number of early reflections to detect (legacy mode only).
    ned_transition_threshold : float or None
        NED value for sparse→dense phase transition. Default 0.3.
        Set to None for legacy fixed-count early reflection detection.
    random_seed : int or None
        Random seed for reproducibility.
    iccc_window_ms : float
        ICCC estimation window in ms. Default 50ms.
    energy_window_ms : float
        Band energy smoothing window in ms. Default 10ms.
    extrapolate : bool
        Whether to extrapolate energy below noise floor.
    noise_floor_db : float
        Noise floor threshold for extrapolation.
    gain_smoothing_ms : float
        Gain function smoothing in ms. Default 0.
    pulse_halo_ms : float
        Half-width of gain halo around early pulses in ms. Default 2.0.
    f_min : float
        Lowest filter bank center frequency in Hz.
    f_max : float or None
        Highest filter bank center frequency in Hz.
    target_dbfs : float
        Output normalization level. Default -1 dBFS.
    fade_ms : float
        Fade-out duration in ms.
    trim_threshold_db : float
        Tail trimming threshold.
    output_length_s : float or None
        Force output length. None = auto.
    output_bit_depth : int
        WAV bit depth: 16, 24, or 32.
    output_path : str or None
        If specified, export the final IR to this WAV path.
    progress_callback : callable or None
        Optional callback(percent: int, message: str) for progress updates.

    Returns
    -------
    result : dict
        'ir'                : np.ndarray — final synthesized IR (mono or stereo)
        'sr'                : int — sample rate
        'is_stereo'         : bool
        'preprocessing'     : dict — Stage 0 outputs
        'echo_density'      : dict — Stage 1 outputs
        'iccc_profile'      : np.ndarray or None — Stage 2 output
        'output_path'       : str or None — path if exported
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # ===== Stage 0: Preprocessing =====
    _progress(5, "Preprocessing...")
    prep = preprocess(
        file_path,
        target_sr=target_sr,
        onset_threshold_db=onset_threshold_db,
    )
    balloon_mono = prep["balloon_mono"]
    balloon_stereo = prep["balloon_stereo"]
    sr = prep["sr"]
    onset = prep["onset_sample"]
    is_stereo = balloon_stereo is not None
    _progress(10, "Preprocessing complete")

    # ===== Stage 1: Echo Density Analysis & Synthesis =====
    _progress(15, "Analyzing echo density...")
    num_sequences = 2 if is_stereo else 1
    density_result = analyze_and_synthesize_density(
        balloon_mono, sr, onset,
        ned_window_ms=ned_window_ms,
        balloon_diameter_cm=balloon_diameter_cm,
        num_early_reflections=num_early_reflections,
        ned_transition_threshold=ned_transition_threshold,
        num_sequences=num_sequences,
        random_seed=random_seed,
    )
    transition_sample = density_result.get("transition_sample", None)
    _progress(30, "Echo density analysis complete")

    # ===== Stage 2: Spatial Character (stereo only) =====
    iccc_profile = None
    if is_stereo:
        _progress(35, "Analyzing spatial character...")
        echo_left, echo_right, iccc_profile = analyze_and_synthesize_spatial(
            balloon_stereo,
            density_result["echo_sequences"],
            sr,
            window_ms=iccc_window_ms,
        )
        _progress(50, "Spatial analysis complete")
    else:
        echo_sequence = density_result["echo_sequences"][0]

    # ===== Stage 3: Energy Shaping =====
    _progress(55, "Shaping energy...")
    if is_stereo:
        # Process each channel independently with the same balloon mono
        ir_left = shape_energy(
            balloon_mono, echo_left, sr, onset,
            energy_window_ms=energy_window_ms,
            extrapolate=extrapolate,
            noise_floor_db=noise_floor_db,
            gain_smoothing_ms=gain_smoothing_ms,
            f_min=f_min, f_max=f_max,
            transition_sample=transition_sample,
            pulse_halo_ms=pulse_halo_ms,
        )
        _progress(70, "Left channel shaped")

        ir_right = shape_energy(
            balloon_mono, echo_right, sr, onset,
            energy_window_ms=energy_window_ms,
            extrapolate=extrapolate,
            noise_floor_db=noise_floor_db,
            gain_smoothing_ms=gain_smoothing_ms,
            f_min=f_min, f_max=f_max,
            transition_sample=transition_sample,
            pulse_halo_ms=pulse_halo_ms,
        )
        ir_raw = np.column_stack([ir_left, ir_right])
        _progress(80, "Energy shaping complete")
    else:
        ir_raw = shape_energy(
            balloon_mono, echo_sequence, sr, onset,
            energy_window_ms=energy_window_ms,
            extrapolate=extrapolate,
            noise_floor_db=noise_floor_db,
            gain_smoothing_ms=gain_smoothing_ms,
            f_min=f_min, f_max=f_max,
            transition_sample=transition_sample,
            pulse_halo_ms=pulse_halo_ms,
        )
        _progress(80, "Energy shaping complete")

    # ===== Stage 4: Post-Processing =====
    _progress(85, "Post-processing...")
    ir_final = postprocess(
        ir_raw, sr,
        target_dbfs=target_dbfs,
        fade_ms=fade_ms,
        trim_threshold_db=trim_threshold_db,
        output_length_s=output_length_s,
    )
    _progress(95, "Post-processing complete")

    # ===== Export =====
    if output_path:
        export_wav(ir_final, sr, output_path, bit_depth=output_bit_depth)

    _progress(100, "Done")

    return {
        "ir": ir_final,
        "sr": sr,
        "is_stereo": is_stereo,
        "preprocessing": prep,
        "echo_density": density_result,
        "iccc_profile": iccc_profile,
        "output_path": output_path,
    }
