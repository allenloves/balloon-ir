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

Two entry points:
  process_balloon(file_path, ...)             — file in, IR out (used by CLI)
  process_arrays(balloon_mono, balloon_stereo, sr, onset_sample, ...)
                                              — arrays in, IR out (used by
                                                the in-browser frontend, and
                                                internally by process_balloon)
"""

import numpy as np
from typing import Optional

from core.preprocessing import preprocess
from core.echo_density import analyze_and_synthesize_density
from core.spatial import analyze_and_synthesize_spatial
from core.energy_shaping import shape_energy
from core.postprocessing import postprocess, export_wav


def process_arrays(
    balloon_mono: np.ndarray,
    balloon_stereo: Optional[tuple],
    sr: int,
    onset_sample: int,
    *,
    # Stage 1
    ned_window_ms: float = 43.0,
    balloon_diameter_cm: Optional[float] = None,
    num_early_reflections: int = 2,
    ned_transition_threshold: Optional[float] = 0.3,
    random_seed: Optional[int] = None,
    # Stage 2
    iccc_window_ms: float = 50.0,
    # Stage 3
    energy_window_ms: float = 10.0,
    extrapolate: bool = True,
    noise_floor_db: float = -40.0,
    gain_smoothing_ms: float = 0.0,
    pulse_halo_ms: float = 2.0,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
    # Stage 4
    target_dbfs: float = -1.0,
    fade_ms: float = 50.0,
    trim_threshold_db: float = -80.0,
    output_length_s: Optional[float] = None,
    progress_callback: Optional[callable] = None,
) -> dict:
    """
    Run stages 1–4 on already-preprocessed arrays.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Mono balloon signal, normalized and trimmed.
    balloon_stereo : tuple[np.ndarray, np.ndarray] or None
        (L, R) channel pair if stereo input, else None.
    sr : int
        Sample rate in Hz.
    onset_sample : int
        Onset position within `balloon_mono`.
    progress_callback : callable or None
        Optional callback(percent: int, message: str).

    Returns
    -------
    result : dict
        'ir'                : np.ndarray — final synthesized IR (mono or stereo)
        'sr'                : int
        'is_stereo'         : bool
        'echo_density'      : dict — Stage 1 outputs
        'iccc_profile'      : np.ndarray or None — Stage 2 output
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    is_stereo = balloon_stereo is not None

    # ===== Stage 1: Echo Density =====
    _progress(15, "Analyzing echo density...")
    num_sequences = 2 if is_stereo else 1
    density_result = analyze_and_synthesize_density(
        balloon_mono, sr, onset_sample,
        ned_window_ms=ned_window_ms,
        balloon_diameter_cm=balloon_diameter_cm,
        num_early_reflections=num_early_reflections,
        ned_transition_threshold=ned_transition_threshold,
        num_sequences=num_sequences,
        random_seed=random_seed,
    )
    transition_sample = density_result.get("transition_sample", None)
    _progress(30, "Echo density complete")

    # ===== Stage 2: Spatial (stereo only) =====
    iccc_profile = None
    echo_left = echo_right = None
    echo_sequence = None
    if is_stereo:
        _progress(35, "Analyzing spatial character...")
        echo_left, echo_right, iccc_profile = analyze_and_synthesize_spatial(
            balloon_stereo,
            density_result["echo_sequences"],
            sr,
            window_ms=iccc_window_ms,
        )
        _progress(50, "Spatial complete")
    else:
        echo_sequence = density_result["echo_sequences"][0]

    # ===== Stage 3: Energy Shaping =====
    _progress(55, "Shaping energy...")
    common_kwargs = dict(
        energy_window_ms=energy_window_ms,
        extrapolate=extrapolate,
        noise_floor_db=noise_floor_db,
        gain_smoothing_ms=gain_smoothing_ms,
        f_min=f_min, f_max=f_max,
        transition_sample=transition_sample,
        pulse_halo_ms=pulse_halo_ms,
    )
    if is_stereo:
        ir_left = shape_energy(balloon_mono, echo_left, sr, onset_sample, **common_kwargs)
        _progress(70, "Left channel shaped")
        ir_right = shape_energy(balloon_mono, echo_right, sr, onset_sample, **common_kwargs)
        ir_raw = np.column_stack([ir_left, ir_right])
        _progress(80, "Energy shaping complete")
    else:
        ir_raw = shape_energy(balloon_mono, echo_sequence, sr, onset_sample, **common_kwargs)
        _progress(80, "Energy shaping complete")

    # ===== Stage 4: Post-processing =====
    _progress(85, "Post-processing...")
    ir_final = postprocess(
        ir_raw, sr,
        target_dbfs=target_dbfs,
        fade_ms=fade_ms,
        trim_threshold_db=trim_threshold_db,
        output_length_s=output_length_s,
    )
    _progress(95, "Post-processing complete")

    return {
        "ir": ir_final,
        "sr": sr,
        "is_stereo": is_stereo,
        "echo_density": density_result,
        "iccc_profile": iccc_profile,
    }


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
    Full pipeline: balloon pop WAV file → synthesized room impulse response.

    Wraps :func:`process_arrays` with file I/O at the input (Stage 0:
    :func:`core.preprocessing.preprocess`) and the output (optional WAV
    export via :func:`core.postprocessing.export_wav`).

    Parameters
    ----------
    file_path : str
        Path to the input balloon pop WAV file.
    output_path : str or None
        If specified, export the final IR to this WAV path.

    See :func:`process_arrays` for the meaning of all other parameters.

    Returns
    -------
    result : dict
        Same as :func:`process_arrays`, plus:
        'preprocessing'     : dict — Stage 0 outputs
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
    _progress(10, "Preprocessing complete")

    # ===== Stages 1–4 =====
    result = process_arrays(
        prep["balloon_mono"],
        prep["balloon_stereo"],
        prep["sr"],
        prep["onset_sample"],
        ned_window_ms=ned_window_ms,
        balloon_diameter_cm=balloon_diameter_cm,
        num_early_reflections=num_early_reflections,
        ned_transition_threshold=ned_transition_threshold,
        random_seed=random_seed,
        iccc_window_ms=iccc_window_ms,
        energy_window_ms=energy_window_ms,
        extrapolate=extrapolate,
        noise_floor_db=noise_floor_db,
        gain_smoothing_ms=gain_smoothing_ms,
        pulse_halo_ms=pulse_halo_ms,
        f_min=f_min, f_max=f_max,
        target_dbfs=target_dbfs,
        fade_ms=fade_ms,
        trim_threshold_db=trim_threshold_db,
        output_length_s=output_length_s,
        progress_callback=progress_callback,
    )

    # ===== Export =====
    if output_path:
        export_wav(result["ir"], result["sr"], output_path, bit_depth=output_bit_depth)

    _progress(100, "Done")

    return {
        **result,
        "preprocessing": prep,
        "output_path": output_path,
    }
