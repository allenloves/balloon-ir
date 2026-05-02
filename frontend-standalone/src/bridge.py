"""
Pyodide bridge for the balloon-IR pipeline.

Mirrors core.pipeline.process_balloon but takes an in-memory audio array
instead of a file path, so the WAV decode/encode happens in JavaScript
(via wav.js) rather than going through soundfile + Pyodide's virtual FS.
"""

import numpy as np

from core.preprocessing import detect_onset
from core.echo_density import analyze_and_synthesize_density
from core.spatial import analyze_and_synthesize_spatial
from core.energy_shaping import shape_energy
from core.postprocessing import postprocess


def process_array(
    audio_left,
    audio_right,
    sr: int,
    *,
    onset_threshold_db: float = -40.0,
    pre_onset_ms: float = 10.0,
    ned_window_ms: float = 43.0,
    balloon_diameter_cm=None,
    num_early_reflections: int = 2,
    ned_transition_threshold=0.3,
    random_seed=None,
    iccc_window_ms: float = 50.0,
    energy_window_ms: float = 10.0,
    extrapolate: bool = True,
    noise_floor_db: float = -40.0,
    gain_smoothing_ms: float = 0.0,
    pulse_halo_ms: float = 2.0,
    f_min: float = 50.0,
    f_max=None,
    target_dbfs: float = -1.0,
    fade_ms: float = 50.0,
    trim_threshold_db: float = -80.0,
    output_length_s=None,
    progress=None,
):
    """
    Run the full balloon → IR pipeline on raw float arrays.

    audio_left  : np.ndarray (mono samples)
    audio_right : np.ndarray or None (right channel for stereo input)
    sr          : sample rate in Hz
    progress    : optional callable(percent: int, message: str)
    """

    def _p(pct, msg):
        if progress is not None:
            progress(pct, msg)

    audio_left = np.asarray(audio_left, dtype=np.float64)
    is_stereo = audio_right is not None
    if is_stereo:
        audio_right = np.asarray(audio_right, dtype=np.float64)
        peak = max(np.max(np.abs(audio_left)), np.max(np.abs(audio_right)))
    else:
        peak = np.max(np.abs(audio_left))

    if peak > 0:
        audio_left = audio_left / peak
        if is_stereo:
            audio_right = audio_right / peak

    if is_stereo:
        mono = (audio_left + audio_right) / 2.0
    else:
        mono = audio_left

    _p(5, "Detecting onset…")
    onset = detect_onset(mono, sr, threshold_db=onset_threshold_db)

    pre_onset_samples = int(sr * pre_onset_ms / 1000.0)
    trim_start = max(0, onset - pre_onset_samples)
    mono = mono[trim_start:]
    onset_in_trimmed = onset - trim_start
    if is_stereo:
        audio_left = audio_left[trim_start:]
        audio_right = audio_right[trim_start:]
    _p(10, "Preprocessing complete")

    _p(15, "Analyzing echo density…")
    num_sequences = 2 if is_stereo else 1
    density_result = analyze_and_synthesize_density(
        mono, sr, onset_in_trimmed,
        ned_window_ms=ned_window_ms,
        balloon_diameter_cm=balloon_diameter_cm,
        num_early_reflections=num_early_reflections,
        ned_transition_threshold=ned_transition_threshold,
        num_sequences=num_sequences,
        random_seed=random_seed,
    )
    transition_sample = density_result.get("transition_sample", None)
    _p(30, "Echo density complete")

    iccc_profile = None
    if is_stereo:
        _p(35, "Analyzing spatial character…")
        echo_left, echo_right, iccc_profile = analyze_and_synthesize_spatial(
            (audio_left, audio_right),
            density_result["echo_sequences"],
            sr,
            window_ms=iccc_window_ms,
        )
        _p(50, "Spatial complete")
    else:
        echo_sequence = density_result["echo_sequences"][0]

    _p(55, "Shaping energy…")
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
        ir_left = shape_energy(mono, echo_left, sr, onset_in_trimmed, **common_kwargs)
        _p(70, "Left channel done")
        ir_right = shape_energy(mono, echo_right, sr, onset_in_trimmed, **common_kwargs)
        ir_raw = np.column_stack([ir_left, ir_right])
        _p(80, "Energy shaping complete")
    else:
        ir_raw = shape_energy(mono, echo_sequence, sr, onset_in_trimmed, **common_kwargs)
        _p(80, "Energy shaping complete")

    _p(85, "Post-processing…")
    ir_final = postprocess(
        ir_raw, sr,
        target_dbfs=target_dbfs,
        fade_ms=fade_ms,
        trim_threshold_db=trim_threshold_db,
        output_length_s=output_length_s,
    )
    _p(100, "Done")

    # Return raw float32 bytes — Pyodide hands these to JS as Uint8Array,
    # which we reinterpret as Float32Array on the other side. This avoids
    # the surprises of nested numpy → JS conversion inside dicts.
    def _f32_bytes(a):
        return np.ascontiguousarray(a, dtype=np.float32).tobytes()

    if ir_final.ndim == 1:
        return {
            "sr": int(sr),
            "channels": 1,
            "left": _f32_bytes(ir_final),
            "right": None,
        }
    return {
        "sr": int(sr),
        "channels": 2,
        "left": _f32_bytes(ir_final[:, 0]),
        "right": _f32_bytes(ir_final[:, 1]),
    }
