"""
Pyodide bridge for the balloon-IR pipeline.

Adapts core.pipeline.process_arrays for in-browser use: takes Float32
audio arrays from JavaScript (decoded by wav.js), normalizes and trims
them, then runs the shared stages 1–4 from core/. Also exposes a
render_plot(name) function that produces diagnostic PNGs one at a time
from cached intermediate state.
"""

import numpy as np

from core.preprocessing import detect_onset
from core.pipeline import process_arrays


# Module-level cache for the most recent run, used by render_plot() so
# JavaScript can pull plots one at a time without re-running the pipeline.
_LAST = {}


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

    # ----- Preprocessing (Stage 0, in-memory variant) -----
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

    mono = (audio_left + audio_right) / 2.0 if is_stereo else audio_left

    _p(5, "Detecting onset…")
    onset = detect_onset(mono, sr, threshold_db=onset_threshold_db)

    pre_onset_samples = int(sr * pre_onset_ms / 1000.0)
    trim_start = max(0, onset - pre_onset_samples)
    mono = mono[trim_start:]
    onset_in_trimmed = onset - trim_start
    balloon_stereo = None
    if is_stereo:
        balloon_stereo = (audio_left[trim_start:], audio_right[trim_start:])
    _p(10, "Preprocessing complete")

    # ----- Stages 1–4 (shared with the CLI path) -----
    result = process_arrays(
        mono, balloon_stereo, sr, onset_in_trimmed,
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
        progress_callback=progress,
    )
    ir_final = result["ir"]
    _p(100, "Done")

    # Stash everything plot-rendering needs so JS can fetch plots one at a
    # time afterwards (see render_plot). We don't ship this state to JS now —
    # the user wants the audio back as soon as the pipeline finishes.
    _LAST.clear()
    _LAST.update({
        "result": result,
        "balloon_mono": mono,
        "sr": sr,
        "onset": onset_in_trimmed,
        "energy_window_ms": energy_window_ms,
    })

    # Return raw float32 bytes — Pyodide hands these to JS as Uint8Array,
    # which we reinterpret as Float32Array on the other side. This avoids
    # the surprises of nested numpy → JS conversion inside dicts.
    def _f32_bytes(a):
        return np.ascontiguousarray(a, dtype=np.float32).tobytes()

    plot_names = _available_plot_names(result["is_stereo"], result["iccc_profile"])

    if ir_final.ndim == 1:
        return {
            "sr": int(sr),
            "channels": 1,
            "left": _f32_bytes(ir_final),
            "right": None,
            "plot_names": plot_names,
        }
    return {
        "sr": int(sr),
        "channels": 2,
        "left": _f32_bytes(ir_final[:, 0]),
        "right": _f32_bytes(ir_final[:, 1]),
        "plot_names": plot_names,
    }


def _available_plot_names(is_stereo, iccc_profile):
    names = [
        "summary",
        "ned_profile",
        "waveform_comparison",
        "spectrogram_comparison",
        "band_energy",
        "echo_sequence",
    ]
    if is_stereo and iccc_profile is not None:
        names.append("iccc_profile")
    return names


def render_plot(name: str, dpi: int = 100) -> str:
    """
    Render a single diagnostic plot from the most recent process_array() run.
    Returns a base64-encoded PNG string. Each call yields control back to the
    JS event loop between calls, so audio playback stays responsive.
    """
    if not _LAST:
        raise RuntimeError("No pipeline result cached — call process_array first.")

    import base64
    from io import BytesIO

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _patch_specgram(plt)

    from core.visualization import (
        plot_ned_profile,
        plot_waveform_comparison,
        plot_spectrogram_comparison,
        plot_band_energy,
        plot_echo_sequence,
        plot_iccc_profile,
        plot_summary,
    )

    result = _LAST["result"]
    balloon_mono = _LAST["balloon_mono"]
    sr = _LAST["sr"]
    onset = _LAST["onset"]
    energy_window_ms = _LAST["energy_window_ms"]
    ir = result["ir"]

    if name == "ned_profile":
        fig = plot_ned_profile(result, sr, onset=onset)
    elif name == "waveform_comparison":
        fig = plot_waveform_comparison(balloon_mono, ir, sr)
    elif name == "spectrogram_comparison":
        fig = plot_spectrogram_comparison(balloon_mono, ir, sr)
    elif name == "band_energy":
        fig = plot_band_energy(balloon_mono, ir, sr, onset=onset,
                               energy_window_ms=energy_window_ms)
    elif name == "echo_sequence":
        fig = plot_echo_sequence(result, sr, onset=onset)
    elif name == "iccc_profile":
        fig = plot_iccc_profile(result["iccc_profile"], sr)
    elif name == "summary":
        fig = plot_summary(result, balloon_mono, sr, onset=onset,
                           energy_window_ms=energy_window_ms)
    else:
        raise ValueError(f"Unknown plot name: {name}")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _patch_specgram(plt):
    """
    Replace matplotlib.axes.Axes.specgram with a chunked rFFT version that
    avoids `numpy.lib.stride_tricks.sliding_window_view` (which fails with
    `ValueError: array is too big` in WASM/Pyodide for long signals).

    Idempotent — only patches once per process.
    """
    from matplotlib.axes import Axes
    if getattr(Axes.specgram, "_pyodide_patched", False):
        return

    def specgram(self, x, NFFT=256, Fs=2, Fc=0, detrend=None, window=None,
                 noverlap=128, cmap=None, xextent=None, pad_to=None,
                 sides=None, scale_by_freq=None, mode=None, scale=None,
                 vmin=None, vmax=None, *, data=None, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        nfft = int(NFFT)
        nover = int(noverlap)
        step = nfft - nover
        if step <= 0:
            raise ValueError("noverlap must be < NFFT")

        n_segs = max(1, (len(x) - nfft) // step + 1)
        win = np.hanning(nfft)
        win_norm = (win ** 2).sum() * Fs

        Pxx = np.empty((nfft // 2 + 1, n_segs), dtype=np.float64)
        for i in range(n_segs):
            seg = x[i * step : i * step + nfft] * win
            spec = np.fft.rfft(seg, n=nfft)
            Pxx[:, i] = (spec.conj() * spec).real / win_norm

        # Match matplotlib's default 'psd' density convention; double interior bins.
        Pxx[1:-1, :] *= 2

        freqs = np.fft.rfftfreq(nfft, 1.0 / Fs) + Fc
        bins = (np.arange(n_segs) * step + nfft / 2.0) / Fs

        # Convert to dB if requested.
        Z = 10.0 * np.log10(np.maximum(Pxx, 1e-20)) if scale == "dB" else Pxx

        if xextent is None:
            xextent = (bins[0], bins[-1]) if n_segs > 1 else (0, nfft / Fs)
        extent = (xextent[0], xextent[1], freqs[0], freqs[-1])

        im = self.imshow(
            Z, cmap=cmap, extent=extent, origin="lower",
            aspect="auto", vmin=vmin, vmax=vmax,
        )
        return Pxx, freqs, bins, im

    specgram._pyodide_patched = True
    Axes.specgram = specgram
