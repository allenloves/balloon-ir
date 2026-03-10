"""
Visualization Module

Generates diagnostic and comparison plots for the balloon pop → room IR
synthesis pipeline (Abel et al., 2010, AES Convention Paper 8171).

Provides individual plot functions and a composite summary generator.
Each function accepts the pipeline result dict (from process_balloon())
plus the original balloon signal, and returns a matplotlib Figure.

Plots:
  1. NED profile — balloon η_b vs full-bandwidth η_h, with transition
     point and early reflections marked
  2. Waveform comparison — balloon recording vs synthesized IR
  3. Spectrogram comparison — balloon vs IR side by side
  4. Band energy decay — per-band energy envelopes for balloon and IR
  5. ICCC profile — inter-channel cross-correlation (stereo only)
  6. Composite summary — all plots on a single figure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

from core.filterbank import compute_band_frequencies, apply_filterbank
from core.energy_shaping import estimate_band_energy, estimate_all_band_energies


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
_STYLE = {
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}


def _apply_style():
    """Apply consistent plot style."""
    plt.rcParams.update(_STYLE)


def _time_axis(n_samples: int, sr: int) -> np.ndarray:
    """Create a time axis in milliseconds."""
    return np.arange(n_samples) / sr * 1000


# ---------------------------------------------------------------------------
# 1. NED Profile
# ---------------------------------------------------------------------------
def plot_ned_profile(
    result: dict,
    sr: int,
    onset: int = 0,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot Normalized Echo Density profiles.

    Shows balloon NED η_b(t) and converted full-bandwidth NED η_h(t)
    on the same axes. Marks the sparse→dense transition point and
    detected early reflection positions.

    Parameters
    ----------
    result : dict
        Pipeline result from process_balloon().
    sr : int
        Sample rate in Hz.
    onset : int
        Onset sample position, used to convert onset-relative times
        (from early_reflections and transition_time_ms) to absolute.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    density = result["echo_density"]
    ned_b = density["ned_balloon"]
    ned_h = density["ned_fullband"]
    t_ms = _time_axis(len(ned_b), sr)
    onset_ms = onset / sr * 1000

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_ms, ned_b, color="#1f77b4", alpha=0.6, linewidth=0.8,
            label="η_b (balloon NED)")
    ax.plot(t_ms, ned_h, color="#d62728", linewidth=1.2,
            label="η_h (full-bandwidth NED)")

    # Mark transition point (onset-relative → absolute)
    trans_ms = density.get("transition_time_ms")
    if trans_ms is not None:
        trans_abs = onset_ms + trans_ms
        ax.axvline(trans_abs, color="#2ca02c", linestyle="--", linewidth=1,
                   label=f"Transition @ {trans_ms:.1f} ms (re onset)")

    # Mark early reflections (onset-relative → absolute)
    refs = density.get("early_reflections", [])
    if refs:
        ref_times_abs = [onset_ms + r["time_ms"] for r in refs]
        ref_ned = np.interp(ref_times_abs, t_ms, ned_h)
        ax.scatter(ref_times_abs, ref_ned, marker="v", color="#ff7f0e", s=30,
                   zorder=5, label=f"Early reflections ({len(refs)})")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("NED")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Normalized Echo Density Profile")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Waveform Comparison
# ---------------------------------------------------------------------------
def plot_waveform_comparison(
    balloon_mono: np.ndarray,
    ir: np.ndarray,
    sr: int,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot balloon recording and synthesized IR waveforms.

    Two subplots sharing the same time axis for direct visual comparison
    of temporal structure.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Original balloon pop recording (mono).
    ir : np.ndarray
        Synthesized impulse response. If stereo (2D), only the left
        channel is plotted.
    sr : int
        Sample rate in Hz.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    ir_mono = ir[:, 0] if ir.ndim == 2 else ir
    t_balloon = _time_axis(len(balloon_mono), sr)
    t_ir = _time_axis(len(ir_mono), sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    ax1.plot(t_balloon, balloon_mono, color="#1f77b4", linewidth=0.3)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Balloon Pop Recording")
    ax1.set_xlim(0, t_balloon[-1])

    ax2.plot(t_ir, ir_mono, color="#d62728", linewidth=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Synthesized Impulse Response")
    ax2.set_xlim(0, t_ir[-1])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Spectrogram Comparison
# ---------------------------------------------------------------------------
def plot_spectrogram_comparison(
    balloon_mono: np.ndarray,
    ir: np.ndarray,
    sr: int,
    figsize: tuple = (10, 5),
    dynamic_range_db: float = 80.0,
) -> plt.Figure:
    """
    Side-by-side spectrograms of balloon recording and synthesized IR.

    Uses matplotlib's specgram with a consistent color scale and dynamic
    range for fair comparison.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Original balloon pop recording (mono).
    ir : np.ndarray
        Synthesized impulse response.
    sr : int
        Sample rate in Hz.
    figsize : tuple
        Figure size in inches.
    dynamic_range_db : float
        Dynamic range for color mapping.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    ir_mono = ir[:, 0] if ir.ndim == 2 else ir

    # NFFT and overlap for reasonable resolution
    nfft = min(2048, len(balloon_mono) // 4)
    nfft = max(256, nfft)
    noverlap = nfft // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Balloon spectrogram
    Pxx1, freqs1, bins1, im1 = ax1.specgram(
        balloon_mono, NFFT=nfft, Fs=sr, noverlap=noverlap,
        cmap="inferno", scale="dB",
    )
    ax1.set_title("Balloon Pop")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(0, sr / 2)

    # IR spectrogram
    Pxx2, freqs2, bins2, im2 = ax2.specgram(
        ir_mono, NFFT=nfft, Fs=sr, noverlap=noverlap,
        cmap="inferno", scale="dB",
    )
    ax2.set_title("Synthesized IR")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, sr / 2)

    # Consistent color scale
    vmax = max(im1.get_clim()[1], im2.get_clim()[1])
    vmin = vmax - dynamic_range_db
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)

    fig.colorbar(im2, ax=[ax1, ax2], label="Power (dB)", shrink=0.8)
    fig.subplots_adjust(left=0.08, right=0.88, wspace=0.25)
    return fig


# ---------------------------------------------------------------------------
# 4. Band Energy Decay
# ---------------------------------------------------------------------------
def plot_band_energy(
    balloon_mono: np.ndarray,
    ir: np.ndarray,
    sr: int,
    onset: int = 0,
    energy_window_ms: float = 10.0,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
    bands_to_show: Optional[list] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot 1/3-octave band energy decay curves for balloon and IR.

    Re-runs the filter bank and energy estimation on both signals to
    produce per-band energy envelopes (in dB), plotted on a shared axes.
    A subset of representative bands is shown to avoid visual clutter.

    Parameters
    ----------
    balloon_mono : np.ndarray
        Original balloon pop recording (mono).
    ir : np.ndarray
        Synthesized impulse response.
    sr : int
        Sample rate in Hz.
    onset : int
        Onset sample position.
    energy_window_ms : float
        Smoothing window for band energy estimation.
    f_min, f_max : float
        Filter bank frequency range.
    bands_to_show : list or None
        Indices of bands to plot. None = auto-select ~6 representative bands.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    ir_mono = ir[:, 0] if ir.ndim == 2 else ir

    if f_max is None:
        f_max = sr / 2 * 0.9

    centers, crossovers = compute_band_frequencies(sr, f_min, f_max)
    n_bands = len(centers)

    # Filter and compute energy for balloon
    balloon_bands_list, _, _ = apply_filterbank(balloon_mono, sr,
                                                f_min=f_min, f_max=f_max)
    balloon_energy = [
        estimate_band_energy(b, sr, window_ms=energy_window_ms)
        for b in balloon_bands_list
    ]

    # Filter and compute energy for IR
    ir_bands_list, _, _ = apply_filterbank(ir_mono, sr,
                                           f_min=f_min, f_max=f_max)
    ir_energy = [
        estimate_band_energy(b, sr, window_ms=energy_window_ms)
        for b in ir_bands_list
    ]

    # Select bands to show
    if bands_to_show is None:
        # Pick ~6 evenly spaced bands
        step = max(1, n_bands // 6)
        bands_to_show = list(range(0, n_bands, step))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(bands_to_show) - 1))
              for i in range(len(bands_to_show))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for idx, band_i in enumerate(bands_to_show):
        if band_i >= n_bands:
            continue
        freq_label = f"{centers[band_i]:.0f} Hz"
        color = colors[idx]

        # Balloon band energy (dB)
        e_b = balloon_energy[band_i]
        e_b_db = 10 * np.log10(np.maximum(e_b, 1e-20))
        t_b = _time_axis(len(e_b_db), sr)
        ax1.plot(t_b, e_b_db, color=color, linewidth=0.8, label=freq_label)

        # IR band energy (dB)
        e_ir = ir_energy[band_i]
        e_ir_db = 10 * np.log10(np.maximum(e_ir, 1e-20))
        t_ir = _time_axis(len(e_ir_db), sr)
        ax2.plot(t_ir, e_ir_db, color=color, linewidth=0.8, label=freq_label)

    ax1.set_title("Balloon — Band Energy")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Energy (dB)")
    ax1.legend(fontsize=7, loc="upper right")

    ax2.set_title("Synthesized IR — Band Energy")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(fontsize=7, loc="upper right")

    # Set reasonable y-axis range
    ax1.set_ylim(bottom=-100)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. ICCC Profile (stereo only)
# ---------------------------------------------------------------------------
def plot_iccc_profile(
    iccc_profile: np.ndarray,
    sr: int,
    figsize: tuple = (10, 3),
) -> plt.Figure:
    """
    Plot Inter-Channel Cross-Correlation profile.

    Shows how the spatial correlation between L and R channels evolves
    over time. Values near 1 = mono-like (directional); near 0 =
    diffuse (enveloping).

    Parameters
    ----------
    iccc_profile : np.ndarray
        ICCC profile C(t) from Stage 2 (values in [-1, 1]).
    sr : int
        Sample rate in Hz.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    t_ms = _time_axis(len(iccc_profile), sr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_ms, iccc_profile, color="#9467bd", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ICCC")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Inter-Channel Cross-Correlation (ICCC)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Echo Sequence Detail
# ---------------------------------------------------------------------------
def plot_echo_sequence(
    result: dict,
    sr: int,
    onset: int = 0,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot the synthesized echo sequence with early reflections marked.

    Shows the pulse sequence from Stage 1, highlighting the manually
    placed early reflections and the transition point where Poisson
    synthesis begins.

    Parameters
    ----------
    result : dict
        Pipeline result from process_balloon().
    sr : int
        Sample rate in Hz.
    onset : int
        Onset sample position, used to convert onset-relative times
        to absolute.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()
    density = result["echo_density"]
    sequences = density.get("echo_sequences", [])
    if not sequences:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Echo Sequence (no data)")
        return fig

    echo_seq = sequences[0]
    t_ms = _time_axis(len(echo_seq), sr)
    onset_ms = onset / sr * 1000

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_ms, echo_seq, color="#1f77b4", linewidth=0.3, alpha=0.7)

    # Mark early reflections (onset-relative → absolute)
    refs = density.get("early_reflections", [])
    for r in refs:
        ax.axvline(onset_ms + r["time_ms"], color="#ff7f0e", linewidth=0.6,
                   alpha=0.5)

    # Mark transition (onset-relative → absolute)
    trans_ms = density.get("transition_time_ms")
    if trans_ms is not None:
        ax.axvline(onset_ms + trans_ms, color="#2ca02c", linestyle="--",
                   linewidth=1.2,
                   label=f"Transition @ {trans_ms:.1f} ms (re onset)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Synthesized Echo Sequence")
    if trans_ms is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Composite Summary
# ---------------------------------------------------------------------------
def plot_summary(
    result: dict,
    balloon_mono: np.ndarray,
    sr: int,
    onset: int = 0,
    energy_window_ms: float = 10.0,
    figsize: tuple = (14, 16),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a composite summary figure with all diagnostic plots.

    Layout (6 rows):
      Row 1: NED profile
      Row 2: Waveform comparison (balloon | IR)
      Row 3: Spectrogram comparison (balloon | IR)
      Row 4: Band energy decay (balloon | IR)
      Row 5: Echo sequence detail
      Row 6: ICCC profile (if stereo, otherwise empty)

    Parameters
    ----------
    result : dict
        Pipeline result from process_balloon().
    balloon_mono : np.ndarray
        Original balloon pop recording (mono).
    sr : int
        Sample rate in Hz.
    onset : int
        Onset sample in balloon_mono.
    energy_window_ms : float
        Smoothing window for band energy.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If provided, save the figure to this path (PNG, PDF, etc.).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _apply_style()

    ir = result["ir"]
    ir_mono = ir[:, 0] if ir.ndim == 2 else ir
    density = result["echo_density"]
    is_stereo = result["is_stereo"]

    n_rows = 6 if is_stereo else 5
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- Row 1: NED Profile (full width) ---
    ax_ned = fig.add_subplot(gs[0, :])
    ned_b = density["ned_balloon"]
    ned_h = density["ned_fullband"]
    t_ned = _time_axis(len(ned_b), sr)
    onset_ms = onset / sr * 1000
    ax_ned.plot(t_ned, ned_b, color="#1f77b4", alpha=0.6, linewidth=0.8,
                label="η_b (balloon)")
    ax_ned.plot(t_ned, ned_h, color="#d62728", linewidth=1.2,
                label="η_h (full-bandwidth)")
    trans_ms = density.get("transition_time_ms")
    if trans_ms is not None:
        ax_ned.axvline(onset_ms + trans_ms, color="#2ca02c", linestyle="--",
                       linewidth=1, label=f"Transition @ {trans_ms:.1f} ms (re onset)")
    refs = density.get("early_reflections", [])
    if refs:
        ref_times_abs = [onset_ms + r["time_ms"] for r in refs]
        ref_ned = np.interp(ref_times_abs, t_ned, ned_h)
        ax_ned.scatter(ref_times_abs, ref_ned, marker="v", color="#ff7f0e",
                       s=25, zorder=5, label=f"Early refs ({len(refs)})")
    ax_ned.set_ylim(-0.05, 1.15)
    ax_ned.set_xlabel("Time (ms)")
    ax_ned.set_ylabel("NED")
    ax_ned.set_title("Normalized Echo Density")
    ax_ned.legend(fontsize=7, loc="lower right")

    # --- Row 2: Waveforms ---
    t_balloon = _time_axis(len(balloon_mono), sr)
    t_ir = _time_axis(len(ir_mono), sr)

    ax_wb = fig.add_subplot(gs[1, 0])
    ax_wb.plot(t_balloon, balloon_mono, color="#1f77b4", linewidth=0.3)
    ax_wb.set_title("Balloon Pop")
    ax_wb.set_ylabel("Amplitude")
    ax_wb.set_xlabel("Time (ms)")

    ax_wi = fig.add_subplot(gs[1, 1])
    ax_wi.plot(t_ir, ir_mono, color="#d62728", linewidth=0.3)
    ax_wi.set_title("Synthesized IR")
    ax_wi.set_ylabel("Amplitude")
    ax_wi.set_xlabel("Time (ms)")

    # --- Row 3: Spectrograms ---
    nfft = min(2048, len(balloon_mono) // 4)
    nfft = max(256, nfft)
    noverlap = nfft // 2

    ax_sb = fig.add_subplot(gs[2, 0])
    _, _, _, im1 = ax_sb.specgram(
        balloon_mono, NFFT=nfft, Fs=sr, noverlap=noverlap,
        cmap="inferno", scale="dB")
    ax_sb.set_title("Balloon — Spectrogram")
    ax_sb.set_ylabel("Freq (Hz)")
    ax_sb.set_xlabel("Time (s)")
    ax_sb.set_ylim(0, sr / 2)

    ax_si = fig.add_subplot(gs[2, 1])
    _, _, _, im2 = ax_si.specgram(
        ir_mono, NFFT=nfft, Fs=sr, noverlap=noverlap,
        cmap="inferno", scale="dB")
    ax_si.set_title("IR — Spectrogram")
    ax_si.set_ylabel("Freq (Hz)")
    ax_si.set_xlabel("Time (s)")
    ax_si.set_ylim(0, sr / 2)

    # Match color scales
    vmax = max(im1.get_clim()[1], im2.get_clim()[1])
    vmin = vmax - 80
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)

    # --- Row 4: Band Energy Decay ---
    f_max = sr / 2 * 0.9
    centers, crossovers = compute_band_frequencies(sr, 50.0, f_max)
    n_bands = len(centers)
    step = max(1, n_bands // 6)
    show_bands = list(range(0, n_bands, step))

    balloon_bands_list, _, _ = apply_filterbank(balloon_mono, sr,
                                                f_min=50.0, f_max=f_max)
    ir_bands_list, _, _ = apply_filterbank(ir_mono, sr,
                                           f_min=50.0, f_max=f_max)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(show_bands) - 1))
              for i in range(len(show_bands))]

    ax_eb = fig.add_subplot(gs[3, 0])
    ax_ei = fig.add_subplot(gs[3, 1], sharey=ax_eb)

    for idx, band_i in enumerate(show_bands):
        if band_i >= n_bands:
            continue
        freq_label = f"{centers[band_i]:.0f} Hz"
        color = colors[idx]

        e_b = estimate_band_energy(balloon_bands_list[band_i], sr,
                                  window_ms=energy_window_ms)
        e_b_db = 10 * np.log10(np.maximum(e_b, 1e-20))
        ax_eb.plot(_time_axis(len(e_b_db), sr), e_b_db,
                   color=color, linewidth=0.8, label=freq_label)

        e_i = estimate_band_energy(ir_bands_list[band_i], sr,
                                  window_ms=energy_window_ms)
        e_i_db = 10 * np.log10(np.maximum(e_i, 1e-20))
        ax_ei.plot(_time_axis(len(e_i_db), sr), e_i_db,
                   color=color, linewidth=0.8, label=freq_label)

    ax_eb.set_title("Balloon — Band Energy")
    ax_eb.set_xlabel("Time (ms)")
    ax_eb.set_ylabel("Energy (dB)")
    ax_eb.legend(fontsize=6, loc="upper right")
    ax_eb.set_ylim(bottom=-100)

    ax_ei.set_title("IR — Band Energy")
    ax_ei.set_xlabel("Time (ms)")
    ax_ei.legend(fontsize=6, loc="upper right")

    # --- Row 5: Echo Sequence ---
    ax_echo = fig.add_subplot(gs[4, :])
    sequences = density.get("echo_sequences", [])
    if sequences:
        echo_seq = sequences[0]
        t_echo = _time_axis(len(echo_seq), sr)
        ax_echo.plot(t_echo, echo_seq, color="#1f77b4", linewidth=0.3,
                     alpha=0.7)
        for r in refs:
            ax_echo.axvline(onset_ms + r["time_ms"], color="#ff7f0e",
                            linewidth=0.5, alpha=0.5)
        if trans_ms is not None:
            ax_echo.axvline(onset_ms + trans_ms, color="#2ca02c",
                            linestyle="--", linewidth=1,
                            label=f"Transition @ {trans_ms:.1f} ms (re onset)")
    ax_echo.set_xlabel("Time (ms)")
    ax_echo.set_ylabel("Amplitude")
    ax_echo.set_title("Echo Sequence (Stage 1 output)")
    if trans_ms is not None and sequences:
        ax_echo.legend(fontsize=7)

    # --- Row 6: ICCC (stereo only) ---
    if is_stereo and result.get("iccc_profile") is not None:
        ax_iccc = fig.add_subplot(gs[5, :])
        iccc = result["iccc_profile"]
        t_iccc = _time_axis(len(iccc), sr)
        ax_iccc.plot(t_iccc, iccc, color="#9467bd", linewidth=0.8)
        ax_iccc.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax_iccc.set_xlabel("Time (ms)")
        ax_iccc.set_ylabel("ICCC")
        ax_iccc.set_ylim(-1.1, 1.1)
        ax_iccc.set_title("Inter-Channel Cross-Correlation")

    # --- Title ---
    balloon_r = density.get("balloon_radius_m", 0) * 100
    nwave_ms = density.get("nwave_duration_s", 0) * 1000
    fig.suptitle(
        f"Balloon IR Synthesis Summary  |  sr={sr} Hz  |  "
        f"balloon ρ={balloon_r:.1f} cm  |  N-wave={nwave_ms:.2f} ms  |  "
        f"{'stereo' if is_stereo else 'mono'}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Convenience: save individual plots
# ---------------------------------------------------------------------------
def save_all_plots(
    result: dict,
    balloon_mono: np.ndarray,
    sr: int,
    output_dir: str,
    onset: int = 0,
    energy_window_ms: float = 10.0,
    fmt: str = "png",
    dpi: int = 150,
) -> list:
    """
    Save all individual plots to a directory.

    Parameters
    ----------
    result : dict
        Pipeline result from process_balloon().
    balloon_mono : np.ndarray
        Original balloon pop recording (mono).
    sr : int
        Sample rate in Hz.
    output_dir : str
        Directory to save plots.
    onset : int
        Onset sample position.
    energy_window_ms : float
        Smoothing window for band energy plots.
    fmt : str
        Image format (png, pdf, svg).
    dpi : int
        Output resolution.

    Returns
    -------
    paths : list[str]
        List of saved file paths.
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    ir = result["ir"]

    # 1. NED
    fig = plot_ned_profile(result, sr, onset=onset)
    p = str(out / f"ned_profile.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 2. Waveform
    fig = plot_waveform_comparison(balloon_mono, ir, sr)
    p = str(out / f"waveform_comparison.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3. Spectrogram
    fig = plot_spectrogram_comparison(balloon_mono, ir, sr)
    p = str(out / f"spectrogram_comparison.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4. Band energy
    fig = plot_band_energy(balloon_mono, ir, sr, onset=onset,
                           energy_window_ms=energy_window_ms)
    p = str(out / f"band_energy.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 5. Echo sequence
    fig = plot_echo_sequence(result, sr, onset=onset)
    p = str(out / f"echo_sequence.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 6. ICCC (stereo only)
    if result["is_stereo"] and result.get("iccc_profile") is not None:
        fig = plot_iccc_profile(result["iccc_profile"], sr)
        p = str(out / f"iccc_profile.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 7. Composite summary
    fig = plot_summary(result, balloon_mono, sr, onset=onset,
                       energy_window_ms=energy_window_ms)
    p = str(out / f"summary.{fmt}")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths
