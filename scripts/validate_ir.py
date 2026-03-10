#!/usr/bin/env python3
"""
IR Validation — Room Acoustic Parameter Comparison

Usage:
    python scripts/validate_ir.py ir.wav
    python scripts/validate_ir.py ir.wav --reference ref_ir.wav
    python scripts/validate_ir.py ir.wav --reference ref_ir.wav --plot params.png

Computes standard room acoustic parameters from a synthesized IR:
  - RT60 (T30 extrapolated) per 1/3-octave band
  - EDT  (Early Decay Time)
  - C80  (Clarity, early-to-late energy ratio at 80ms)
  - D50  (Definition, early-to-total energy ratio at 50ms)
  - Ts   (Centre Time)

If a reference IR is provided, prints a side-by-side comparison.

References:
  - ISO 3382-1:2009 — Measurement of room acoustic parameters
  - Kuttruff, H. (2009). Room Acoustics, 5th ed. Spon Press.
"""

import argparse
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.filterbank import apply_filterbank, compute_band_frequencies


# ---------------------------------------------------------------------------
# Schroeder backward integration
# ---------------------------------------------------------------------------

def schroeder_integral(ir: np.ndarray) -> np.ndarray:
    """
    Compute the Schroeder backward integration curve.

    The Schroeder curve represents the energy remaining in the IR
    from time t to the end, expressed in dB relative to the total
    energy. It is the standard basis for RT60 and EDT estimation.

        L(t) = 10 * log10( sum(ir[t:]^2) / sum(ir[:]^2) )

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).

    Returns
    -------
    curve_db : np.ndarray
        Schroeder curve in dB, same length as ir. Starts at 0 dB.
    """
    sq = ir ** 2
    total_energy = np.sum(sq)
    if total_energy <= 0:
        return np.full(len(ir), -np.inf)

    # Backward cumulative sum: energy from t to end
    backward_energy = np.cumsum(sq[::-1])[::-1]
    curve_db = 10.0 * np.log10(backward_energy / total_energy + 1e-30)
    return curve_db


# ---------------------------------------------------------------------------
# RT60 (via T30 extrapolation)
# ---------------------------------------------------------------------------

def estimate_rt60(
    ir: np.ndarray,
    sr: int,
    method: str = "T30",
) -> float:
    """
    Estimate RT60 from an impulse response using Schroeder integration.

    T30 method (ISO 3382-1): fit a linear regression to the Schroeder
    curve between -5 dB and -35 dB, then extrapolate to -60 dB.
    This avoids the direct path (above -5 dB) and the noise floor
    (below -35 dB).

    T20 method: fit between -5 dB and -25 dB, extrapolate to -60 dB.

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).
    sr : int
        Sample rate in Hz.
    method : str
        "T30" (default) or "T20".

    Returns
    -------
    rt60 : float
        Estimated RT60 in seconds. Returns np.nan if estimation fails.
    """
    curve_db = schroeder_integral(ir)

    if method == "T30":
        upper, lower = -5.0, -35.0
        extrap_range = 60.0 / 30.0  # multiply by 2
    elif method == "T20":
        upper, lower = -5.0, -25.0
        extrap_range = 60.0 / 20.0  # multiply by 3
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find samples in the fit range
    fit_mask = (curve_db <= upper) & (curve_db >= lower)
    fit_indices = np.where(fit_mask)[0]

    if len(fit_indices) < 10:
        return np.nan

    # Linear regression: curve_db ≈ slope * sample_index + intercept
    t_fit = fit_indices.astype(np.float64)
    db_fit = curve_db[fit_indices]
    slope, intercept = np.polyfit(t_fit, db_fit, 1)

    if slope >= 0:
        return np.nan

    # RT60 = time for the line to drop 60 dB
    # The fit covers (lower - upper) dB over some sample span.
    # Extrapolate: samples_for_60dB = 60 / |slope|
    # But using the extrapolation factor is cleaner:
    fit_drop = upper - lower  # e.g., 30 dB for T30
    fit_samples = (fit_drop) / abs(slope)
    rt60_samples = fit_samples * extrap_range
    rt60 = rt60_samples / sr

    return rt60


# ---------------------------------------------------------------------------
# EDT (Early Decay Time)
# ---------------------------------------------------------------------------

def estimate_edt(ir: np.ndarray, sr: int) -> float:
    """
    Estimate Early Decay Time (EDT) from an impulse response.

    EDT is defined as 6× the time for the Schroeder curve to drop
    from 0 dB to -10 dB (ISO 3382-1). It emphasizes the early part
    of the decay, which is perceptually more important than RT60.

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    edt : float
        Early Decay Time in seconds. Returns np.nan if estimation fails.
    """
    curve_db = schroeder_integral(ir)

    # Fit between 0 dB and -10 dB
    fit_mask = (curve_db <= 0.0) & (curve_db >= -10.0)
    fit_indices = np.where(fit_mask)[0]

    if len(fit_indices) < 5:
        return np.nan

    t_fit = fit_indices.astype(np.float64)
    db_fit = curve_db[fit_indices]
    slope, intercept = np.polyfit(t_fit, db_fit, 1)

    if slope >= 0:
        return np.nan

    # EDT = 6 × time to drop 10 dB
    samples_10db = 10.0 / abs(slope)
    edt = (samples_10db * 6.0) / sr

    return edt


# ---------------------------------------------------------------------------
# C80 (Clarity)
# ---------------------------------------------------------------------------

def compute_c80(ir: np.ndarray, sr: int) -> float:
    """
    Compute Clarity Index C80.

    C80 is the ratio of early energy (first 80ms) to late energy
    (after 80ms), in dB. Higher C80 = more "clarity" (good for
    speech and fast music); lower C80 = more reverberant.

        C80 = 10 * log10( sum(ir[0:80ms]^2) / sum(ir[80ms:]^2) )

    Typical values: -5 dB (very reverberant) to +10 dB (very dry).

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    c80 : float
        Clarity index in dB. Returns np.nan if late energy is zero.
    """
    boundary = int(sr * 0.080)
    early = np.sum(ir[:boundary] ** 2)
    late = np.sum(ir[boundary:] ** 2)

    if late <= 0:
        return np.nan

    return 10.0 * np.log10(early / late)


# ---------------------------------------------------------------------------
# D50 (Definition)
# ---------------------------------------------------------------------------

def compute_d50(ir: np.ndarray, sr: int) -> float:
    """
    Compute Definition D50.

    D50 is the ratio of early energy (first 50ms) to total energy,
    expressed as a percentage. Higher D50 = better speech intelligibility.

        D50 = sum(ir[0:50ms]^2) / sum(ir[:]^2) * 100

    Typical values: 30% (very reverberant) to 80% (dry/clear).

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    d50 : float
        Definition as a percentage (0–100). Returns np.nan if total
        energy is zero.
    """
    boundary = int(sr * 0.050)
    early = np.sum(ir[:boundary] ** 2)
    total = np.sum(ir ** 2)

    if total <= 0:
        return np.nan

    return (early / total) * 100.0


# ---------------------------------------------------------------------------
# Ts (Centre Time)
# ---------------------------------------------------------------------------

def compute_ts(ir: np.ndarray, sr: int) -> float:
    """
    Compute Centre Time Ts.

    Ts is the first moment of the squared impulse response — the
    "centre of gravity" of the energy in time. Lower Ts = energy
    concentrated early (clear); higher Ts = energy spread late
    (reverberant).

        Ts = sum(t * ir(t)^2) / sum(ir(t)^2)

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (1D).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    ts : float
        Centre time in seconds. Returns np.nan if total energy is zero.
    """
    sq = ir ** 2
    total = np.sum(sq)

    if total <= 0:
        return np.nan

    t = np.arange(len(ir), dtype=np.float64) / sr
    return np.sum(t * sq) / total


# ---------------------------------------------------------------------------
# Per-band analysis
# ---------------------------------------------------------------------------

def analyze_ir(
    ir: np.ndarray,
    sr: int,
    f_min: float = 50.0,
    f_max: Optional[float] = None,
) -> dict:
    """
    Compute all room acoustic parameters for an IR.

    Computes broadband parameters and per-band RT60.

    Parameters
    ----------
    ir : np.ndarray
        Impulse response (mono). If stereo (2D), uses left channel.
    sr : int
        Sample rate in Hz.
    f_min, f_max : float
        Filter bank frequency range for per-band analysis.

    Returns
    -------
    result : dict
        'rt60_broadband' : float — broadband RT60 (T30) in seconds
        'edt'            : float — Early Decay Time in seconds
        'c80'            : float — Clarity index in dB
        'd50'            : float — Definition in %
        'ts'             : float — Centre Time in seconds
        'rt60_bands'     : dict — {center_freq_hz: rt60_s} per band
        'centers'        : np.ndarray — band center frequencies
    """
    ir_mono = ir[:, 0] if ir.ndim == 2 else ir

    # Broadband parameters
    rt60 = estimate_rt60(ir_mono, sr)
    edt = estimate_edt(ir_mono, sr)
    c80 = compute_c80(ir_mono, sr)
    d50 = compute_d50(ir_mono, sr)
    ts = compute_ts(ir_mono, sr)

    # Per-band RT60
    if f_max is None:
        f_max = sr / 2 * 0.9
    centers, _ = compute_band_frequencies(sr, f_min, f_max)

    bands, _, _ = apply_filterbank(ir_mono, sr, f_min=f_min, f_max=f_max)
    rt60_bands = {}
    for i, (center, band) in enumerate(zip(centers, bands)):
        rt60_bands[float(center)] = estimate_rt60(band, sr)

    return {
        "rt60_broadband": rt60,
        "edt": edt,
        "c80": c80,
        "d50": d50,
        "ts": ts,
        "rt60_bands": rt60_bands,
        "centers": centers,
    }


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------

def compare_irs(
    result_a: dict,
    result_b: dict,
    label_a: str = "Synthesized",
    label_b: str = "Reference",
) -> str:
    """
    Generate a formatted comparison report between two IR analyses.

    Parameters
    ----------
    result_a, result_b : dict
        Output from analyze_ir().
    label_a, label_b : str
        Labels for the two IRs.

    Returns
    -------
    report : str
        Formatted text report.
    """
    lines = []
    lines.append(f"{'Parameter':<20s}  {label_a:>12s}  {label_b:>12s}  {'Δ':>10s}")
    lines.append("-" * 60)

    def _row(name, va, vb, unit="", fmt=".3f"):
        if np.isnan(va) or np.isnan(vb):
            delta = "N/A"
        else:
            d = va - vb
            delta = f"{d:+{fmt}} {unit}"
        sa = f"{va:{fmt}} {unit}" if not np.isnan(va) else "N/A"
        sb = f"{vb:{fmt}} {unit}" if not np.isnan(vb) else "N/A"
        lines.append(f"{name:<20s}  {sa:>12s}  {sb:>12s}  {delta:>10s}")

    _row("RT60 (T30)", result_a["rt60_broadband"], result_b["rt60_broadband"], "s")
    _row("EDT", result_a["edt"], result_b["edt"], "s")
    _row("C80", result_a["c80"], result_b["c80"], "dB", ".1f")
    _row("D50", result_a["d50"], result_b["d50"], "%", ".1f")
    _row("Ts", result_a["ts"], result_b["ts"], "s")

    lines.append("")
    lines.append(f"{'RT60 per band (T30)'}")
    lines.append(f"{'Freq (Hz)':<12s}  {label_a:>10s}  {label_b:>10s}  {'Δ':>10s}  {'Δ%':>8s}")
    lines.append("-" * 55)

    for freq in sorted(result_a["rt60_bands"].keys()):
        va = result_a["rt60_bands"].get(freq, np.nan)
        vb = result_b["rt60_bands"].get(freq, np.nan)
        if np.isnan(va) or np.isnan(vb):
            delta = "N/A"
            pct = "N/A"
        else:
            d = va - vb
            delta = f"{d:+.3f} s"
            pct = f"{d/vb*100:+.0f}%" if vb != 0 else "N/A"
        sa = f"{va:.3f} s" if not np.isnan(va) else "N/A"
        sb = f"{vb:.3f} s" if not np.isnan(vb) else "N/A"
        lines.append(f"{freq:>10.0f}    {sa:>10s}  {sb:>10s}  {delta:>10s}  {pct:>8s}")

    return "\n".join(lines)


def format_single(result: dict) -> str:
    """Format a single IR analysis result."""
    lines = []
    lines.append(f"  RT60 (T30):  {result['rt60_broadband']:.3f} s")
    lines.append(f"  EDT:         {result['edt']:.3f} s")
    lines.append(f"  C80:         {result['c80']:.1f} dB")
    lines.append(f"  D50:         {result['d50']:.1f} %")
    lines.append(f"  Ts:          {result['ts']*1000:.1f} ms")
    lines.append("")
    lines.append("  RT60 per 1/3-octave band (T30):")
    lines.append(f"  {'Freq (Hz)':<12s} {'RT60 (s)':>10s}")
    lines.append("  " + "-" * 25)
    for freq in sorted(result["rt60_bands"].keys()):
        rt = result["rt60_bands"][freq]
        rt_str = f"{rt:.3f}" if not np.isnan(rt) else "N/A"
        lines.append(f"  {freq:>10.0f}   {rt_str:>10s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    result_a: dict,
    result_b: dict,
    label_a: str = "Synthesized",
    label_b: str = "Reference",
    save_path: Optional[str] = None,
):
    """
    Plot per-band RT60 comparison bar chart.

    Parameters
    ----------
    result_a, result_b : dict
        Output from analyze_ir().
    label_a, label_b : str
        Labels for the two IRs.
    save_path : str or None
        If provided, save the plot to this path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = sorted(result_a["rt60_bands"].keys())
    rt60_a = [result_a["rt60_bands"].get(f, np.nan) for f in freqs]
    rt60_b = [result_b["rt60_bands"].get(f, np.nan) for f in freqs]

    x = np.arange(len(freqs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, rt60_a, width, label=label_a, color="#d62728", alpha=0.8)
    ax.bar(x + width / 2, rt60_b, width, label=label_b, color="#1f77b4", alpha=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("RT60 (s)")
    ax.set_title("RT60 per 1/3-Octave Band")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0f}" for f in freqs], rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute room acoustic parameters from an IR WAV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s ir.wav
  %(prog)s ir.wav --reference ref_ir.wav
  %(prog)s ir.wav --reference ref_ir.wav --plot comparison.png
        """,
    )

    parser.add_argument("input", help="Input IR WAV file")
    parser.add_argument("--reference", default=None,
                        help="Reference IR WAV file for comparison")
    parser.add_argument("--plot", default=None,
                        help="Save RT60 comparison plot to this path")

    args = parser.parse_args()

    # Load input IR
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ir, sr = sf.read(str(input_path))
    print(f"\n  Input: {input_path}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(ir)/sr:.2f} s")
    print(f"  Channels: {'stereo' if ir.ndim == 2 else 'mono'}")
    print()

    result_a = analyze_ir(ir, sr)

    if args.reference:
        # Load reference IR
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"Error: file not found: {ref_path}", file=sys.stderr)
            sys.exit(1)

        ir_ref, sr_ref = sf.read(str(ref_path))
        if sr_ref != sr:
            print(f"Warning: sample rate mismatch ({sr} vs {sr_ref})",
                  file=sys.stderr)

        print(f"  Reference: {ref_path}")
        print(f"  Duration: {len(ir_ref)/sr_ref:.2f} s")
        print()

        result_b = analyze_ir(ir_ref, sr_ref)
        print(compare_irs(result_a, result_b))

        if args.plot:
            plot_comparison(result_a, result_b, save_path=args.plot)
    else:
        print(format_single(result_a))

    print()


if __name__ == "__main__":
    main()
