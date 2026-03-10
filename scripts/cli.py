#!/usr/bin/env python3
"""
Balloon Pop → Room Impulse Response — Command-Line Interface

Usage:
    python scripts/cli.py input.wav -o output_ir.wav
    python scripts/cli.py input.wav -o output_ir.wav --balloon-diameter 30
    python scripts/cli.py input.wav -o output_ir.wav --no-extrapolate --bit-depth 32

Processes a balloon pop WAV recording through the full analysis-resynthesis
pipeline (Abel et al., 2010) and exports a clean, full-bandwidth room
impulse response.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import process_balloon


def make_progress_bar(width: int = 40):
    """Create a terminal progress callback."""
    def callback(pct: int, msg: str):
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r  [{bar}] {pct:3d}% {msg:<40s}", end="", flush=True)
        if pct >= 100:
            print()
    return callback


def main():
    parser = argparse.ArgumentParser(
        description="Convert a balloon pop recording into a room impulse response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s balloon.wav -o ir.wav
  %(prog)s balloon.wav -o ir.wav --balloon-diameter 30 --seed 42
  %(prog)s balloon.wav -o ir.wav --bit-depth 32 --output-length 3.0
        """,
    )

    # Required
    parser.add_argument("input", help="Input balloon pop WAV file")
    parser.add_argument("-o", "--output", required=True, help="Output IR WAV file")

    # Stage 0
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample to this sample rate (default: keep original)")
    parser.add_argument("--onset-threshold", type=float, default=-40.0,
                        help="Onset detection threshold in dB (default: -40)")

    # Stage 1
    parser.add_argument("--ned-window", type=float, default=43.0,
                        help="NED estimation window in ms (default: 43)")
    parser.add_argument("--balloon-diameter", type=float, default=None,
                        help="Balloon diameter in cm (default: auto-detect)")
    parser.add_argument("--early-reflections", type=int, default=2,
                        help="Number of early reflections to detect (legacy mode, default: 2)")
    parser.add_argument("--ned-transition", type=float, default=0.3,
                        help="NED threshold for sparse→dense transition (default: 0.3). "
                             "Set to 0 to disable NED-guided detection (legacy mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # Stage 2
    parser.add_argument("--iccc-window", type=float, default=50.0,
                        help="ICCC estimation window in ms (default: 50)")

    # Stage 3
    parser.add_argument("--energy-window", type=float, default=10.0,
                        help="Band energy smoothing window in ms (default: 10)")
    parser.add_argument("--no-extrapolate", action="store_true",
                        help="Disable energy extrapolation below noise floor")
    parser.add_argument("--noise-floor", type=float, default=-40.0,
                        help="Noise floor threshold in dB (default: -40)")
    parser.add_argument("--gain-smoothing", type=float, default=0.0,
                        help="Gain function smoothing in ms (default: 0)")
    parser.add_argument("--pulse-halo", type=float, default=2.0,
                        help="Half-width of gain halo around early pulses in ms (default: 2.0)")

    # Stage 4
    parser.add_argument("--target-dbfs", type=float, default=-1.0,
                        help="Output normalization level in dBFS (default: -1)")
    parser.add_argument("--fade", type=float, default=50.0,
                        help="Fade-out duration in ms (default: 50)")
    parser.add_argument("--trim-threshold", type=float, default=-80.0,
                        help="Tail trimming threshold in dB (default: -80)")
    parser.add_argument("--output-length", type=float, default=None,
                        help="Force output length in seconds (default: auto)")
    parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32],
                        help="Output bit depth (default: 24)")

    # UI
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not input_path.suffix.lower() in (".wav", ".wave"):
        print(f"Warning: input file may not be WAV: {input_path}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    if not args.quiet:
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print()

    progress = None if args.quiet else make_progress_bar()
    t0 = time.time()

    # ned_transition=0 means legacy mode (no NED-guided detection)
    ned_threshold = args.ned_transition if args.ned_transition > 0 else None

    result = process_balloon(
        str(input_path),
        target_sr=args.target_sr,
        onset_threshold_db=args.onset_threshold,
        ned_window_ms=args.ned_window,
        balloon_diameter_cm=args.balloon_diameter,
        num_early_reflections=args.early_reflections,
        ned_transition_threshold=ned_threshold,
        random_seed=args.seed,
        iccc_window_ms=args.iccc_window,
        energy_window_ms=args.energy_window,
        extrapolate=not args.no_extrapolate,
        noise_floor_db=args.noise_floor,
        gain_smoothing_ms=args.gain_smoothing,
        pulse_halo_ms=args.pulse_halo,
        target_dbfs=args.target_dbfs,
        fade_ms=args.fade,
        trim_threshold_db=args.trim_threshold,
        output_length_s=args.output_length,
        output_bit_depth=args.bit_depth,
        output_path=str(output_path),
        progress_callback=progress,
    )

    elapsed = time.time() - t0

    if not args.quiet:
        ir = result["ir"]
        sr = result["sr"]
        duration = len(ir) / sr
        stereo = result["is_stereo"]
        density = result["echo_density"]

        print()
        print(f"  Sample rate:      {sr} Hz")
        print(f"  Channels:         {'stereo' if stereo else 'mono'}")
        print(f"  IR duration:      {duration:.2f} s")
        print(f"  Balloon radius:   {density['balloon_radius_m']*100:.1f} cm")
        print(f"  N-wave duration:  {density['nwave_duration_s']*1000:.2f} ms")
        print(f"  Early refs:       {len(density['early_reflections'])}")
        if density.get('transition_time_ms') is not None:
            print(f"  Transition:       {density['transition_time_ms']:.1f} ms (NED-guided)")
        print(f"  Processing time:  {elapsed:.1f} s")
        print()
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
