#!/usr/bin/env python3
"""
Generate Diagnostic Plots — Balloon Pop → Room IR

Usage:
    python scripts/generate_plots.py input.wav -o plots/
    python scripts/generate_plots.py input.wav -o plots/ --fmt pdf --dpi 300
    python scripts/generate_plots.py input.wav -o plots/ --seed 42

Runs the full pipeline on the input WAV, then saves all diagnostic
plots (NED profile, waveforms, spectrograms, band energy, echo
sequence, and composite summary) to the output directory.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import process_balloon
from core.visualization import save_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline and generate diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s balloon.wav -o plots/
  %(prog)s balloon.wav -o plots/ --fmt pdf --dpi 300
  %(prog)s balloon.wav -o plots/ --seed 42 --ned-transition 0.3
        """,
    )

    # Required
    parser.add_argument("input", help="Input balloon pop WAV file")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for plots")

    # Pipeline params (subset of most useful ones)
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample to this sample rate (default: keep original)")
    parser.add_argument("--balloon-diameter", type=float, default=None,
                        help="Balloon diameter in cm (default: auto-detect)")
    parser.add_argument("--ned-transition", type=float, default=0.3,
                        help="NED threshold for sparse→dense transition (default: 0.3)")
    parser.add_argument("--pulse-halo", type=float, default=2.0,
                        help="Half-width of gain halo in ms (default: 2.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # Plot params
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"],
                        help="Image format (default: png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output resolution (default: 150)")
    parser.add_argument("--energy-window", type=float, default=10.0,
                        help="Band energy smoothing window in ms (default: 10)")

    # Also export IR
    parser.add_argument("--export-ir", default=None,
                        help="Also export the synthesized IR to this WAV path")

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ned_threshold = args.ned_transition if args.ned_transition > 0 else None

    print(f"  Input:  {input_path}")
    print(f"  Output: {args.output_dir}/")
    print()

    t0 = time.time()

    # Run pipeline
    print("  Running pipeline...")
    result = process_balloon(
        str(input_path),
        target_sr=args.target_sr,
        balloon_diameter_cm=args.balloon_diameter,
        ned_transition_threshold=ned_threshold,
        pulse_halo_ms=args.pulse_halo,
        random_seed=args.seed,
        output_path=args.export_ir,
    )

    balloon_mono = result["preprocessing"]["balloon_mono"]
    onset = result["preprocessing"]["onset_sample"]
    sr = result["sr"]

    # Generate plots
    print("  Generating plots...")
    paths = save_all_plots(
        result, balloon_mono, sr, args.output_dir,
        onset=onset,
        energy_window_ms=args.energy_window,
        fmt=args.fmt,
        dpi=args.dpi,
    )

    elapsed = time.time() - t0

    print()
    for p in paths:
        print(f"  Saved: {p}")
    if args.export_ir:
        print(f"  IR:    {args.export_ir}")
    print()
    print(f"  Total time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
