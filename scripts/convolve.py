"""
Convolve a dry audio signal with an impulse response.

Usage:
    python scripts/convolve.py dry.wav ir.wav -o wet.wav
    python scripts/convolve.py dry.wav ir.wav              # outputs dry_convolved.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def main():
    parser = argparse.ArgumentParser(
        description="Convolve a dry audio signal with an impulse response."
    )
    parser.add_argument("dry", help="Dry audio file (WAV)")
    parser.add_argument("ir", help="Impulse response file (WAV)")
    parser.add_argument("-o", "--output", help="Output file (default: <dry>_convolved.wav)")
    parser.add_argument("--mono", action="store_true", help="Force mono processing")
    parser.add_argument("--normalize", type=float, default=-1.0,
                        help="Peak normalize to dBFS (default: -1.0)")

    args = parser.parse_args()

    # Read files
    dry, sr_dry = sf.read(args.dry)
    ir, sr_ir = sf.read(args.ir)

    if sr_dry != sr_ir:
        print(f"Warning: sample rate mismatch (dry={sr_dry}, ir={sr_ir})")
        print("Results may not be accurate. Consider resampling first.")

    # Convert to mono if requested
    if args.mono:
        if dry.ndim > 1:
            dry = dry[:, 0]
        if ir.ndim > 1:
            ir = ir[:, 0]

    # Handle channel combinations
    if dry.ndim == 1 and ir.ndim == 1:
        # Mono dry + mono IR
        wet = fftconvolve(dry, ir, mode="full")
    elif dry.ndim == 1 and ir.ndim > 1:
        # Mono dry + stereo IR → stereo output
        wet = np.column_stack([
            fftconvolve(dry, ir[:, ch], mode="full")
            for ch in range(ir.shape[1])
        ])
    elif dry.ndim > 1 and ir.ndim == 1:
        # Stereo dry + mono IR → stereo output
        wet = np.column_stack([
            fftconvolve(dry[:, ch], ir, mode="full")
            for ch in range(dry.shape[1])
        ])
    else:
        # Stereo dry + stereo IR → convolve per channel
        channels = min(dry.shape[1], ir.shape[1])
        wet = np.column_stack([
            fftconvolve(dry[:, ch], ir[:, ch], mode="full")
            for ch in range(channels)
        ])

    # Normalize
    peak = np.max(np.abs(wet))
    if peak > 0:
        target_linear = 10 ** (args.normalize / 20.0)
        wet = wet / peak * target_linear

    # Output path
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.dry).stem
        out_path = str(Path(args.dry).parent / f"{stem}_convolved.wav")

    sf.write(out_path, wet, sr_dry)
    print(f"Written: {out_path} ({len(wet)/sr_dry:.2f}s, {sr_dry} Hz)")


if __name__ == "__main__":
    main()
