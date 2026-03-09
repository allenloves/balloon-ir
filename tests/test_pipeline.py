"""
Tests for Full Pipeline (End-to-End)

Follows spec §10, "Full Pipeline Validation (End-to-End)".
Test E2E-1: Synthetic round-trip.
"""

import numpy as np
import pytest
import tempfile
import soundfile as sf

from core.pipeline import process_balloon

SR = 48000


def create_synthetic_balloon_wav(sr=SR, duration_s=2.0, rho_m=0.15):
    """
    Create a synthetic balloon pop recording with KNOWN properties.

    1. Create a known IR:
       - Direct path delta at t=0
       - Floor reflection at t=15ms, amplitude 0.7
       - Exponential noise tail with frequency-dependent decay
    2. Create a synthetic N-wave from balloon radius ρ
    3. Convolve IR with N-wave → synthetic balloon recording
    4. Write to a temp WAV file

    Returns: (file_path, known_ir, rho_m)
    """
    rng = np.random.default_rng(42)
    n = int(sr * duration_s)
    t = np.arange(n) / sr

    # --- Known IR ---
    known_ir = np.zeros(n)
    # Direct path
    known_ir[0] = 1.0
    # Floor reflection at 15ms
    floor_sample = int(sr * 0.015)
    known_ir[floor_sample] = 0.7
    # A few more early reflections
    known_ir[int(sr * 0.025)] = 0.4
    known_ir[int(sr * 0.040)] = -0.3
    # Exponential decay tail (Gaussian noise * exponential envelope)
    tail_start = int(sr * 0.05)
    tail = rng.standard_normal(n - tail_start) * np.exp(-3.0 * t[tail_start:])
    known_ir[tail_start:] += tail * 0.5

    # --- Synthetic N-wave ---
    c = 343.0  # speed of sound
    nwave_duration = 2.0 * rho_m / c
    nwave_samples = max(2, int(sr * nwave_duration))
    nwave = np.linspace(1.0, -1.0, nwave_samples)

    # --- Convolve ---
    balloon = np.convolve(known_ir, nwave, mode="full")[:n]

    # Add slight noise floor
    balloon += rng.standard_normal(n) * 1e-4

    # Prepend some silence
    silence = np.zeros(int(sr * 0.1))  # 100ms silence
    balloon_with_silence = np.concatenate([silence, balloon])

    # Write WAV
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, balloon_with_silence, sr)
    f.close()

    return f.name, known_ir, rho_m


class TestE2E:
    """
    Spec Test E2E-1: Synthetic round-trip.
    Run synthetic balloon through pipeline, verify output is a valid IR.
    """

    def test_pipeline_produces_valid_ir(self):
        """Pipeline should produce a finite, non-zero IR of reasonable length."""
        wav_path, known_ir, rho = create_synthetic_balloon_wav()

        result = process_balloon(
            wav_path,
            random_seed=42,
            num_early_reflections=2,
        )

        ir = result["ir"]
        sr = result["sr"]

        # Basic validity
        assert ir is not None
        assert len(ir) > 0
        assert not np.any(np.isnan(ir))
        assert not np.any(np.isinf(ir))
        assert sr == SR

        # Should have significant energy (not all zeros)
        assert np.max(np.abs(ir)) > 0.1

    def test_pipeline_output_length(self):
        """Output IR length should be reasonable (not too short or long)."""
        wav_path, _, _ = create_synthetic_balloon_wav(duration_s=2.0)

        result = process_balloon(wav_path, random_seed=42)
        ir = result["ir"]

        # Output should be between 0.1s and 3s for a 2s input
        duration_s = len(ir) / result["sr"]
        assert 0.1 < duration_s < 3.0, f"Output duration {duration_s:.2f}s"

    def test_pipeline_export(self):
        """Pipeline should export a valid WAV file when output_path is given."""
        wav_path, _, _ = create_synthetic_balloon_wav()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            result = process_balloon(
                wav_path,
                random_seed=42,
                output_path=out.name,
                output_bit_depth=24,
            )

            # Re-read the exported file
            data, sr = sf.read(out.name)
            assert sr == SR
            assert len(data) == len(result["ir"])

    def test_pipeline_progress_callback(self):
        """Progress callback should be called with increasing percentages."""
        wav_path, _, _ = create_synthetic_balloon_wav()

        progress_log = []

        def on_progress(pct, msg):
            progress_log.append((pct, msg))

        process_balloon(
            wav_path,
            random_seed=42,
            progress_callback=on_progress,
        )

        # Should have multiple progress updates
        assert len(progress_log) >= 5

        # Percentages should be non-decreasing
        pcts = [p for p, _ in progress_log]
        assert pcts == sorted(pcts)

        # Should start low and end at 100
        assert pcts[0] <= 10
        assert pcts[-1] == 100

    def test_pipeline_mono_result(self):
        """Mono input should produce mono output with is_stereo=False."""
        wav_path, _, _ = create_synthetic_balloon_wav()

        result = process_balloon(wav_path, random_seed=42)

        assert result["is_stereo"] is False
        assert result["ir"].ndim == 1
        assert result["iccc_profile"] is None
