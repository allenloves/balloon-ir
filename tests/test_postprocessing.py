"""
Tests for Stage 4: Post-Processing & Export

Follows spec §10, "Stage 4: Post-Processing — Validation".
Tests 4a through 4c.
"""

import numpy as np
import pytest
import tempfile
import soundfile as sf

from core.postprocessing import (
    normalize,
    apply_fade_out,
    trim_tail,
    export_wav,
    postprocess,
)

SR = 48000


# ---------------------------------------------------------------------------
# Test 4a: Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """Spec Test 4a: Output peak should be at specified level."""

    def test_normalize_to_minus_1dbfs(self):
        """Default normalization: peak at -1 dBFS."""
        rng = np.random.default_rng(42)
        ir = rng.standard_normal(SR) * 0.3

        result = normalize(ir, target_dbfs=-1.0)
        peak = np.max(np.abs(result))
        expected = 10.0 ** (-1.0 / 20.0)  # ≈ 0.891
        assert abs(peak - expected) < 1e-6, (
            f"Peak should be {expected:.4f}, got {peak:.4f}"
        )

    def test_normalize_custom_level(self):
        """Normalization to -6 dBFS."""
        ir = np.array([0.5, -0.3, 0.1])
        result = normalize(ir, target_dbfs=-6.0)
        peak = np.max(np.abs(result))
        expected = 10.0 ** (-6.0 / 20.0)  # ≈ 0.501
        assert abs(peak - expected) < 1e-6

    def test_normalize_silent(self):
        """All-zero signal should remain zero."""
        ir = np.zeros(SR)
        result = normalize(ir)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# Test 4b: Fade-out
# ---------------------------------------------------------------------------

class TestFadeOut:
    """
    Spec Test 4b: Last N samples should follow a smooth cosine curve
    to zero; no abrupt discontinuity at the end.
    """

    def test_fade_ends_at_zero(self):
        """The very last sample should be zero (or near-zero)."""
        ir = np.ones(SR)
        result = apply_fade_out(ir, SR, fade_ms=50.0)
        assert abs(result[-1]) < 1e-10

    def test_fade_preserves_beginning(self):
        """Samples before the fade region should be unchanged."""
        ir = np.ones(SR)
        fade_samples = int(SR * 0.05)  # 50ms
        result = apply_fade_out(ir, SR, fade_ms=50.0)

        # Everything before the fade should still be 1.0
        np.testing.assert_array_equal(result[:-fade_samples], ir[:-fade_samples])

    def test_fade_is_monotonic(self):
        """Fade region should monotonically decrease from 1 to 0."""
        ir = np.ones(SR)
        fade_samples = int(SR * 0.05)
        result = apply_fade_out(ir, SR, fade_ms=50.0)

        fade_region = result[-fade_samples:]
        # Should be monotonically non-increasing
        diffs = np.diff(fade_region)
        assert np.all(diffs <= 1e-10), "Fade should be monotonically decreasing"

    def test_fade_smooth(self):
        """Fade should be smooth (no abrupt jumps)."""
        ir = np.ones(SR)
        fade_samples = int(SR * 0.05)
        result = apply_fade_out(ir, SR, fade_ms=50.0)

        fade_region = result[-fade_samples:]
        # Max sample-to-sample change should be small
        max_jump = np.max(np.abs(np.diff(fade_region)))
        # For a 50ms cosine fade at 48kHz, max jump ≈ π/(2400) ≈ 0.0013
        assert max_jump < 0.01, f"Max jump {max_jump:.4f} is too large"

    def test_fade_stereo(self):
        """Fade should work on stereo signals."""
        ir = np.ones((SR, 2))
        result = apply_fade_out(ir, SR, fade_ms=50.0)

        assert abs(result[-1, 0]) < 1e-10
        assert abs(result[-1, 1]) < 1e-10


# ---------------------------------------------------------------------------
# Test 4c: Tail trimming
# ---------------------------------------------------------------------------

class TestTrimTail:
    """
    Spec Test 4c: IR with tail below -80dB at t=1.5s should be
    trimmed to approximately 1.5s.
    """

    def test_trim_at_threshold(self):
        """Decaying IR should be trimmed where it drops below -80dB."""
        t = np.arange(SR * 3) / SR  # 3 seconds
        # Exponential decay: hits -80dB at approximately 1.84s
        # -80dB = 20*log10(exp(-2*t)) → t = 80/(20*2*log10(e)) ≈ 4.6s
        # Use faster decay: -80dB at ~1.5s → decay_rate ≈ 80/(20*1.5*log10(e)) ≈ 6.14
        decay_rate = 80.0 / (20.0 * 1.5 * np.log10(np.e))
        ir = np.exp(-decay_rate * t)

        result = trim_tail(ir, SR, threshold_db=-80.0)

        # Should be trimmed to approximately 1.5 seconds (±0.1s)
        result_duration = len(result) / SR
        assert 1.3 < result_duration < 1.7, (
            f"Expected ~1.5s, got {result_duration:.2f}s"
        )

    def test_no_trim_if_always_loud(self):
        """Signal always above threshold should not be trimmed."""
        ir = np.ones(SR)  # constant 0 dBFS
        result = trim_tail(ir, SR, threshold_db=-80.0)
        assert len(result) == len(ir)

    def test_min_length_enforced(self):
        """Output should respect minimum length even if signal is quiet."""
        ir = np.zeros(SR)
        ir[0] = 1e-10  # tiny signal

        result = trim_tail(ir, SR, threshold_db=-80.0, min_length_s=0.5)
        assert len(result) >= int(SR * 0.5)


# ---------------------------------------------------------------------------
# Test: Export WAV
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_24bit(self):
        """Export and re-read a 24-bit WAV."""
        rng = np.random.default_rng(42)
        ir = rng.standard_normal(SR).astype(np.float64) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            export_wav(ir, SR, f.name, bit_depth=24)
            data, sr = sf.read(f.name)
            assert sr == SR
            assert len(data) == len(ir)

    def test_export_32float(self):
        """Export and re-read a 32-bit float WAV."""
        rng = np.random.default_rng(42)
        ir = rng.standard_normal(SR).astype(np.float64) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            export_wav(ir, SR, f.name, bit_depth=32)
            data, sr = sf.read(f.name)
            assert sr == SR
            # 32-bit float should preserve values closely
            np.testing.assert_allclose(data, ir, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: Full pipeline
# ---------------------------------------------------------------------------

class TestPostprocessPipeline:
    def test_pipeline_output(self):
        """Full postprocess pipeline should produce valid output."""
        rng = np.random.default_rng(42)
        t = np.arange(SR * 3) / SR
        # Decay rate ~6.14 → hits -80dB at ~1.5s, well within 3s
        ir = rng.standard_normal(len(t)) * np.exp(-6.0 * t)

        result = postprocess(ir, SR, target_dbfs=-1.0, fade_ms=50.0)

        # Should be shorter than input (tail trimmed)
        assert len(result) < len(ir)
        # Peak should be at -1 dBFS (before fade modifies the very end)
        # Check that peak is close to target
        peak = np.max(np.abs(result))
        target = 10.0 ** (-1.0 / 20.0)
        assert peak <= target + 1e-6
        # Last sample should be near zero (fade-out)
        assert abs(result[-1]) < 0.01
        # No NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_forced_length(self):
        """output_length_s should force the output duration."""
        ir = np.ones(SR * 5)
        result = postprocess(ir, SR, output_length_s=2.0)
        assert len(result) == SR * 2
