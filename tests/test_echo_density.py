"""
Tests for Stage 1: Echo Density Analysis & Synthesis

Follows the validation plan in spec §10, "Stage 1: Echo Density — Validation".
Test execution order matches the spec recommendation.
"""

import numpy as np
import pytest
from scipy.special import erfc

from core.echo_density import (
    integrate_balloon,
    compute_ned,
    compute_ned_fast,
    convert_ned_balloon_to_fullband,
    clamp_ned,
    compute_aed,
    synthesize_echo_sequence,
)

SR = 48000


def make_nwave(duration_s: float = 0.001, sr: int = SR) -> np.ndarray:
    """Create a synthetic N-wave: linear ramp from +1 to -1."""
    n = int(sr * duration_s)
    return np.linspace(1.0, -1.0, n)


# ---------------------------------------------------------------------------
# Test 1a: Integration converts N-wave to single peak
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_nwave_to_single_peak(self):
        """Spec Test 1a: integrating an N-wave yields a single positive bump."""
        nwave = make_nwave(duration_s=0.001, sr=SR)
        result = integrate_balloon(nwave)

        # Result should be a single positive bump (parabolic shape)
        assert np.max(result) > 0, "Integrated N-wave should have positive peak"
        # The integral of a linear ramp from +1 to -1 is a parabola opening
        # downward, peaking in the middle, starting and ending near 0.
        # All values should be >= 0 (approximately)
        assert np.min(result) >= -1e-10, "Integrated N-wave should be non-negative"
        # Area under curve should be > 0
        assert np.sum(result) > 0, "Total area should be positive"

    def test_output_length(self):
        """Integration output has same length as input (initial=0)."""
        signal = np.random.randn(1000)
        result = integrate_balloon(signal)
        assert len(result) == len(signal)


# ---------------------------------------------------------------------------
# Test 1b: NED of known signals
# ---------------------------------------------------------------------------

class TestNED:
    def test_gaussian_noise_ned_near_one(self):
        """Spec Test 1b: Pure Gaussian noise should give NED ≈ 1.0."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(SR * 2)  # 2 seconds
        half_window = 1024  # ~43ms at 48kHz

        ned = compute_ned_fast(noise, half_window, SR)

        # Check the center portion (avoid edges where window is truncated)
        center = ned[half_window:-half_window]
        mean_ned = np.mean(center)
        assert abs(mean_ned - 1.0) < 0.05, (
            f"NED of Gaussian noise should be ≈1.0, got {mean_ned:.3f}"
        )

    def test_single_impulse_ned_near_zero(self):
        """Spec Test 1b: Single impulse in silence should give NED ≈ 0."""
        signal = np.zeros(SR)
        signal[SR // 2] = 1.0  # spike in the middle
        half_window = 1024

        ned = compute_ned_fast(signal, half_window, SR)

        # At the impulse location, NED should be very low
        # (one sample exceeds σ² out of 2049 samples)
        ned_at_impulse = ned[SR // 2]
        assert ned_at_impulse < 0.1, (
            f"NED at isolated impulse should be ≈0, got {ned_at_impulse:.3f}"
        )

    def test_sparse_to_dense_transition(self):
        """Spec Test 1b: NED should rise from ~0 (sparse) to ~1 (dense)."""
        rng = np.random.default_rng(42)
        n = SR * 2
        signal = np.zeros(n)

        # First half: sparse impulses (every ~50ms)
        for i in range(0, n // 2, int(SR * 0.05)):
            signal[i] = rng.standard_normal()

        # Second half: dense Gaussian noise
        signal[n // 2:] = rng.standard_normal(n // 2)

        half_window = 1024
        ned = compute_ned_fast(signal, half_window, SR)

        # NED in first quarter should be lower than NED in last quarter
        q1 = np.mean(ned[half_window:n // 4])
        q4 = np.mean(ned[3 * n // 4:-half_window])
        assert q4 > q1, (
            f"NED should increase from sparse to dense: q1={q1:.3f}, q4={q4:.3f}"
        )

    def test_fast_matches_reference(self):
        """compute_ned_fast should match compute_ned within tolerance."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(4000)
        half_window = 256

        ned_ref = compute_ned(signal, half_window, SR)
        ned_fast = compute_ned_fast(signal, half_window, SR)

        np.testing.assert_allclose(ned_fast, ned_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 1c: NED conversion (balloon → full-bandwidth)
# ---------------------------------------------------------------------------

class TestNEDConversion:
    def test_zero_stays_zero(self):
        """Spec Test 1c: η_b = 0 → η_h = 0."""
        eta_b = np.array([0.0])
        eta_h = convert_ned_balloon_to_fullband(eta_b, duration_ratio=10)
        assert eta_h[0] == pytest.approx(0.0)

    def test_one_stays_one(self):
        """Spec Test 1c: η_b ≈ 1 → η_h ≈ 1."""
        # Note: η_b is clamped to 0.999 internally
        eta_b = np.array([0.999])
        eta_h = convert_ned_balloon_to_fullband(eta_b, duration_ratio=10)
        assert eta_h[0] == pytest.approx(0.999, abs=0.01)

    def test_intermediate_value(self):
        """Spec Test 1c: η_b = 0.5, ratio = 10 → η_h < η_b."""
        eta_b = np.array([0.5])
        eta_h = convert_ned_balloon_to_fullband(eta_b, duration_ratio=10)
        assert eta_h[0] < 0.5, f"η_h should be < η_b, got {eta_h[0]:.4f}"

        # Cross-check: Eq. (8): η_h = 0.5 / ((1-0.5)*10 + 0.5) = 0.5/5.5 ≈ 0.0909
        expected = 0.5 / ((1.0 - 0.5) * 10 + 0.5)
        assert eta_h[0] == pytest.approx(expected, abs=1e-10)

    def test_monotonic(self):
        """Higher η_b should give higher η_h."""
        eta_b = np.linspace(0, 0.99, 100)
        eta_h = convert_ned_balloon_to_fullband(eta_b, duration_ratio=10)
        assert np.all(np.diff(eta_h) >= 0), "η_h should be monotonically increasing"


# ---------------------------------------------------------------------------
# Test 1d: NED clamping
# ---------------------------------------------------------------------------

class TestClamping:
    def test_clamp_after_threshold(self):
        """Spec Test 1d: all values after first crossing should be exactly 0.995."""
        eta_h = np.array([0.0, 0.5, 0.99, 0.996, 0.98, 0.997, 0.5])
        clamped = clamp_ned(eta_h, threshold=0.995)

        # First crossing is at index 3 (0.996 >= 0.995)
        assert clamped[0] == 0.0
        assert clamped[1] == 0.5
        assert clamped[2] == 0.99
        assert clamped[3] == 0.995
        assert clamped[4] == 0.995  # held, even though original was 0.98
        assert clamped[5] == 0.995
        assert clamped[6] == 0.995

    def test_no_clamp_if_never_reached(self):
        """If NED never reaches threshold, output equals input."""
        eta_h = np.array([0.0, 0.3, 0.5, 0.8, 0.9])
        clamped = clamp_ned(eta_h, threshold=0.995)
        np.testing.assert_array_equal(clamped, eta_h)


# ---------------------------------------------------------------------------
# Test 1e: AED sanity check
# ---------------------------------------------------------------------------

class TestAED:
    def test_hand_calculation(self):
        """Spec Test 1e: η_h = 0.5, sr = 48000 → AED = 48000."""
        eta_h = np.array([0.5])
        aed = compute_aed(eta_h, SR)
        expected = 0.5 * SR / (1.0 - 0.5)  # = 48000
        assert aed[0] == pytest.approx(expected)

    def test_zero_ned_zero_aed(self):
        """η_h = 0 → AED = 0."""
        eta_h = np.array([0.0])
        aed = compute_aed(eta_h, SR)
        assert aed[0] == 0.0

    def test_high_ned_high_aed(self):
        """η_h near 1 → very high AED."""
        eta_h = np.array([0.995])
        aed = compute_aed(eta_h, SR)
        expected = 0.995 * SR / 0.005  # = 9,552,000
        assert aed[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 1f: Echo sequence statistics
# ---------------------------------------------------------------------------

class TestEchoSequence:
    def test_pulse_count(self):
        """Spec Test 1f: constant AED=1000 → ~1000 pulses per second."""
        duration_s = 2.0
        duration_samples = int(SR * duration_s)
        aed_profile = np.full(duration_samples, 1000.0)

        rng = np.random.default_rng(42)
        seq = synthesize_echo_sequence(
            aed_profile, SR, duration_samples, rng=rng,
        )

        pulse_count = np.count_nonzero(seq)
        expected = 1000 * duration_s  # 2000 pulses
        # Allow ±15% due to Poisson randomness
        assert abs(pulse_count - expected) / expected < 0.15, (
            f"Expected ~{expected} pulses, got {pulse_count}"
        )

    def test_early_reflections_placed(self):
        """Early reflections should appear at specified positions."""
        duration_samples = SR
        aed_profile = np.full(duration_samples, 100.0)

        early_refs = [
            {"sample": 100, "amplitude": 1.0},
            {"sample": 500, "amplitude": 0.7},
        ]
        rng = np.random.default_rng(42)
        seq = synthesize_echo_sequence(
            aed_profile, SR, duration_samples,
            early_reflections=early_refs, rng=rng,
        )

        assert seq[100] == 1.0
        assert seq[500] == 0.7

    def test_different_seeds_different_sequences(self):
        """Different seeds should produce different sequences."""
        duration_samples = SR
        aed_profile = np.full(duration_samples, 500.0)

        seq1 = synthesize_echo_sequence(
            aed_profile, SR, duration_samples,
            rng=np.random.default_rng(1),
        )
        seq2 = synthesize_echo_sequence(
            aed_profile, SR, duration_samples,
            rng=np.random.default_rng(2),
        )

        assert not np.array_equal(seq1, seq2)
