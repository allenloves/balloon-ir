"""
Tests for Stage 2: Spatial Character Analysis & Synthesis

Follows spec §10, "Stage 2: Spatial Character — Validation".
Tests 2a through 2c.
"""

import numpy as np
import pytest

from core.spatial import estimate_iccc, impose_correlation

SR = 48000


# ---------------------------------------------------------------------------
# Test 2a: Cross-correlation of known signals
# ---------------------------------------------------------------------------

class TestICCC:
    """
    Spec Test 2a: Cross-correlation of signals with known relationships.
    """

    def test_identical_signals(self):
        """L = R (identical) → C(t) = 1.0."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SR)

        iccc = estimate_iccc(signal, signal, SR)

        # Check center portion (avoid edge windowing effects)
        hw = int(SR * 0.05 / 2)
        center = iccc[hw:-hw]
        assert np.all(np.abs(center - 1.0) < 0.01), (
            f"Identical signals should give C≈1.0, got mean {np.mean(center):.3f}"
        )

    def test_independent_signals(self):
        """L and R are independent Gaussian noise → C(t) ≈ 0."""
        rng = np.random.default_rng(42)
        left = rng.standard_normal(SR * 4)  # long for stable stats
        right = rng.standard_normal(SR * 4)

        iccc = estimate_iccc(left, right, SR)

        hw = int(SR * 0.05 / 2)
        center = iccc[hw:-hw]
        mean_c = np.mean(np.abs(center))
        assert mean_c < 0.1, (
            f"Independent signals should give C≈0, got mean|C|={mean_c:.3f}"
        )

    def test_inverted_signals(self):
        """L = -R (inverted) → C(t) = -1.0."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SR)

        iccc = estimate_iccc(signal, -signal, SR)

        hw = int(SR * 0.05 / 2)
        center = iccc[hw:-hw]
        assert np.all(np.abs(center - (-1.0)) < 0.01), (
            f"Inverted signals should give C≈-1.0, got mean {np.mean(center):.3f}"
        )

    def test_output_range(self):
        """ICCC values should always be in [-1, 1]."""
        rng = np.random.default_rng(42)
        left = rng.standard_normal(SR)
        right = rng.standard_normal(SR)

        iccc = estimate_iccc(left, right, SR)
        assert np.all(iccc >= -1.0) and np.all(iccc <= 1.0)


# ---------------------------------------------------------------------------
# Test 2b: Correlation imposition
# ---------------------------------------------------------------------------

class TestCorrelationImposition:
    """
    Spec Test 2b: Start with independent sequences, impose a target
    correlation, verify the result matches.
    """

    def _measure_correlation(self, left, right, window_size=4800):
        """Helper: measure average zero-lag correlation."""
        n = len(left)
        # Measure over the full signal
        sum_lr = np.sum(left * right)
        sum_l_sq = np.sum(left ** 2)
        sum_r_sq = np.sum(right ** 2)
        denom = np.sqrt(sum_l_sq * sum_r_sq)
        if denom > 0:
            return sum_lr / denom
        return 0.0

    def test_impose_zero(self):
        """Impose C=0 → sequences stay independent."""
        rng = np.random.default_rng(42)
        n = SR * 4
        seq1 = rng.standard_normal(n)
        seq2 = rng.standard_normal(n)

        target = np.zeros(n)
        new_l, new_r = impose_correlation(seq1, seq2, target)

        measured = self._measure_correlation(new_l, new_r)
        assert abs(measured) < 0.05, (
            f"C=0 imposed, measured {measured:.3f}"
        )

    def test_impose_one(self):
        """Impose C=1 → sequences become identical."""
        rng = np.random.default_rng(42)
        n = SR * 4
        seq1 = rng.standard_normal(n)
        seq2 = rng.standard_normal(n)

        target = np.ones(n)
        new_l, new_r = impose_correlation(seq1, seq2, target)

        measured = self._measure_correlation(new_l, new_r)
        assert measured > 0.99, (
            f"C=1 imposed, measured {measured:.3f}"
        )

    def test_impose_half(self):
        """Impose C=0.5 → measured correlation ≈ 0.5."""
        rng = np.random.default_rng(42)
        n = SR * 4
        seq1 = rng.standard_normal(n)
        seq2 = rng.standard_normal(n)

        target = np.full(n, 0.5)
        new_l, new_r = impose_correlation(seq1, seq2, target)

        measured = self._measure_correlation(new_l, new_r)
        assert abs(measured - 0.5) < 0.05, (
            f"C=0.5 imposed, measured {measured:.3f}"
        )

    def test_impose_negative(self):
        """Impose C=-0.5 → measured correlation ≈ -0.5."""
        rng = np.random.default_rng(42)
        n = SR * 4
        seq1 = rng.standard_normal(n)
        seq2 = rng.standard_normal(n)

        target = np.full(n, -0.5)
        new_l, new_r = impose_correlation(seq1, seq2, target)

        measured = self._measure_correlation(new_l, new_r)
        assert abs(measured - (-0.5)) < 0.05, (
            f"C=-0.5 imposed, measured {measured:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2c: Time-varying correlation
# ---------------------------------------------------------------------------

class TestTimeVaryingCorrelation:
    """
    Spec Test 2c: Impose a ramp from C=1 to C=0, verify the measured
    profile roughly follows.
    """

    def test_correlation_ramp(self):
        """Ramp from C=1 to C=0 over 2 seconds — measured should follow."""
        rng = np.random.default_rng(42)
        n = SR * 2
        seq1 = rng.standard_normal(n)
        seq2 = rng.standard_normal(n)

        # Target: linearly from 1.0 to 0.0
        target = np.linspace(1.0, 0.0, n)
        new_l, new_r = impose_correlation(seq1, seq2, target)

        # Measure ICCC in sliding windows
        measured = estimate_iccc(new_l, new_r, SR, window_ms=100.0)

        # Check that the first quarter has higher correlation than the last
        q1 = np.mean(measured[:n // 4])
        q4 = np.mean(measured[3 * n // 4:])
        assert q1 > q4, (
            f"Ramp: first quarter C={q1:.2f} should exceed last quarter C={q4:.2f}"
        )

        # First quarter should be near 1.0 (target range [1.0, 0.75])
        assert q1 > 0.6, f"First quarter C={q1:.2f}, expected >0.6"

        # Last quarter should be near 0.0 (target range [0.25, 0.0])
        assert q4 < 0.4, f"Last quarter C={q4:.2f}, expected <0.4"
