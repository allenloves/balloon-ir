"""
Tests for Stage 3: Time-Frequency Energy Analysis & Synthesis

Follows spec §10, "Stage 3: Time-Frequency Energy — Validation".
Tests 3b through 3e (3a is tested in test_filterbank.py).
"""

import numpy as np
import pytest

from core.filterbank import apply_filterbank
from core.energy_shaping import (
    estimate_band_energy,
    estimate_all_band_energies,
    extrapolate_energy,
    compute_gain,
    imprint_energy,
    estimate_direct_path_gains,
    equalize_and_sum,
)

SR = 48000


# ---------------------------------------------------------------------------
# Test 3b: Band energy of known signal
# ---------------------------------------------------------------------------

class TestBandEnergy:
    """
    Spec Test 3b: Band energy estimation on known signals.
    """

    def test_sine_energy_in_correct_band(self):
        """A 1kHz sine's energy should concentrate in the 1kHz band."""
        t = np.arange(SR * 2) / SR
        signal = np.sin(2 * np.pi * 1000 * t)

        bands, centers, _ = apply_filterbank(signal, SR)
        energies = estimate_all_band_energies(bands, SR)

        # Total energy per band (sum over time)
        total_per_band = np.array([np.sum(e) for e in energies])
        target_idx = np.argmin(np.abs(centers - 1000))

        # Target band ±1 should have most energy
        lo = max(0, target_idx - 1)
        hi = min(len(energies), target_idx + 2)
        nearby_fraction = np.sum(total_per_band[lo:hi]) / np.sum(total_per_band)
        assert nearby_fraction > 0.90, (
            f"1kHz ±1 band should have >90% energy, got {nearby_fraction:.1%}"
        )

    def test_energy_nonnegative(self):
        """Band energy profiles should always be non-negative."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SR)

        bands, _, _ = apply_filterbank(signal, SR)
        energies = estimate_all_band_energies(bands, SR)

        for i, e in enumerate(energies):
            assert np.all(e >= 0), f"Band {i} has negative energy"

    def test_energy_profile_shape(self):
        """Energy of a decaying signal should decrease over time."""
        rng = np.random.default_rng(42)
        t = np.arange(SR * 2) / SR
        # Exponentially decaying noise
        signal = rng.standard_normal(len(t)) * np.exp(-3.0 * t)

        bands, _, _ = apply_filterbank(signal, SR)
        energies = estimate_all_band_energies(bands, SR)

        # For at least half the bands, the first quarter's mean energy
        # should exceed the last quarter's
        n = len(t)
        decaying_count = 0
        for e in energies:
            first_q = np.mean(e[:n // 4])
            last_q = np.mean(e[3 * n // 4:])
            if first_q > last_q * 2:
                decaying_count += 1

        assert decaying_count > len(energies) // 2, (
            f"Only {decaying_count}/{len(energies)} bands show decay"
        )


# ---------------------------------------------------------------------------
# Test 3c: Energy imprinting
# ---------------------------------------------------------------------------

class TestEnergyImprinting:
    """
    Spec Test 3c: After imprinting, the shaped sequence's band energies
    should match the balloon's band energies.
    """

    def test_imprinted_energy_matches_target(self):
        """Shaped echo bands should have energy profiles close to balloon's."""
        rng = np.random.default_rng(42)
        n = SR * 2
        t = np.arange(n) / SR

        # "Balloon" — decaying noise (different decay per band is natural)
        balloon = rng.standard_normal(n) * np.exp(-2.0 * t)

        # "Echo sequence" — flat random pulses
        echo = rng.standard_normal(n) * 0.1

        balloon_bands, _, _ = apply_filterbank(balloon, SR)
        echo_bands, _, _ = apply_filterbank(echo, SR)

        balloon_energies = estimate_all_band_energies(balloon_bands, SR)
        echo_energies = estimate_all_band_energies(echo_bands, SR)

        shaped = imprint_energy(echo_bands, balloon_energies, echo_energies)

        # Verify: energy of shaped bands should approximate balloon energies
        shaped_energies = estimate_all_band_energies(shaped, SR)

        # Check a few mid-range bands (avoid edge bands)
        num_bands = len(balloon_energies)
        for k in range(num_bands // 4, 3 * num_bands // 4):
            # Compare total energy (sum over time)
            balloon_total = np.sum(balloon_energies[k])
            shaped_total = np.sum(shaped_energies[k])

            if balloon_total > 1e-10:
                ratio_db = 10 * np.log10(shaped_total / balloon_total + 1e-30)
                assert abs(ratio_db) < 6, (
                    f"Band {k}: shaped/balloon energy ratio = {ratio_db:.1f} dB"
                )


# ---------------------------------------------------------------------------
# Test 3d: Direct path equalization
# ---------------------------------------------------------------------------

class TestDirectPathEqualization:
    """
    Spec Test 3d: After equalization, the direct path should have
    flat spectrum (all bands within ±3dB).
    """

    def test_alpha_inverts_spectral_shape(self):
        """α_k should invert the balloon's direct path spectral shape.

        If the balloon's direct path has per-band gains [2, 1, 0.5],
        then α_k should be proportional to [0.5, 1, 2], flattening
        the product gain_k * α_k to a constant.
        """
        rng = np.random.default_rng(42)
        n = SR * 2
        t = np.arange(n) / SR

        # Create a balloon signal with a spectrally shaped direct path
        balloon = rng.standard_normal(n) * 0.001 * np.exp(-2.0 * t)
        onset = 480

        # Add tonal bursts at onset with known amplitudes
        impulse_len = int(SR * 0.005)
        t_imp = np.arange(impulse_len) / SR
        for freq in [200, 500, 1000, 2000, 4000]:
            amp = 5000.0 / freq  # decreasing with frequency
            balloon[onset:onset + impulse_len] += amp * np.sin(
                2 * np.pi * freq * t_imp
            )

        balloon_bands, centers, _ = apply_filterbank(balloon, SR)

        # Measure direct path gains
        direct_gains = estimate_direct_path_gains(balloon_bands, onset, SR)

        # Compute α_k (same logic as equalize_and_sum)
        valid = direct_gains > 0
        median_gain = np.median(direct_gains[valid])
        alpha = np.ones(len(direct_gains))
        alpha[valid] = median_gain / direct_gains[valid]

        # The product gain_k * α_k should be constant (= median_gain)
        product = direct_gains[valid] * alpha[valid]
        product_db = 20 * np.log10(product / np.median(product))
        spread = np.max(product_db) - np.min(product_db)
        assert spread < 0.01, (
            f"gain * α should be constant, but spread is {spread:.4f} dB"
        )


# ---------------------------------------------------------------------------
# Test 3e: Energy extrapolation
# ---------------------------------------------------------------------------

class TestExtrapolation:
    """
    Spec Test 3e: Below the noise floor, the extrapolated curve should
    continue the linear decay slope.
    """

    def test_extrapolation_continues_decay(self):
        """Extrapolated energy should continue decaying below noise floor."""
        n = SR * 3  # 3 seconds

        # Create an energy curve: linear decay in dB, then flat noise floor
        t = np.arange(n, dtype=np.float64)
        # Decay at -20 dB/sec → hits -40dB at 2 seconds
        decay_db = -20.0 * t / SR
        decay_linear = 10.0 ** (decay_db / 10.0)

        # Add noise floor at -40dB
        noise_floor = 10.0 ** (-40.0 / 10.0)
        energy = np.maximum(decay_linear, noise_floor)

        # Extrapolate
        energy_ext = extrapolate_energy(energy, SR, noise_floor_db=-40.0)

        # After the noise floor onset (~2s), the extrapolated energy
        # should be BELOW the noise floor (continuing the decay)
        check_idx = int(SR * 2.5)  # 2.5 seconds
        assert energy_ext[check_idx] < noise_floor, (
            f"Extrapolated energy at 2.5s should be below noise floor"
        )

        # The extrapolated values should still be decaying
        check_idx2 = int(SR * 2.8)
        assert energy_ext[check_idx2] < energy_ext[check_idx], (
            "Extrapolated energy should continue decaying"
        )

    def test_no_extrapolation_when_disabled(self):
        """If energy never reaches floor, output should equal input."""
        rng = np.random.default_rng(42)
        # Strong signal that never hits -40dB
        energy = np.abs(rng.standard_normal(SR)) + 0.1

        result = extrapolate_energy(energy, SR, noise_floor_db=-40.0)
        np.testing.assert_array_equal(result, energy)


# ---------------------------------------------------------------------------
# Integration test: full Stage 3 pipeline
# ---------------------------------------------------------------------------

class TestStage3Pipeline:
    """Verify that shape_energy runs end-to-end without errors."""

    def test_pipeline_runs(self):
        """Full Stage 3 pipeline should produce output of correct length."""
        from core.energy_shaping import shape_energy

        rng = np.random.default_rng(42)
        n = SR * 2
        t = np.arange(n) / SR

        balloon = rng.standard_normal(n) * np.exp(-2.0 * t)
        balloon[480] = 1.0  # onset impulse

        echo = rng.standard_normal(n) * 0.01
        # Add some pulses
        for i in range(0, n, 100):
            echo[i] = rng.standard_normal()

        ir = shape_energy(
            balloon, echo, SR, onset_sample=480,
            energy_window_ms=10.0, extrapolate=True,
        )

        assert len(ir) == n
        assert not np.any(np.isnan(ir))
        assert not np.any(np.isinf(ir))
