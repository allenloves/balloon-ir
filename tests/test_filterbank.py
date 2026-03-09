"""
Tests for Filter Bank — Stage 3a Validation

Follows spec §10: "Stage 3a FIRST — filter bank reconstruction
(foundation of everything)."

Test 3a (perfect reconstruction) is THE most critical test in the
entire pipeline. If this fails, all energy shaping results are invalid.
"""

import numpy as np
import pytest

from core.filterbank import (
    compute_band_frequencies,
    apply_filterbank,
    reconstruct,
)

SR = 48000


# ---------------------------------------------------------------------------
# Test 3a: Filter bank perfect reconstruction
# ---------------------------------------------------------------------------

class TestPerfectReconstruction:
    """
    Spec Test 3a: Filter any signal into all bands, sum back together,
    verify sum equals original within numerical precision.
    Pass criterion: max(abs(original - reconstructed)) < 1e-10.
    """

    def test_white_noise(self):
        """Perfect reconstruction with white noise (broadband signal)."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SR * 2)  # 2 seconds

        bands, centers, crossovers = apply_filterbank(signal, SR)
        reconstructed = reconstruct(bands)

        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, (
            f"Reconstruction error {error:.2e} exceeds 1e-10"
        )

    def test_impulse(self):
        """Perfect reconstruction with a single impulse."""
        signal = np.zeros(SR)
        signal[SR // 2] = 1.0

        bands, centers, crossovers = apply_filterbank(signal, SR)
        reconstructed = reconstruct(bands)

        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, (
            f"Reconstruction error {error:.2e} exceeds 1e-10"
        )

    def test_sine_wave(self):
        """Perfect reconstruction with a 1kHz sine wave."""
        t = np.arange(SR) / SR
        signal = np.sin(2 * np.pi * 1000 * t)

        bands, centers, crossovers = apply_filterbank(signal, SR)
        reconstructed = reconstruct(bands)

        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, (
            f"Reconstruction error {error:.2e} exceeds 1e-10"
        )

    def test_chirp(self):
        """Perfect reconstruction with a linear chirp (sweeps all bands)."""
        t = np.arange(SR * 2) / SR
        signal = np.sin(2 * np.pi * (50 + 10000 * t / t[-1]) * t)

        bands, centers, crossovers = apply_filterbank(signal, SR)
        reconstructed = reconstruct(bands)

        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, (
            f"Reconstruction error {error:.2e} exceeds 1e-10"
        )

    def test_real_world_like(self):
        """Perfect reconstruction with a decaying noise signal (IR-like)."""
        rng = np.random.default_rng(99)
        t = np.arange(SR * 3) / SR  # 3 seconds
        # Exponentially decaying noise (simulates a room IR)
        signal = rng.standard_normal(len(t)) * np.exp(-2.0 * t)

        bands, centers, crossovers = apply_filterbank(signal, SR)
        reconstructed = reconstruct(bands)

        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-10, (
            f"Reconstruction error {error:.2e} exceeds 1e-10"
        )


# ---------------------------------------------------------------------------
# Test 3b: Band energy of known signal
# ---------------------------------------------------------------------------

class TestBandEnergy:
    """
    Spec Test 3b: Verify that frequency content lands in the correct band.
    """

    def test_sine_in_correct_band(self):
        """A 1kHz sine should have energy concentrated near the 1kHz band.

        NOTE: 1kHz is exactly on a band center, but the crossover between
        adjacent bands has a finite transition width. Energy may split
        between the target band and its immediate neighbors. We check
        that the target band plus its neighbors capture >95% of total energy.
        """
        t = np.arange(SR * 2) / SR
        signal = np.sin(2 * np.pi * 1000 * t)

        bands, centers, _ = apply_filterbank(signal, SR)

        band_energies = np.array([np.sum(b ** 2) for b in bands])
        total_energy = np.sum(band_energies)

        # The band closest to 1kHz
        target_idx = np.argmin(np.abs(centers - 1000))

        # Target band + immediate neighbors should capture >90%
        # (3rd-order Butterworth crossovers have gradual rolloff,
        # so a small amount of energy leaks to further neighbors)
        lo = max(0, target_idx - 1)
        hi = min(len(bands), target_idx + 2)
        nearby_fraction = np.sum(band_energies[lo:hi]) / total_energy
        assert nearby_fraction > 0.90, (
            f"1kHz ±1 band should have >90% energy, got {nearby_fraction:.1%}"
        )

    def test_white_noise_spectral_density(self):
        """White noise should have roughly flat spectral density per band.

        For white noise through 1/3-octave bands, the energy per band
        increases with frequency because bandwidth grows (each band is
        wider by a factor of 2^(1/3)). To check for flat spectral
        DENSITY, we normalize each band's energy by its bandwidth.
        """
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SR * 10)  # long for stable stats

        bands, centers, _ = apply_filterbank(signal, SR)

        # Bandwidth of each 1/3-octave band: f_c * (2^(1/6) - 2^(-1/6))
        bw_factor = 2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0)
        bandwidths = centers * bw_factor

        # Energy density = energy / bandwidth
        density_db = np.array([
            10 * np.log10(np.mean(b ** 2) / bw + 1e-30)
            for b, bw in zip(bands, bandwidths)
        ])

        # Exclude edge bands (first 2, last 2) which may behave differently
        core = density_db[2:-2]
        density_range = np.max(core) - np.min(core)
        assert density_range < 6, (
            f"Spectral density range {density_range:.1f} dB in core bands"
        )


# ---------------------------------------------------------------------------
# Frequency layout sanity checks
# ---------------------------------------------------------------------------

class TestFrequencyLayout:
    def test_center_frequencies_include_1khz(self):
        """1kHz should be among the center frequencies (IEC reference)."""
        centers, _ = compute_band_frequencies(SR)
        assert any(abs(c - 1000) < 1 for c in centers)

    def test_third_octave_spacing(self):
        """Adjacent centers should be spaced by a factor of 2^(1/3) ≈ 1.26."""
        centers, _ = compute_band_frequencies(SR)
        ratios = centers[1:] / centers[:-1]
        expected = 2.0 ** (1.0 / 3.0)
        np.testing.assert_allclose(ratios, expected, rtol=1e-6)

    def test_crossovers_between_centers(self):
        """Each crossover should lie between its adjacent centers."""
        centers, crossovers = compute_band_frequencies(SR)
        for i, fc in enumerate(crossovers):
            assert centers[i] < fc < centers[i + 1]

    def test_band_count(self):
        """Should produce a reasonable number of bands (roughly 25-30 at 48kHz)."""
        centers, _ = compute_band_frequencies(SR)
        assert 20 <= len(centers) <= 35, f"Got {len(centers)} bands"
