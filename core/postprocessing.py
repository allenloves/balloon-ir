"""
Post-Processing & Export (Stage 4)

Implements the final output stage described in §6 of Abel et al. (2010)
"Estimating Room Impulse Responses from Recorded Balloon Pops,"
AES Convention Paper 8171.

This module takes the raw synthesized impulse response from Stage 3
and prepares it for export as a WAV file suitable for use in convolution
reverberators. The operations are:

  1. Normalize to a target peak level (e.g., -1 dBFS)
  2. Apply a cosine fade-out at the tail to avoid clicks
  3. Trim trailing silence below a threshold (e.g., -80 dB)
  4. Export as WAV (16/24-bit integer or 32-bit float)
  5. Optionally compute room acoustic parameters (RT60, EDT, C80, D50)
"""

import numpy as np
import soundfile as sf
from typing import Optional


# ---------------------------------------------------------------------------
# 4a. Normalization
# ---------------------------------------------------------------------------

def normalize(
    ir: np.ndarray,
    target_dbfs: float = -1.0,
) -> np.ndarray:
    """
    Normalize the impulse response to a target peak level.

    Parameters
    ----------
    ir : np.ndarray
        Input impulse response (mono or stereo).
    target_dbfs : float
        Target peak level in dBFS. Default -1 dBFS (peak ≈ 0.891).

    Returns
    -------
    ir_norm : np.ndarray
        Normalized impulse response.
    """
    target_linear = 10.0 ** (target_dbfs / 20.0)
    peak = np.max(np.abs(ir))
    if peak > 0:
        return ir * (target_linear / peak)
    return ir.copy()


# ---------------------------------------------------------------------------
# 4b. Fade-out
# ---------------------------------------------------------------------------

def apply_fade_out(
    ir: np.ndarray,
    sr: int,
    fade_ms: float = 50.0,
) -> np.ndarray:
    """
    Apply a cosine fade-out at the end of the IR to avoid clicks.

    A half-cosine window smoothly brings the signal from full amplitude
    to zero over the specified duration. This prevents discontinuities
    when the IR is loaded into a convolution reverberator.

    Parameters
    ----------
    ir : np.ndarray
        Input impulse response, shape (num_samples,) or (num_samples, 2).
    sr : int
        Sample rate in Hz.
    fade_ms : float
        Fade-out duration in milliseconds. Default 50ms.

    Returns
    -------
    ir_faded : np.ndarray
        Impulse response with fade-out applied.
    """
    fade_samples = int(sr * fade_ms / 1000.0)
    fade_samples = min(fade_samples, len(ir))

    ir_faded = ir.copy()

    # Half-cosine: goes from 1.0 to 0.0
    fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))

    if ir_faded.ndim == 1:
        ir_faded[-fade_samples:] *= fade_curve
    else:
        # Stereo: apply to both channels
        ir_faded[-fade_samples:, :] *= fade_curve[:, np.newaxis]

    return ir_faded


# ---------------------------------------------------------------------------
# 4c. Trim trailing silence
# ---------------------------------------------------------------------------

def trim_tail(
    ir: np.ndarray,
    sr: int,
    threshold_db: float = -80.0,
    min_length_s: float = 0.1,
) -> np.ndarray:
    """
    Trim trailing silence below a threshold.

    Finds the last sample where the signal exceeds the threshold
    (relative to peak) and trims everything after it, keeping a
    small safety margin.

    Parameters
    ----------
    ir : np.ndarray
        Input impulse response.
    sr : int
        Sample rate in Hz.
    threshold_db : float
        Silence threshold in dB relative to peak. Default -80 dB.
    min_length_s : float
        Minimum output length in seconds. Default 0.1s.

    Returns
    -------
    ir_trimmed : np.ndarray
        Trimmed impulse response.
    """
    if ir.ndim == 1:
        abs_ir = np.abs(ir)
    else:
        abs_ir = np.max(np.abs(ir), axis=1)

    peak = np.max(abs_ir)
    if peak == 0:
        return ir.copy()

    threshold_linear = peak * 10.0 ** (threshold_db / 20.0)

    # Find last sample above threshold
    above = np.where(abs_ir > threshold_linear)[0]
    if len(above) == 0:
        min_samples = int(sr * min_length_s)
        return ir[:min_samples].copy()

    last_above = above[-1]

    # Add a small margin (10ms) after the last active sample
    margin = int(sr * 0.01)
    trim_end = min(len(ir), last_above + margin + 1)

    # Enforce minimum length
    min_samples = int(sr * min_length_s)
    trim_end = max(trim_end, min_samples)

    return ir[:trim_end].copy()


# ---------------------------------------------------------------------------
# 4d. Export as WAV
# ---------------------------------------------------------------------------

def export_wav(
    ir: np.ndarray,
    sr: int,
    file_path: str,
    bit_depth: int = 24,
) -> None:
    """
    Export the impulse response as a WAV file.

    Parameters
    ----------
    ir : np.ndarray
        Impulse response to export.
    sr : int
        Sample rate in Hz.
    file_path : str
        Output file path.
    bit_depth : int
        Output bit depth: 16, 24, or 32 (float). Default 24.
    """
    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "FLOAT",
    }
    subtype = subtype_map.get(bit_depth, "PCM_24")
    sf.write(file_path, ir, sr, subtype=subtype)


# ---------------------------------------------------------------------------
# Main entry point for Stage 4
# ---------------------------------------------------------------------------

def postprocess(
    ir: np.ndarray,
    sr: int,
    target_dbfs: float = -1.0,
    fade_ms: float = 50.0,
    trim_threshold_db: float = -80.0,
    output_length_s: Optional[float] = None,
) -> np.ndarray:
    """
    Complete Stage 4 pipeline: normalize, fade, trim.

    Parameters
    ----------
    ir : np.ndarray
        Raw synthesized IR from Stage 3.
    sr : int
        Sample rate in Hz.
    target_dbfs : float
        Normalization target in dBFS. Default -1 dBFS.
    fade_ms : float
        Fade-out duration in ms. Default 50ms.
    trim_threshold_db : float
        Tail trimming threshold in dB. Default -80 dB.
    output_length_s : float or None
        If specified, force output to this length in seconds
        (truncate or zero-pad as needed).

    Returns
    -------
    ir_final : np.ndarray
        Post-processed impulse response, ready for export.
    """
    # 1. Normalize
    ir_out = normalize(ir, target_dbfs)

    # 2. Trim trailing silence
    ir_out = trim_tail(ir_out, sr, trim_threshold_db)

    # 3. Force output length if specified
    if output_length_s is not None:
        target_len = int(sr * output_length_s)
        if len(ir_out) > target_len:
            ir_out = ir_out[:target_len]
        elif len(ir_out) < target_len:
            if ir_out.ndim == 1:
                ir_out = np.pad(ir_out, (0, target_len - len(ir_out)))
            else:
                ir_out = np.pad(
                    ir_out, ((0, target_len - len(ir_out)), (0, 0))
                )

    # 4. Apply fade-out (after trimming, so fade is at the actual end)
    ir_out = apply_fade_out(ir_out, sr, fade_ms)

    return ir_out
