# Balloon Pop → Room Impulse Response Synthesizer

## Project Specification & Claude Code Prompt

Based on: Abel, J.S. et al. (2010). "Estimating Room Impulse Responses from Recorded Balloon Pops." AES Convention Paper 8171, 129th AES Convention, San Francisco.

---

## 1. Project Overview

### Goal
Build a web application that converts recorded balloon pop audio into a clean, full-bandwidth Room Impulse Response (IR) suitable for use in convolution reverberators. The user uploads a balloon pop WAV file and downloads a synthesized IR.

### Why This Exists
Balloon pops are convenient for probing room acoustics (portable, uniform radiation pattern), but their N-wave waveform has spectral nulls that cause audible comb-filtering artifacts when used directly for convolution. This tool performs analysis-resynthesis: it analyzes the balloon recording's temporal, spatial, and spectral properties, then synthesizes a clean IR that preserves the room's acoustic character without the N-wave artifacts.

### Tech Stack
- **Core DSP**: Python 3.11+ (numpy, scipy)
- **Backend API**: FastAPI
- **Frontend**: React (single-page app)
- **Audio I/O**: soundfile (libsndfile), or scipy.io.wavfile
- **Visualization**: matplotlib (server-side plot generation for intermediate results)

---

## 2. Algorithm Pipeline

The processing chain has 5 stages. Each stage should be an independent Python module with clear input/output interfaces.

### Stage 0: Preprocessing (`preprocessing.py`)

**Input**: Raw balloon pop WAV file (mono or stereo, any sample rate)

**Operations**:
1. Read WAV, normalize to [-1, 1]
2. If stereo, keep both channels for spatial analysis; also create a mono mixdown for density/energy analysis
3. Detect the onset of the balloon pop (threshold-based or energy-based onset detection)
4. Trim pre-onset silence (keep ~10ms before onset for safety)
5. Resample to a target sample rate if needed (e.g., 48kHz)

**Output**: 
- `balloon_mono: np.ndarray` — mono balloon recording, trimmed
- `balloon_stereo: Optional[tuple[np.ndarray, np.ndarray]]` — L/R channels if stereo
- `sr: int` — sample rate
- `onset_sample: int` — detected onset position

**Notes**:
- Onset detection should be robust to low-level background noise
- Consider using a simple energy threshold: find the first sample where a short-window RMS exceeds, say, -40dB of the peak

---

### Stage 1: Echo Density Analysis & Synthesis (`echo_density.py`)

**Purpose**: Analyze the temporal structure of reflections and generate a synthetic pulse sequence with matching perceived echo density.

#### 1a. Integration
The balloon pop echoes are N-wave doublets, not simple pulses. Integrate the recording to convert N-waves into parabolic pulses:

```
b_integrated(t) = cumulative_trapezoid(balloon_mono)
```

This makes the echo density estimation more reliable (see paper Fig. 5).

#### 1b. Normalized Echo Density (NED) Estimation

Compute NED η(t) in a sliding window — implements Equation (3):

```
For each center time t with half-window Δ:
    window = b_integrated[t-Δ : t+Δ]
    N = 2Δ + 1                                # window length in samples
    σ² = mean(window²)                        # Eq. (4): window variance (zero-mean assumed)
    count = number of samples where window² > σ²  # indicator function 1{h²(τ) > σ²(t)}
    fraction = count / N                       # proportion of samples exceeding σ
    η(t) = fraction / erfc(1/√2)              # Eq. (3): normalize so Gaussian noise → η ≈ 1
```

- `erfc(1/√2) ≈ 0.3173` is the expected fraction of samples outside ±1σ for Gaussian noise
- Dividing by this reference value (not multiplying) calibrates the scale: Gaussian → 1.0
- Window length: use ~2048 samples at 48kHz (~43ms) as a starting point; make this a parameter
- NED ranges from ~0 (sparse, specular reflections) to ~1 (fully diffuse, Gaussian-like)
- NED can slightly exceed 1.0 if amplitude distribution is more uniform than Gaussian

#### 1c. NED Conversion (Balloon → Full-Bandwidth)

The balloon's N-wave echoes are longer than ideal full-bandwidth pulses (duration ratio ≈ 10–20x). Convert the balloon NED to equivalent full-bandwidth NED:

```
η_h = η_b / ((1 - η_b) * (2ρ / (δ*c)) + η_b)
```

Where:
- `η_b` = balloon NED
- `2ρ/c` = N-wave duration (estimate from balloon diameter, or from the direct-path waveform width)
- `δ` = full-bandwidth pulse duration ≈ 1/sr (one sample)
- `2ρ/(δ*c)` is the duration ratio, typically 10–20

**Estimating balloon diameter**: Measure the time between zero-crossings of the direct-path N-wave arrival → duration = 2ρ/c → ρ = duration * c / 2.

#### 1d. NED Clamping

Once η_h(t) first reaches 0.995, hold it fixed at that value for all subsequent time points (the paper does this explicitly — see Fig. 6). This prevents numerical instability and unnecessary density fluctuations in the fully diffuse late-field.

Also clamp η_b to [0, 0.999] before the conversion in Step 1c to avoid division by zero.

#### 1e. Absolute Echo Density (AED)

Convert the **full-bandwidth** NED η_h(t) (not the balloon NED η_b) to AED (echoes per second). Rearrange formula (5) using the full-bandwidth pulse duration δ ≈ 1/sr:

```
e(t) = η_h(t) * sr / (1 - η_h(t))
```

This gives the number of full-bandwidth echoes per second. Note: the spec previously had this formula using η_b and c/2ρ, which corresponds to formula (6) in the paper — that gives the same e(t) mathematically, but conceptually it is clearer to work from η_h with the full-bandwidth δ, since e(t) is the density of full-bandwidth pulses we are about to synthesize.

#### 1f. Echo Sequence Synthesis

Generate a sequence of Dirac-like pulses with Poisson-distributed inter-arrival times following e(t):

```python
def synthesize_echo_sequence(aed_profile, sr, duration):
    """
    Generate pulse sequence with time-varying Poisson density.
    
    For each time step:
        expected_interval = sr / e(t)  # samples between pulses
        actual_interval = exponential_random(expected_interval)
        place a pulse at current_time + actual_interval
        pulse amplitude drawn from Gaussian distribution
    
    Scale amplitudes locally so energy is roughly constant.
    """
```

**Manual placement of early reflections**:
- The first few clear arrivals (direct path, floor reflection) should be placed manually rather than generated randomly
- Detect these from the integrated balloon response (find the first 2–4 peaks above a threshold)
- Place corresponding pulses at those times in the synthetic sequence
- Use the NED-based synthesis only for the remaining (later) portion

**Output**: `echo_sequence: np.ndarray` — synthetic pulse train, same length and sr as input. For stereo processing (Stage 2), generate TWO independent sequences using different random seeds.

---

### Stage 2: Spatial Character Analysis & Synthesis (`spatial.py`)

**Purpose**: For stereo recordings, preserve the room's spatial character (envelopment vs. directional energy) in the synthesized IR.

**Skip this stage entirely for mono recordings.**

**Input for Stage 2b**: Two statistically independent echo sequences from Stage 1f (generated with different random seeds but the same AED profile). These start with near-zero inter-channel correlation.

#### 2a. Inter-Channel Cross-Correlation (ICCC) Estimation

Compute running zero-lag cross-correlation between L and R channels:

```python
def estimate_iccc(left, right, window_length=2400):
    """
    window_length: ~50ms at 48kHz (paper uses 50ms running window)
    Use zero-lag only (l=0) as the paper recommends for simplicity.
    The full method searches lags in [-1, 1]ms and takes the max,
    but the perceptual difference is minimal.
    
    For each center time t:
        L_win = left[t-Δ:t+Δ]
        R_win = right[t-Δ:t+Δ]
        C(t) = sum(L_win * R_win) / (norm(L_win) * norm(R_win))
    
    Returns: C(t) array, values in [-1, 1]
    """
```

The paper also mentions evaluating at lags in [-1, 1]ms and taking the maximum, but suggests that zero-lag is sufficient for practical purposes.

#### 2b. Stereo Echo Sequence Synthesis

Start with TWO statistically independent echo sequences (both generated from Stage 1 with different random seeds). These have near-zero correlation.

Apply a rotation matrix to impose the measured cross-correlation:

```python
def impose_correlation(seq_left, seq_right, correlation_profile):
    """
    For each sample pair (L, R) at time t:
        θ = arcsin(C(t)) / 2
        
        new_L = L * cos(θ) + R * sin(θ)
        new_R = L * sin(θ) + R * cos(θ)
    
    When C=0: θ=0, channels unchanged (independent)
    When C=1: θ=π/4, channels become identical (mono)
    """
```

**Output**: `(echo_seq_left, echo_seq_right)` — stereo pulse sequences with prescribed correlation profile

---

### Stage 3: Time-Frequency Energy Analysis & Synthesis (`energy_shaping.py`)

**Purpose**: Shape the spectral content of the synthetic echo sequence to match the room's frequency-dependent decay characteristics.

#### 3a. Filter Bank

**IMPORTANT**: The filter bank operates on the **original** balloon recording b(t) and the synthesized echo sequence p(t) — NOT on the integrated signal from Stage 1. The integration in Stage 1a was only used for NED estimation.

Implement a perfect-amplitude-reconstruction zero-phase filter bank with 1/3-octave bands:

```python
def create_filterbank(sr, num_bands=30):
    """
    Band-splitting via cascaded squared Butterworth filters.
    
    Implementation approach:
    - Use scipy.signal.butter to design 3rd-order Butterworth bandpass filters
    - Apply forward-backward (filtfilt) for zero-phase, doubling the effective order
    - This gives 60dB/octave transitions
    
    Center frequencies: 1/3-octave spacing from ~50Hz to sr/2
    f_center[k] = f_ref * 2^(k/3), where f_ref is chosen to include
    standard 1/3-octave centers (100, 125, 160, 200, 250, 315, ... Hz)
    
    Returns: list of (b, a) filter coefficients for each band
    """
```

**Implementation note**: The paper describes a cascade of band-splitting filters (Fig. 11). A practical alternative is to use a parallel bank of bandpass filters, as long as the sum of all band outputs reconstructs the original signal. Verify reconstruction by summing all filtered bands and comparing to input.

#### 3b. Band Energy Estimation

For each band k, compute smoothed energy profiles for both the balloon recording and the echo sequence:

```python
def estimate_band_energy(signal_bands, window_ms=10, sr=48000):
    """
    For each band k:
        β²_k(t) = convolve(b_k(t)², hanning_window)
        
    Window: 10ms Hanning for fine temporal resolution
    
    Returns: list of energy profiles, one per band
    """
```

#### 3c. Energy Extrapolation (Below Noise Floor)

The balloon recording's band energies hit a frequency-dependent noise floor (typically around -40dB). Extrapolate the decay beyond the noise floor for a more natural result:

```python
def extrapolate_energy(band_energy_db, noise_floor_db=-40):
    """
    For each band:
    1. Estimate the noise floor (e.g., median of last 500ms)
    2. Find the time where energy first drops below noise_floor + 3dB
    3. Fit a linear regression (in dB) to the energy curve 
       in a region above the noise floor (e.g., from -10dB to noise_floor)
    4. Extrapolate the linear decay beyond the noise floor
    
    This is a simplified version of Bryan & Abel [13].
    A more sophisticated approach would use their actual method.
    """
```

#### 3d. Band Energy Imprinting

Apply the balloon's energy envelope to the synthetic echo sequence — Equation (14):

```python
def imprint_energy(echo_bands, balloon_energy_sq, echo_energy_sq):
    """
    For each band k — Equation (14):
        γ_k(t) = β_k(t) / ν_k(t)
    
    where β_k and ν_k are AMPLITUDES (square root of the energy profiles):
        β_k(t) = sqrt(β²_k(t))   — from Eq. (12)
        ν_k(t) = sqrt(ν²_k(t))   — from Eq. (13)
    
    So in terms of the energy profiles computed in Step 3b:
        γ_k(t) = sqrt(balloon_energy_sq_k(t)) / sqrt(echo_energy_sq_k(t))
               = sqrt(balloon_energy_sq_k(t) / echo_energy_sq_k(t))
    
    Then apply: shaped_k(t) = p_k(t) * γ_k(t)
    
    Handle division by zero: where echo_energy_sq is very small, 
    set γ to 0 (or a small value). Also consider smoothing γ_k(t) 
    to avoid rapid gain fluctuations (see Known Gaps #6).
    """
```

#### 3e. Direct Path Equalization & Final Summation

Equalize the shaped bands so the direct path arrival is spectrally flat — Equation (16):

```python
def equalize_and_sum(shaped_bands, balloon_bands, onset_sample, sr):
    """
    Equation (16): h̃(t) = Σ_k p_k(t) · γ_k(t) · α_k
    
    α_k is the inverse of the direct path arrival's band gain, estimated 
    from the BALLOON recording's band-filtered signal (not the shaped bands).
    
    For each band k:
        1. Take b_k(t) in a short window around onset (e.g., first 5ms)
        2. Measure the peak amplitude or RMS energy of that window
        3. α_k = 1.0 / that measurement
    
    This "whitens" the direct path: if the balloon's direct arrival 
    was louder at 1kHz than 10kHz (due to N-wave spectral shape), 
    α_k compensates so the final IR's direct path is flat.
    
    Final IR = sum over all k of: shaped_band_k(t) * α_k
    
    For stereo: apply independently to both channels (same α_k values).
    """
```

**Output**: `ir: np.ndarray` (or stereo tuple) — the final synthesized impulse response

---

### Stage 4: Post-Processing & Export (`postprocessing.py`)

1. **Normalize** the final IR to peak = 0.95 (or -1dBFS)
2. **Fade out**: Apply a short cosine fade at the end to avoid clicks
3. **Trim trailing silence**: Remove tail below -80dB
4. **Export as WAV**: 24-bit or 32-bit float, at the original or specified sample rate
5. **Generate metadata**: RT60 estimate per band, EDT, C80, D50 (standard room acoustic parameters — optional but useful for validation)

---

## 3. Module Dependency Graph

```
Input WAV
    │
    ▼
[Stage 0: Preprocessing]
    │
    ├──── mono signal ────▶ [Stage 1: Echo Density] ──▶ echo_sequence(s)
    │                              │
    │                              │ (NED profile for visualization)
    │                              ▼
    ├──── stereo signals ──▶ [Stage 2: Spatial] ──▶ stereo echo_sequences
    │                              │
    │                              │ (ICCC profile for visualization)
    │                              ▼
    ├──── mono/stereo ────▶ [Stage 3: Energy Shaping] ──▶ shaped IR
    │                              │
    │                              │ (spectrograms, band energies for visualization)
    │                              ▼
    └─────────────────────▶ [Stage 4: Post-Processing] ──▶ final IR WAV
```

---

## 4. Configurable Parameters

Expose these to the user via the web UI:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ned_window_ms` | 43 | 10–100 | NED estimation window length (ms) |
| `balloon_diameter_cm` | auto | 5–40 | Balloon diameter; auto-detect from direct path |
| `energy_window_ms` | 10 | 5–50 | Band energy smoothing window (ms) |
| `extrapolate` | true | bool | Whether to extrapolate energy below noise floor |
| `noise_floor_db` | -40 | -60 to -20 | Noise floor threshold for extrapolation |
| `num_early_reflections` | 2 | 0–10 | Number of early reflections to place manually |
| `output_sr` | (same as input) | 44100/48000/96000 | Output sample rate |
| `output_bit_depth` | 24 | 16/24/32float | Output bit depth |
| `output_length_s` | auto | 0.5–30 | Output IR length in seconds |

---

## 5. Backend API (FastAPI)

### File: `api.py`

```
POST /api/process
    - Multipart form upload: WAV file + JSON parameters
    - Returns: job_id
    - Starts async processing

GET /api/status/{job_id}
    - Returns: { status: "processing"|"done"|"error", progress: 0-100, stage: "..." }

GET /api/result/{job_id}
    - Returns: ZIP containing:
        - synthesized_ir.wav (final IR)
        - analysis_plots.png (composite visualization)
        - parameters.json (all parameters used)
    
GET /api/preview/{job_id}
    - Returns: JSON with base64-encoded preview plots:
        - NED profile (balloon vs. estimated full-bandwidth)
        - Cross-correlation profile (if stereo)
        - Spectrogram comparison (balloon vs. synthesized)
        - Band energy decay curves

POST /api/audition/{job_id}
    - Body: { "audio_url": "..." } or uploaded dry audio file
    - Returns: convolved audio preview (first 10 seconds)
    - Purpose: let user hear the IR applied to dry audio
```

### Processing Pipeline (async)

```python
async def process_balloon(wav_bytes, params):
    # Stage 0
    balloon_mono, balloon_stereo, sr, onset = preprocess(wav_bytes, params)
    yield progress(10, "Preprocessing complete")
    
    # Stage 1
    ned_profile, echo_seq = analyze_and_synthesize_density(
        balloon_mono, sr, onset, params
    )
    yield progress(30, "Echo density analysis complete")
    
    # Stage 2 (stereo only)
    if balloon_stereo:
        iccc_profile = estimate_iccc(balloon_stereo, sr)
        echo_seq_L, echo_seq_R = impose_stereo(echo_seq, iccc_profile, sr, params)
        yield progress(50, "Spatial analysis complete")
    
    # Stage 3
    ir = shape_energy(
        balloon_mono, echo_seq, sr, onset, params,
        stereo=(echo_seq_L, echo_seq_R) if balloon_stereo else None
    )
    yield progress(80, "Energy shaping complete")
    
    # Stage 4
    final_ir = postprocess(ir, sr, params)
    yield progress(100, "Done")
    
    return final_ir
```

---

## 6. Frontend (React)

### Pages / Views

#### Main View: Upload & Process
- Drag-and-drop WAV upload zone
- Waveform preview of uploaded file (use Web Audio API or a lightweight waveform library)
- Parameter panel (collapsible "Advanced Settings")
- "Process" button → shows progress bar with stage labels

#### Results View
- **Tab 1: Comparison**
    - Side-by-side spectrograms: original balloon pop vs. synthesized IR
    - Overlaid NED profiles
    - RT60 bar chart per frequency band
    
- **Tab 2: Analysis Details**
    - NED profile plot
    - Cross-correlation profile (if stereo)
    - Band energy decay curves
    - Detected balloon diameter and early reflection times

- **Tab 3: Audition**
    - Built-in dry audio samples (speech, clap, music snippet)
    - Or upload custom dry audio
    - Play button: convolves with synthesized IR in real-time (Web Audio API ConvolverNode)
    - A/B comparison: original balloon convolution vs. synthesized IR convolution

- **Download button**: ZIP with IR + plots + params

### Key UI Components
- `WaveformDisplay` — canvas-based waveform viewer
- `SpectrogramView` — renders spectrogram images from API
- `ParameterPanel` — sliders/inputs for all configurable params
- `AudioPlayer` — Web Audio API player with convolution
- `ProgressTracker` — progress bar with stage labels

---

## 7. File Structure

```
balloon-to-ir/
├── README.md
├── requirements.txt          # numpy, scipy, soundfile, matplotlib, fastapi, uvicorn, python-multipart
├── pyproject.toml
│
├── core/                     # DSP modules (no web dependencies)
│   ├── __init__.py
│   ├── preprocessing.py      # Stage 0
│   ├── echo_density.py       # Stage 1
│   ├── spatial.py            # Stage 2
│   ├── energy_shaping.py     # Stage 3
│   ├── postprocessing.py     # Stage 4
│   ├── filterbank.py         # 1/3-octave filter bank (used by Stage 3)
│   ├── pipeline.py           # Orchestrates all stages
│   └── visualization.py      # Plot generation (matplotlib)
│
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── main.py               # App entry point, CORS config
│   ├── routes.py             # API endpoints
│   └── tasks.py              # Async processing, job management
│
├── frontend/                 # React app
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── ParameterPanel.jsx
│   │   │   ├── ProgressTracker.jsx
│   │   │   ├── ResultsView.jsx
│   │   │   ├── SpectrogramView.jsx
│   │   │   ├── AudioPlayer.jsx
│   │   │   └── WaveformDisplay.jsx
│   │   └── utils/
│   │       └── audioUtils.js  # Web Audio API helpers
│   └── public/
│       └── dry_samples/       # Built-in dry audio for auditioning
│
├── tests/                    # Unit tests
│   ├── test_preprocessing.py
│   ├── test_echo_density.py
│   ├── test_spatial.py
│   ├── test_energy_shaping.py
│   ├── test_filterbank.py
│   └── test_pipeline.py
│
├── examples/                 # Example balloon pop recordings for testing
│   └── README.md             # Where to find test recordings (e.g., OpenAIR database)
│
└── scripts/
    ├── cli.py                # Command-line interface (process a WAV without the web app)
    └── validate_ir.py        # Compare synthesized IR against reference measurements
```

---

## 8. Development Order (Suggested for Claude Code)

### Phase 1: Core DSP (get the algorithm working)
1. `filterbank.py` — implement and verify perfect reconstruction
2. `preprocessing.py` — onset detection, normalization
3. `echo_density.py` — NED estimation, AED conversion, Poisson synthesis
4. `energy_shaping.py` — band analysis, energy imprinting, equalization
5. `spatial.py` — ICCC estimation, stereo synthesis
6. `postprocessing.py` — normalization, fade, trim, export
7. `pipeline.py` — wire everything together
8. `cli.py` — command-line tool for quick testing

**Test at each step** with a synthetic test signal first (e.g., a known IR convolved with a synthetic N-wave), then with a real balloon pop recording.

### Phase 2: Visualization & Validation
9. `visualization.py` — NED plots, spectrograms, band energies, comparison views
10. `validate_ir.py` — compare RT60, EDT, C80 between synthesized IR and reference

### Phase 3: Web Backend
11. `api/main.py` + `api/routes.py` — basic upload/download endpoints
12. `api/tasks.py` — async job processing with progress tracking

### Phase 4: Web Frontend
13. Upload + parameter UI
14. Progress display
15. Results visualization
16. Audio auditioning (ConvolverNode)

---

## 9. Coding Style & Documentation Requirements

This codebase serves a dual purpose: functional tool AND educational reference for a university recording engineering course. **Every DSP function must be thoroughly documented** so that a student reading the code can understand the acoustic/mathematical reasoning behind each step.

### Comment Requirements

#### Module-level docstrings
Each `.py` file must begin with a docstring explaining:
- What stage of the pipeline this module implements
- Which section(s) of Abel et al. (2010) it corresponds to
- The acoustic/physical motivation (why this step exists)

Example:
```python
"""
Echo Density Analysis & Synthesis (Stage 1)

Implements §3 of Abel et al. (2010) "Estimating Room Impulse Responses 
from Recorded Balloon Pops," AES Convention Paper 8171.

This module analyzes the temporal structure of reflections in a balloon 
pop recording and generates a synthetic pulse sequence with matching 
perceptual echo density. The key insight (from psychoacoustic research 
by Huang et al., 2008) is that human perception of echo density depends 
on the overall density of arrivals, not their precise timing — allowing 
us to synthesize a statistically equivalent sequence rather than 
reproducing each reflection exactly.
"""
```

#### Function-level docstrings
Every function must include:
1. Plain-language explanation of what it does and why
2. The paper's equation number(s) being implemented
3. Parameter descriptions with units and typical values
4. Return value description

Example:
```python
def compute_ned(signal, half_window, sr):
    """
    Compute Normalized Echo Density (NED) over a sliding window.
    
    Implements Equation (3) from Abel et al. (2010), §3.1.
    
    NED measures how "diffuse" the signal is at each point in time,
    on a scale from 0 (sparse specular reflections) to ~1 (fully 
    diffuse, statistically Gaussian). It works by counting what 
    fraction of samples in a window exceed the window's standard 
    deviation, then normalizing by the fraction expected for 
    Gaussian noise (≈ 0.3173, from the complementary error function).
    
    Parameters
    ----------
    signal : np.ndarray
        The INTEGRATED balloon pop recording (not the raw recording).
        Integration converts N-wave doublets into single parabolic 
        pulses, preventing each echo from being double-counted.
        See §3.2, Equation (7).
    half_window : int
        Half-window size Δ in samples. The full window is 2Δ+1.
        Paper uses ~43ms; at 48kHz this is Δ ≈ 1024.
    sr : int
        Sample rate in Hz.
    
    Returns
    -------
    ned : np.ndarray
        NED profile η(t), same length as input signal.
        Values typically range from 0 to ~1, but can slightly 
        exceed 1 if the amplitude distribution is more uniform 
        than Gaussian (see discussion in §3.1).
    """
```

#### Inline comments for formula steps
When implementing a formula, break it into labeled steps with the equation number and a plain-language gloss:

```python
# --- Equation (3): Normalized Echo Density ---
# For each window position, count how many samples have
# instantaneous energy (h²) exceeding the window variance (σ²).
# This ratio, normalized by the Gaussian expectation (≈ 0.3173),
# gives NED: 0 = sparse reflections, 1 = fully diffuse.

# Step 1: Window variance σ²(t) — Equation (4)
# This is the mean squared value in the window, equivalent to 
# RMS² since we assume zero-mean signal.
sigma_sq = np.mean(window ** 2)

# Step 2: Count samples exceeding σ² — the indicator function 1{h²(τ) > σ²(t)}
# Each sample is compared against the window's average energy.
# Sparse reflections: only a few large peaks exceed σ² → low count
# Gaussian noise: ~31.73% of samples exceed σ² → count ≈ 0.3173 * N
exceeding_count = np.sum(window ** 2 > sigma_sq)

# Step 3: Normalize — divide by window length and Gaussian reference
# erfc(1/√2) ≈ 0.3173 is the theoretical fraction of samples 
# outside ±1σ for a Gaussian distribution. Dividing by this 
# calibrates the scale so that Gaussian noise → NED ≈ 1.0.
GAUSSIAN_REFERENCE = erfc(1 / np.sqrt(2))  # ≈ 0.3173
ned_value = exceeding_count / (window_length * GAUSSIAN_REFERENCE)
```

#### Key decision points
When the code makes an engineering decision not fully specified in the paper, mark it clearly:

```python
# NOTE: Paper §3.2 (below Eq. 8) states that η_h is held fixed 
# at 0.995 after first reaching that value, to prevent numerical
# instability in the fully diffuse late-field. We implement this
# as a forward pass: once the threshold is crossed, all subsequent
# values are clamped.
if ned_h[i] >= 0.995:
    ned_h[i:] = 0.995
    break
```

```python
# ENGINEERING DECISION: The paper (§5.2) cites Bryan & Abel [13] 
# for energy extrapolation below the noise floor, but does not 
# detail the algorithm. We use a simplified approach: fit a linear 
# regression (in dB) to the energy curve in a region 10–20 dB 
# above the estimated noise floor, then extend that line.
```

### Language

- All code, comments, variable names, and docstrings in **English**
- Use descriptive variable names that map to the paper's notation:
  - `eta_b` for η_b (balloon NED)
  - `eta_h` for η_h (full-bandwidth NED) 
  - `aed` or `echo_density` for e(t)
  - `beta_sq` for β²_k(t)
  - `gamma` for γ_k(t)
  - `alpha` for α_k
- When a variable corresponds to a paper symbol, note it:
  ```python
  half_window = 1024  # Δ in Equation (3), ~43ms at 48kHz
  ```

### Type Hints

Use Python type hints on all function signatures:
```python
def compute_ned(signal: np.ndarray, half_window: int, sr: int) -> np.ndarray:
```

---

## 10. Per-Stage Validation & Testing

Each stage needs concrete, verifiable tests. The strategy is to use **synthetic test signals with known properties** so you can check the output against an expected answer.

### Generating the Master Test Signal

Before testing any stage, create a synthetic test environment:

```python
def create_test_data(sr=48000):
    """
    Generate a synthetic balloon pop recording with KNOWN properties.
    
    1. Create a known IR: 
       - Direct path delta at t=0
       - Floor reflection at t=15ms, amplitude 0.7
       - A few more early reflections at known times
       - Exponential noise tail (Gaussian noise * exponential decay)
       - Different decay rates per frequency band (faster at high freq)
       - Total length: ~2 seconds
    
    2. Create a synthetic N-wave:
       - Duration 2ρ/c for a chosen ρ (e.g., ρ=15cm → duration ≈ 0.87ms)
       - Linear ramp from +peak to -peak
    
    3. Convolve the known IR with the N-wave → synthetic balloon recording
    
    Now you have:
    - known_ir: the ground truth
    - synthetic_balloon: what the pipeline receives as input
    - known_rho: the balloon radius used
    
    If the pipeline works correctly, the output should closely 
    resemble known_ir (not identical, because the synthesis is 
    stochastic, but statistically equivalent).
    """
```

This master test signal is used across all stages.

---

### Stage 0: Preprocessing — Validation

**Test 1: Onset detection accuracy**
- Create a test signal with silence followed by a known impulse at sample N
- Run onset detection
- **Pass criterion**: detected onset is within ±5 samples of N

**Test 2: Normalization**
- Input a signal with peak at 0.5
- **Pass criterion**: output peak is exactly 1.0 (or your target)

**Test 3: Stereo mixdown**
- Input L=[1,0,0,...], R=[0,1,0,...]
- **Pass criterion**: mono mixdown = [0.5, 0.5, 0, ...] (or however you define mixdown)

---

### Stage 1: Echo Density — Validation

**Test 1a: Integration converts N-wave to single peak**
- Create a single synthetic N-wave (linear ramp +1 to -1)
- Integrate it
- **Pass criterion**: result is a single positive bump (parabolic shape), no negative values, area > 0
- Visual check: plot should look like Fig. 4 in the paper

**Test 1b: NED of known signals**
- **Pure Gaussian noise**: compute NED over a long window
  - **Pass criterion**: NED ≈ 1.0 (within ±0.05)
- **Single impulse in silence**: one large spike, rest is zeros
  - **Pass criterion**: NED ≈ 0 at the impulse location
- **Gradual transition**: concatenate sparse impulses → dense noise
  - **Pass criterion**: NED curve rises monotonically from ~0 to ~1
  - Visual check: should resemble Fig. 6 in the paper

**Test 1c: NED conversion (balloon → full-bandwidth)**
- Set η_b = 0 → **Pass**: η_h = 0
- Set η_b = 1 → **Pass**: η_h = 1
- Set η_b = 0.5, duration_ratio = 10 → compute η_h
  - **Pass criterion**: η_h < η_b (should be noticeably smaller)
  - Cross-check: plug values into Equation (8) by hand

**Test 1d: NED clamping**
- Feed a signal where η_h exceeds 0.995 at some point
- **Pass criterion**: all η_h values after that point are exactly 0.995

**Test 1e: AED sanity check**
- From known η_h = 0.5 at sr = 48000
- e(t) = 0.5 * 48000 / (1 - 0.5) = 48000 echoes/sec
- **Pass criterion**: computed AED matches hand calculation

**Test 1f: Echo sequence statistics**
- Generate a sequence with constant AED = 1000 echoes/sec, duration = 1 sec
- Count the actual number of pulses generated
- **Pass criterion**: count is approximately 1000 (within ±10%, due to Poisson randomness)
- Also: compute NED of the generated sequence
  - **Pass criterion**: NED profile is roughly constant and matches the target η_h

---

### Stage 2: Spatial Character — Validation

**Test 2a: Cross-correlation of known signals**
- L = R (identical signals) → **Pass**: C(t) = 1.0
- L and R are independent Gaussian noise → **Pass**: C(t) ≈ 0 (within ±0.1)
- L = -R (inverted) → **Pass**: C(t) = -1.0

**Test 2b: Correlation imposition**
- Start with two independent sequences (C ≈ 0)
- Impose C = 0.5 using matrix M
- Measure the resulting cross-correlation
- **Pass criterion**: measured C ≈ 0.5 (within ±0.05)
- Also test extremes: impose C = 0 → sequences stay independent; impose C = 1 → sequences become identical

**Test 2c: Time-varying correlation**
- Impose a ramp from C = 1.0 to C = 0.0 over 1 second
- Measure cross-correlation in sliding windows
- **Pass criterion**: measured profile roughly follows the imposed ramp
- Visual check: should qualitatively resemble Fig. 9 or Fig. 10

---

### Stage 3: Time-Frequency Energy — Validation

**Test 3a: Filter bank perfect reconstruction**
- Filter any signal (e.g., white noise) into all bands
- Sum all bands back together
- **Pass criterion**: sum equals original signal within numerical precision
  - Compute: max(abs(original - reconstructed)) < 1e-10
  - This is the single most important test in the entire pipeline. If this fails, everything downstream is wrong.

**Test 3b: Band energy of known signal**
- Create a sine wave at 1kHz
- Filter into bands, compute band energies
- **Pass criterion**: only the band containing 1kHz has significant energy; all others are near zero
- Also test with white noise: all bands should have roughly equal energy

**Test 3c: Energy imprinting**
- Create an echo sequence with flat energy (all bands equal)
- Create a "balloon" energy profile that decays faster at high frequencies
- Apply γ_k(t) imprinting
- **Pass criterion**: the shaped sequence's band energies now match the balloon's band energies
- Visual check: plot spectrograms of both — they should look similar

**Test 3d: Direct path equalization**
- Create a test signal where the direct path has known per-band gains (e.g., 1kHz band is 6dB louder than 4kHz band)
- Apply α_k equalization
- **Pass criterion**: after equalization, the direct path has flat spectrum (all bands within ±1dB)

**Test 3e: Energy extrapolation**
- Create a band energy curve that decays linearly (in dB) and hits a noise floor at -40dB
- Run extrapolation
- **Pass criterion**: below the noise floor, the curve continues the linear decay slope
- Visual check: compare to Fig. 13 (upper vs. lower panels)

---

### Stage 4: Post-Processing — Validation

**Test 4a: Normalization**
- **Pass criterion**: output peak is at specified level (e.g., -1dBFS)

**Test 4b: Fade-out**
- **Pass criterion**: last N samples follow a smooth cosine curve to zero; no abrupt discontinuity at the end

**Test 4c: Tail trimming**
- Create an IR with a long tail that drops below -80dB at t=1.5s
- **Pass criterion**: output is trimmed to approximately 1.5s

---

### Full Pipeline Validation (End-to-End)

**Test E2E-1: Synthetic round-trip**
- Use the master test signal (known_ir convolved with synthetic N-wave)
- Run through the full pipeline
- Compare output IR to known_ir:
  - **RT60 comparison**: per-band RT60 of output should be within ±10% of known_ir
  - **Spectrogram similarity**: visual comparison of time-frequency energy distribution
  - **NED profile**: output's NED profile should match known_ir's NED profile
  - Note: exact waveform match is NOT expected (synthesis is stochastic)

**Test E2E-2: Real balloon pop (if available)**
- If you have a balloon pop recording from a space where a swept-sine IR also exists (e.g., from OpenAIR database):
  - Run the balloon pop through the pipeline
  - Compare output RT60, EDT, C80 to the swept-sine reference
  - **Pass criterion**: RT60 within ±15%, C80 within ±2dB
  - Listening test: convolve both IRs with dry speech or music; they should sound like the same room

**Test E2E-3: Perceptual sanity check**
- Convolve the output IR with dry audio (speech, clap, music)
- Listen for:
  - ❌ Comb-filtering artifacts (metallic/hollow quality) → energy shaping or equalization problem
  - ❌ Unnatural echo density transitions (sudden change in texture) → NED estimation problem
  - ❌ Spatial image collapse or asymmetry (stereo) → correlation imposition problem
  - ❌ Abrupt cutoff at the tail → extrapolation or trimming problem
  - ✅ Natural reverb that sounds like a real room

---

### Test Execution Order

Follow this order — each stage depends on the previous one being correct:

```
1. Stage 3a FIRST — filter bank reconstruction (foundation of everything)
2. Stage 0 — preprocessing (need correct input for all other tests)
3. Stage 1b — NED with known signals
4. Stage 1c, 1d, 1e — NED conversion, clamping, AED (hand-calculable)
5. Stage 1a — integration (visual check against Fig. 4)
6. Stage 1f — echo sequence statistics
7. Stage 2a, 2b — cross-correlation (if doing stereo)
8. Stage 3b, 3c, 3d, 3e — band energy analysis/synthesis
9. Stage 4 — post-processing
10. Full pipeline E2E tests
```

Note: filter bank (3a) is tested FIRST even though it's Stage 3, because if reconstruction fails, the energy shaping results will be meaningless and you'll waste time debugging the wrong thing.

---

## 11. Known Gaps & Engineering Decisions Needed

These are aspects the paper doesn't fully specify. Make reasonable engineering choices:

1. **Energy extrapolation method**: The paper cites Bryan & Abel [13] but doesn't detail the algorithm. A simple approach: fit a linear regression to the dB energy curve in a region 10–20dB above the noise floor, then extend that line.

2. **Early reflection detection**: The paper says "the first few clear arrivals may be placed by hand." Automate this: detect peaks in the integrated balloon response that exceed a threshold (e.g., 6dB above the local NED-predicted level).

3. **Balloon diameter auto-detection**: Measure the zero-crossing interval of the direct-path N-wave. This requires clean isolation of the direct path, which may need a short analysis window.

4. **Filter bank edge bands**: The lowest and highest bands need special handling (lowpass for the bottom, highpass for the top).

5. **NED clipping**: When η_b approaches 1.0, the AED goes to infinity. Clip η_b at 0.999 and hold η_h fixed once it reaches ~0.995 (as the paper does in Fig. 6).

6. **Smoothing the gain function γ_k(t)**: The ratio β_k/ν_k can be noisy. Apply smoothing to γ_k(t) to avoid rapid gain fluctuations.

7. **Window lengths vary by stage** — use these defaults from the paper:

| Stage | Window Purpose | Length | Reference |
|-------|---------------|--------|-----------|
| 1b | NED estimation | ~43ms (2048 samples @ 48kHz) | §3.1 |
| 2a | Cross-correlation (ICCC) | 50ms | §4, Fig. 9 |
| 3b | Band energy smoothing | 10ms (Hanning) | §5.2 |

---

## 12. Test Data Sources

- **OpenAIR** (openairlib.net): Free impulse response library; some entries include balloon pop recordings alongside swept-sine measurements — perfect for validation
- **Record your own**: Pop a balloon in a reverberant space while recording with a handheld recorder; this is the paper's intended use case
- **Synthetic test**: Generate a known IR, convolve with a synthetic N-wave, run through the pipeline, and compare output to the known IR

---

## 13. References

- Abel, J.S. et al. (2010). "Estimating Room Impulse Responses from Recorded Balloon Pops." AES Convention Paper 8171.
- Abel, J.S. & Huang, P. (2006). "A Simple, Robust Measure of Reverberation Echo Density." AES Convention Paper 6985.
- Huang, P. & Abel, J.S. (2007). "Aspects of Reverberation Echo Density." AES Convention Paper 7163.
- Huang, P. et al. (2008). "Reverberation Echo Density Psychoacoustics." AES Convention Paper 7583.
- Bryan, N.J. & Abel, J.S. (2010). "Methods For Extending Room Impulse Responses Beyond Their Noise Floor." AES Convention Paper, 129th Convention.
- Farina, A. (2000). "Simultaneous Measurement of Impulse Response and Distortion with a Swept-Sine Technique." AES Convention Paper 5093.
