"""
Async Job Management

Manages background processing jobs for the balloon IR pipeline.
Each upload creates a job with a unique ID, which is processed
in a background thread. Clients poll for status via the API.

Job lifecycle:
  1. Client uploads WAV → job created (status: "queued")
  2. Background thread picks up → (status: "processing", progress 0-100)
  3. Processing completes → (status: "done", results available)
  4. Or fails → (status: "error", error message available)

Jobs and their results are stored in memory (dict). For production
use, this would be replaced with a database + file storage backend.
"""

import uuid
import threading
import traceback
import tempfile
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from core.pipeline import process_balloon
from core.visualization import save_all_plots


@dataclass
class Job:
    """Represents a processing job."""
    id: str
    status: str = "queued"        # queued | processing | done | error
    progress: int = 0
    message: str = ""
    error: str = ""
    params: dict = field(default_factory=dict)
    input_path: str = ""
    output_dir: str = ""
    result: Optional[dict] = None
    created_at: float = 0.0
    completed_at: float = 0.0


# In-memory job store
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(input_path: str, params: dict) -> str:
    """
    Create a new processing job.

    Parameters
    ----------
    input_path : str
        Path to the uploaded WAV file (temporary).
    params : dict
        Pipeline parameters from the client.

    Returns
    -------
    job_id : str
        Unique job identifier.
    """
    job_id = uuid.uuid4().hex[:12]
    output_dir = tempfile.mkdtemp(prefix=f"balloon_{job_id}_")

    job = Job(
        id=job_id,
        input_path=input_path,
        output_dir=output_dir,
        params=params,
        created_at=time.time(),
    )

    with _lock:
        _jobs[job_id] = job

    # Start processing in background thread
    thread = threading.Thread(target=_process_job, args=(job_id,), daemon=True)
    thread.start()

    return job_id


def get_job(job_id: str) -> Optional[Job]:
    """Get a job by ID. Returns None if not found."""
    with _lock:
        return _jobs.get(job_id)


def _process_job(job_id: str):
    """Background worker: run the pipeline for a job."""
    job = get_job(job_id)
    if job is None:
        return

    def progress_callback(pct: int, msg: str):
        with _lock:
            job.status = "processing"
            job.progress = pct
            job.message = msg

    try:
        progress_callback(0, "Starting...")

        params = job.params
        output_ir_path = str(Path(job.output_dir) / "ir.wav")

        # Run the full pipeline
        result = process_balloon(
            job.input_path,
            target_sr=params.get("target_sr"),
            onset_threshold_db=params.get("onset_threshold_db", -40.0),
            ned_window_ms=params.get("ned_window_ms", 43.0),
            balloon_diameter_cm=params.get("balloon_diameter_cm"),
            num_early_reflections=params.get("num_early_reflections", 2),
            ned_transition_threshold=params.get("ned_transition_threshold", 0.3),
            random_seed=params.get("random_seed"),
            iccc_window_ms=params.get("iccc_window_ms", 50.0),
            energy_window_ms=params.get("energy_window_ms", 10.0),
            extrapolate=params.get("extrapolate", True),
            noise_floor_db=params.get("noise_floor_db", -40.0),
            gain_smoothing_ms=params.get("gain_smoothing_ms", 0.0),
            pulse_halo_ms=params.get("pulse_halo_ms", 2.0),
            target_dbfs=params.get("target_dbfs", -1.0),
            fade_ms=params.get("fade_ms", 50.0),
            trim_threshold_db=params.get("trim_threshold_db", -80.0),
            output_length_s=params.get("output_length_s"),
            output_bit_depth=params.get("output_bit_depth", 24),
            output_path=output_ir_path,
            progress_callback=progress_callback,
        )

        # Generate plots
        progress_callback(92, "Generating plots...")
        balloon_mono = result["preprocessing"]["balloon_mono"]
        onset = result["preprocessing"]["onset_sample"]
        sr = result["sr"]

        plots_dir = str(Path(job.output_dir) / "plots")
        save_all_plots(
            result, balloon_mono, sr, plots_dir,
            onset=onset,
            energy_window_ms=params.get("energy_window_ms", 10.0),
        )

        with _lock:
            job.result = result
            job.status = "done"
            job.progress = 100
            job.message = "Done"
            job.completed_at = time.time()

    except Exception as e:
        with _lock:
            job.status = "error"
            job.error = str(e)
            job.message = f"Error: {e}"
            job.completed_at = time.time()
        traceback.print_exc()
