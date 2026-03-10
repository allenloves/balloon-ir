"""
API Routes

Endpoints for the balloon IR processing web application:
  POST /api/process        — Upload WAV + params, start async processing
  GET  /api/status/{id}    — Poll job status and progress
  GET  /api/result/{id}    — Download ZIP with IR, plots, and params
  GET  /api/preview/{id}   — Get base64-encoded preview plots as JSON
"""

import json
import base64
import zipfile
import io
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from api.tasks import create_job, get_job

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# POST /api/process — Upload and start processing
# ---------------------------------------------------------------------------

@router.post("/process")
async def process(
    file: UploadFile = File(...),
    params: str = Form("{}"),
):
    """
    Upload a balloon pop WAV file and start async processing.

    The file is saved to a temporary location, and a background job
    is created. Returns a job_id for polling status.

    Parameters (JSON string in `params` form field):
        target_sr, onset_threshold_db, ned_window_ms,
        balloon_diameter_cm, num_early_reflections,
        ned_transition_threshold, random_seed, iccc_window_ms,
        energy_window_ms, extrapolate, noise_floor_db,
        gain_smoothing_ms, pulse_halo_ms, target_dbfs, fade_ms,
        trim_threshold_db, output_length_s, output_bit_depth
    """
    # Validate file
    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".wav", ".wave"):
        raise HTTPException(400, f"Expected WAV file, got {suffix}")

    # Parse parameters
    try:
        pipeline_params = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in params field")

    # Save uploaded file to temp location
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    # Create and start background job
    job_id = create_job(tmp.name, pipeline_params)

    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# GET /api/status/{job_id} — Poll status
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}")
async def status(job_id: str):
    """
    Get the current status of a processing job.

    Returns:
        status: "queued" | "processing" | "done" | "error"
        progress: 0-100
        message: human-readable status message
        error: error message (if status == "error")
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    response = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
    }

    if job.status == "error":
        response["error"] = job.error

    if job.status == "done" and job.result is not None:
        result = job.result
        density = result.get("echo_density", {})
        response["summary"] = {
            "sr": result["sr"],
            "is_stereo": result["is_stereo"],
            "ir_duration_s": len(result["ir"]) / result["sr"],
            "balloon_radius_cm": density.get("balloon_radius_m", 0) * 100,
            "nwave_duration_ms": density.get("nwave_duration_s", 0) * 1000,
            "num_early_reflections": len(density.get("early_reflections", [])),
            "transition_time_ms": density.get("transition_time_ms"),
        }

    return response


# ---------------------------------------------------------------------------
# GET /api/result/{job_id} — Download ZIP
# ---------------------------------------------------------------------------

@router.get("/result/{job_id}")
async def result(job_id: str):
    """
    Download the processing result as a ZIP file.

    ZIP contains:
      - ir.wav              — synthesized impulse response
      - plots/*.png         — all diagnostic plots
      - parameters.json     — parameters used for processing
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != "done":
        raise HTTPException(409, f"Job not ready: status={job.status}")

    output_dir = Path(job.output_dir)

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # IR WAV
        ir_path = output_dir / "ir.wav"
        if ir_path.exists():
            zf.write(ir_path, "ir.wav")

        # Plots
        plots_dir = output_dir / "plots"
        if plots_dir.exists():
            for p in sorted(plots_dir.glob("*.png")):
                zf.write(p, f"plots/{p.name}")

        # Parameters
        zf.writestr("parameters.json", json.dumps(job.params, indent=2))

    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="balloon_ir_{job_id}.zip"'
        },
    )


# ---------------------------------------------------------------------------
# GET /api/preview/{job_id} — Base64 preview plots
# ---------------------------------------------------------------------------

@router.get("/preview/{job_id}")
async def preview(job_id: str):
    """
    Get base64-encoded preview plots as JSON.

    Returns a dict mapping plot names to base64 PNG strings,
    suitable for embedding in <img> tags via data URIs.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != "done":
        raise HTTPException(409, f"Job not ready: status={job.status}")

    plots_dir = Path(job.output_dir) / "plots"
    plots = {}

    if plots_dir.exists():
        for p in sorted(plots_dir.glob("*.png")):
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            plots[p.stem] = b64

    return {"job_id": job_id, "plots": plots}
