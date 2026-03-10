---
title: Balloon IR Synthesizer
emoji: "\U0001F388"
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
---

# Balloon IR Synthesizer

Convert balloon pop recordings into clean, full-bandwidth room impulse responses.

Based on Abel et al. (2010), *"Estimating Room Impulse Responses from Recorded Balloon Pops"*, AES Convention Paper 8171.

## Project Structure

```
core/           DSP pipeline (5-stage analysis-resynthesis)
api/            FastAPI web backend
frontend/       React + Vite frontend
scripts/        CLI tools (cli.py, generate_plots.py, validate_ir.py)
tests/          Unit tests
```

## Quick Start

### CLI

```bash
conda activate dsp
python scripts/cli.py balloon.wav -o ir.wav
```

### Local Web App

```bash
# Terminal 1: backend
uvicorn api.main:app --reload --port 8000

# Terminal 2: frontend
cd frontend && npm run dev
```

Open http://localhost:3000

## Deployment

- **Frontend**: GitHub Pages (auto-deploy on push via GitHub Actions)
- **Backend**: Hugging Face Spaces (Docker, free tier — 16 GB RAM)

### Known Limitations

- **In-memory job store**: Jobs are lost on container restart. For production use, replace with a database + file storage backend.
- **Cold starts**: HF Spaces may sleep after extended inactivity (~48h). Cold start involves rebuilding the Docker container (1-2 min).
- **Render free tier (deprecated)**: Previously used Render, but 512 MB RAM was insufficient for the DSP pipeline. Kept `render.yaml` for reference.

## References

Abel, J. S., Canfield-Dafilou, E. K., & Holloway, M. (2010). Estimating room impulse responses from recorded balloon pops. *Audio Engineering Society Convention 129*, Paper 8171.
