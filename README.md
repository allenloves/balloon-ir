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
- **Backend**: Render (free tier)

### Known Limitations

- **Render free tier**: Server sleeps after 15 minutes of inactivity. First request after sleep takes 30-60 seconds for cold start. In-memory job store is lost on restart.
- **Plot generation disabled on Render**: Matplotlib plot generation is too slow / memory-intensive for Render's free tier (512 MB RAM, shared CPU). Set `SKIP_PLOTS=1` to disable. Plots are still generated when running locally.
- **For production use**: Consider a persistent job store (database + file storage) and a host with dedicated CPU/RAM for DSP processing.

## References

Abel, J. S., Canfield-Dafilou, E. K., & Holloway, M. (2010). Estimating room impulse responses from recorded balloon pops. *Audio Engineering Society Convention 129*, Paper 8171.
