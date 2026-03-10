"""
FastAPI Application Entry Point

Balloon Pop → Room Impulse Response — Web API

Start the server:
    uvicorn api.main:app --reload --port 8000

Or:
    python -m api.main
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Balloon IR Synthesizer",
    description=(
        "Convert balloon pop recordings into clean, full-bandwidth "
        "room impulse responses. Based on Abel et al. (2010), "
        "AES Convention Paper 8171."
    ),
    version="0.1.0",
)

# CORS: allow frontend dev server + GitHub Pages
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://allenloves.github.io",
]
_extra = os.environ.get("CORS_ORIGINS", "")
if _extra:
    _cors_origins.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "Balloon IR Synthesizer",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
