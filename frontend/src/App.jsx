import React, { useState, useCallback, useRef } from 'react'
import UploadZone from './components/UploadZone'
import ParameterPanel from './components/ParameterPanel'
import ProgressTracker from './components/ProgressTracker'
import ResultsView from './components/ResultsView'

const API_BASE = import.meta.env.VITE_API_URL || ''

const DEFAULT_PARAMS = {
  onset_threshold_db: -40,
  ned_window_ms: 43,
  ned_transition_threshold: 0.3,
  num_early_reflections: 2,
  pulse_halo_ms: 2.0,
  noise_floor_db: -40,
  gain_smoothing_ms: 0,
  target_dbfs: -1.0,
  output_bit_depth: 24,
}

export default function App() {
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null) // null | {status, progress, message, error, summary}
  const [previews, setPreviews] = useState(null)
  const pollRef = useRef(null)

  const resetJob = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setJobId(null)
    setStatus(null)
    setPreviews(null)
  }, [])

  const pollStatus = useCallback((id) => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${id}`)
        if (!res.ok) return
        const data = await res.json()
        setStatus(data)

        if (data.status === 'done' || data.status === 'error') {
          clearInterval(pollRef.current)
          pollRef.current = null

          // Fetch preview plots when done
          if (data.status === 'done') {
            const prev = await fetch(`${API_BASE}/api/preview/${id}`)
            if (prev.ok) {
              const prevData = await prev.json()
              setPreviews(prevData.plots)
            }
          }
        }
      } catch {
        // Network error — keep polling
      }
    }

    poll() // Immediate first poll
    pollRef.current = setInterval(poll, 800)
  }, [])

  const handleProcess = useCallback(async () => {
    if (!file) return
    resetJob()

    const formData = new FormData()
    formData.append('file', file)
    formData.append('params', JSON.stringify(params))

    try {
      const res = await fetch(`${API_BASE}/api/process`, { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json()
        setStatus({ status: 'error', progress: 0, message: '', error: err.detail || 'Upload failed' })
        return
      }
      const data = await res.json()
      setJobId(data.job_id)
      setStatus({ status: 'queued', progress: 0, message: 'Queued...' })
      pollStatus(data.job_id)
    } catch (e) {
      setStatus({ status: 'error', progress: 0, message: '', error: e.message })
    }
  }, [file, params, resetJob, pollStatus])

  const isProcessing = status && (status.status === 'queued' || status.status === 'processing')
  const isDone = status && status.status === 'done'

  return (
    <div className="app">
      <header className="header">
        <h1 className="header__title">
          Balloon <em>IR</em> Synthesizer
        </h1>
        <p className="header__subtitle">
          Balloon pop &rarr; Room impulse response by Allen SC Wu, based on{' '}
          <a
            href="https://www.researchgate.net/profile/Bissera-Pentcheva/publication/277732009_Estimating_Room_Impulse_Responses_from_Recorded_Balloon_Pops/links/563a4b4408ae45b5d284a8ce/Estimating-Room-Impulse-Responses-from-Recorded-Balloon-Pops.pdf"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--text-muted)', textDecoration: 'underline' }}
          >
            Abel et al. 2010
          </a>
        </p>
      </header>

      <UploadZone file={file} onFileChange={setFile} disabled={isProcessing} />

      <ParameterPanel params={params} onChange={setParams} disabled={isProcessing} />

      <button
        className="process-btn"
        disabled={!file || isProcessing}
        onClick={handleProcess}
      >
        {isProcessing ? 'Processing…' : 'Synthesize IR'}
      </button>

      {status && (
        <ProgressTracker
          status={status.status}
          progress={status.progress}
          message={status.message}
          error={status.error}
        />
      )}

      {isDone && (
        <ResultsView
          jobId={jobId}
          summary={status.summary}
          previews={previews}
        />
      )}
    </div>
  )
}
