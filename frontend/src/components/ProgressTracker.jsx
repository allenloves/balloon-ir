import React from 'react'

export default function ProgressTracker({ status, progress, message, error }) {
  return (
    <div className="progress">
      <div className="progress__header">
        <span className="progress__label">
          {status === 'done'
            ? 'Complete'
            : status === 'error'
              ? 'Error'
              : 'Processing'}
        </span>
        <span className="progress__pct">{Math.round(progress)}%</span>
      </div>
      <div className="progress__bar-track">
        <div
          className="progress__bar-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      {message && <p className="progress__message">{message}</p>}
      {error && <p className="progress__error">{error}</p>}
    </div>
  )
}
