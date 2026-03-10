import React, { useState, useMemo } from 'react'
import AudioPlayer from './AudioPlayer'

const TABS = [
  { id: 'comparison', label: 'Comparison' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'audition', label: 'Audition' },
]

// Map plot stems to display names and tab assignments
const PLOT_TAB_MAP = {
  waveform_comparison: { tab: 'comparison', title: 'Waveform Comparison' },
  spectrogram_comparison: { tab: 'comparison', title: 'Spectrogram Comparison' },
  band_energy: { tab: 'comparison', title: 'Band Energy' },
  ned_profile: { tab: 'analysis', title: 'Normalized Echo Density' },
  echo_sequence: { tab: 'analysis', title: 'Echo Sequence' },
  iccc_profile: { tab: 'analysis', title: 'ICCC Profile' },
}

function StatCard({ label, value }) {
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className="stat__value">{value}</div>
    </div>
  )
}

export default function ResultsView({ jobId, summary, previews }) {
  const [activeTab, setActiveTab] = useState('comparison')

  const tabPlots = useMemo(() => {
    if (!previews) return {}
    const grouped = { comparison: [], analysis: [], audition: [] }
    for (const [stem, b64] of Object.entries(previews)) {
      const info = PLOT_TAB_MAP[stem]
      const tab = info?.tab || 'analysis'
      const title = info?.title || stem.replace(/_/g, ' ')
      grouped[tab].push({ stem, title, b64 })
    }
    return grouped
  }, [previews])

  return (
    <div className="results">
      <div className="results__header">
        <h2 className="results__title">Results</h2>
        <a
          className="results__download"
          href={`/api/result/${jobId}`}
          download
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path
              d="M7 1v9M3.5 7L7 10.5 10.5 7M2 12.5h10"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Download ZIP
        </a>
      </div>

      {/* Summary stats */}
      {summary && (
        <div className="summary-grid">
          <StatCard label="Sample Rate" value={`${summary.sr} Hz`} />
          <StatCard label="Channels" value={summary.is_stereo ? 'Stereo' : 'Mono'} />
          <StatCard
            label="IR Duration"
            value={`${summary.ir_duration_s?.toFixed(2)} s`}
          />
          <StatCard
            label="Balloon Radius"
            value={`${summary.balloon_radius_cm?.toFixed(1)} cm`}
          />
          <StatCard
            label="N-wave Duration"
            value={`${summary.nwave_duration_ms?.toFixed(2)} ms`}
          />
          <StatCard
            label="Early Reflections"
            value={summary.num_early_reflections}
          />
          {summary.transition_time_ms != null && (
            <StatCard
              label="Transition"
              value={`${summary.transition_time_ms?.toFixed(1)} ms`}
            />
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'tab--active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'audition' ? (
        <AudioPlayer jobId={jobId} />
      ) : (
        <div className="plot-grid">
          {(tabPlots[activeTab] || []).map(({ stem, title, b64 }) => (
            <div className="plot-card" key={stem}>
              <div className="plot-card__title">{title}</div>
              <img
                className="plot-card__img"
                src={`data:image/png;base64,${b64}`}
                alt={title}
              />
            </div>
          ))}
          {(tabPlots[activeTab] || []).length === 0 && (
            <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              No plots available for this tab.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
