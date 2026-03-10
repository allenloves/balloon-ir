import React, { useState, useCallback } from 'react'

const PARAM_DEFS = [
  { key: 'onset_threshold_db', label: 'Onset Thresh (dB)', type: 'number', min: -60, max: 0, step: 1 },
  { key: 'ned_window_ms', label: 'NED Window (ms)', type: 'number', min: 5, max: 100, step: 1 },
  { key: 'ned_transition_threshold', label: 'NED Transition', type: 'number', min: 0, max: 1, step: 0.05 },
  { key: 'num_early_reflections', label: 'Early Reflections', type: 'number', min: 1, max: 20, step: 1 },
  { key: 'pulse_halo_ms', label: 'Pulse Halo (ms)', type: 'number', min: 0, max: 10, step: 0.5 },
  { key: 'noise_floor_db', label: 'Noise Floor (dB)', type: 'number', min: -80, max: -10, step: 1 },
  { key: 'gain_smoothing_ms', label: 'Gain Smooth (ms)', type: 'number', min: 0, max: 10, step: 0.5 },
  { key: 'target_dbfs', label: 'Target (dBFS)', type: 'number', min: -12, max: 0, step: 0.5 },
  { key: 'output_bit_depth', label: 'Bit Depth', type: 'select', options: [16, 24, 32] },
]

export default function ParameterPanel({ params, onChange, disabled }) {
  const [open, setOpen] = useState(false)

  const handleChange = useCallback(
    (key, raw) => {
      const def = PARAM_DEFS.find((d) => d.key === key)
      const value = def?.type === 'number' || def?.type === 'select' ? Number(raw) : raw
      onChange((prev) => ({ ...prev, [key]: value }))
    },
    [onChange]
  )

  return (
    <div className="params">
      <button
        className={`params__toggle ${open ? 'params__toggle--open' : ''}`}
        onClick={() => setOpen((v) => !v)}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M4 2L8 6L4 10" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        Parameters
      </button>

      {open && (
        <div className="params__grid">
          {PARAM_DEFS.map((def) => (
            <div className="param-field" key={def.key}>
              <label className="param-field__label">{def.label}</label>
              {def.type === 'select' ? (
                <select
                  className="param-field__input"
                  value={params[def.key]}
                  onChange={(e) => handleChange(def.key, e.target.value)}
                  disabled={disabled}
                >
                  {def.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="param-field__input"
                  type="number"
                  value={params[def.key]}
                  min={def.min}
                  max={def.max}
                  step={def.step}
                  onChange={(e) => handleChange(def.key, e.target.value)}
                  disabled={disabled}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
