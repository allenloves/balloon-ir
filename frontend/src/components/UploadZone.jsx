import React, { useCallback, useRef, useState } from 'react'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadZone({ file, onFileChange, disabled }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFile = useCallback(
    (f) => {
      if (disabled) return
      if (f && /\.wav$/i.test(f.name)) {
        onFileChange(f)
      }
    },
    [onFileChange, disabled]
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragActive(false)
      const f = e.dataTransfer?.files?.[0]
      if (f) handleFile(f)
    },
    [handleFile]
  )

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    setDragActive(true)
  }, [])

  const onDragLeave = useCallback(() => {
    setDragActive(false)
  }, [])

  const onClick = useCallback(() => {
    if (!disabled) inputRef.current?.click()
  }, [disabled])

  const onInputChange = useCallback(
    (e) => {
      const f = e.target.files?.[0]
      if (f) handleFile(f)
      e.target.value = ''
    },
    [handleFile]
  )

  const clearFile = useCallback(
    (e) => {
      e.stopPropagation()
      onFileChange(null)
    },
    [onFileChange]
  )

  const classes = [
    'upload-zone',
    dragActive && 'upload-zone--active',
    file && 'upload-zone--has-file',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={classes}
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".wav"
        onChange={onInputChange}
        style={{ display: 'none' }}
      />

      {file ? (
        <div className="upload-zone__file-info">
          <span className="upload-zone__filename">{file.name}</span>
          <span className="upload-zone__meta">{formatSize(file.size)}</span>
          {!disabled && (
            <button className="upload-zone__clear" onClick={clearFile}>
              Remove
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="upload-zone__icon">⦿</div>
          <p className="upload-zone__text">
            Drop a balloon pop <strong>.wav</strong> here, or click to browse
          </p>
          <p className="upload-zone__hint">
            Mono or stereo &middot; Any sample rate
          </p>
        </>
      )}
    </div>
  )
}
