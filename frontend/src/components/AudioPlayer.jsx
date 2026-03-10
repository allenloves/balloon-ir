import React, { useState, useRef, useCallback, useEffect } from 'react'

/**
 * Audition tab — load the synthesized IR and convolve a dry signal in real-time
 * using Web Audio API's ConvolverNode.
 */

const API_BASE = import.meta.env.VITE_API_URL || ''

const DEMO_MESSAGE =
  'Upload a dry audio file to audition the synthesized IR via real-time convolution.'

export default function AudioPlayer({ jobId }) {
  const [dryFile, setDryFile] = useState(null)
  const [irLoaded, setIrLoaded] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [mix, setMix] = useState(50) // dry/wet 0-100
  const [error, setError] = useState(null)

  const ctxRef = useRef(null)
  const irBufferRef = useRef(null)
  const sourceRef = useRef(null)
  const dryGainRef = useRef(null)
  const wetGainRef = useRef(null)
  const convolverRef = useRef(null)
  const dryBufferRef = useRef(null)
  const inputRef = useRef(null)

  // Load IR buffer on mount / jobId change
  useEffect(() => {
    let cancelled = false

    async function loadIR() {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)()
        ctxRef.current = ctx

        const res = await fetch(`${API_BASE}/api/result/${jobId}`)
        if (!res.ok) throw new Error('Failed to fetch IR')

        // The result endpoint returns a ZIP. We need to extract ir.wav.
        // Use JSZip-free approach: fetch the raw WAV from the ZIP.
        // Since we can't easily unzip in browser without a lib, we'll
        // add a dedicated audio endpoint. For now, use a simpler approach:
        // re-fetch the preview endpoint and see if there's an audio file,
        // or we create a dedicated endpoint. Let's use the result ZIP.
        //
        // Actually, let's just create a simple fetch for the WAV inside the ZIP.
        // We'll parse the ZIP manually for the ir.wav entry.
        const blob = await res.blob()
        const wavBlob = await extractWavFromZip(blob)
        if (cancelled) return

        if (!wavBlob) {
          setError('Could not extract ir.wav from result')
          return
        }

        const arrayBuf = await wavBlob.arrayBuffer()
        const audioBuffer = await ctx.decodeAudioData(arrayBuf)
        irBufferRef.current = audioBuffer
        setIrLoaded(true)
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    }

    loadIR()
    return () => {
      cancelled = true
    }
  }, [jobId])

  // Update gain nodes when mix changes
  useEffect(() => {
    const wet = mix / 100
    const dry = 1 - wet
    if (dryGainRef.current) dryGainRef.current.gain.value = dry
    if (wetGainRef.current) wetGainRef.current.gain.value = wet
  }, [mix])

  const handleDryFile = useCallback(
    async (e) => {
      const f = e.target.files?.[0]
      if (!f) return
      setDryFile(f)
      setError(null)

      try {
        const ctx = ctxRef.current
        if (!ctx) return
        const arrayBuf = await f.arrayBuffer()
        const audioBuffer = await ctx.decodeAudioData(arrayBuf)
        dryBufferRef.current = audioBuffer
      } catch (err) {
        setError(`Cannot decode audio: ${err.message}`)
      }
    },
    []
  )

  const stop = useCallback(() => {
    if (sourceRef.current) {
      try {
        sourceRef.current.stop()
      } catch {
        // already stopped
      }
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    setPlaying(false)
  }, [])

  const play = useCallback(() => {
    const ctx = ctxRef.current
    const irBuf = irBufferRef.current
    const dryBuf = dryBufferRef.current
    if (!ctx || !irBuf || !dryBuf) return

    stop()

    // Resume context if suspended (autoplay policy)
    if (ctx.state === 'suspended') ctx.resume()

    // Create convolver
    const convolver = ctx.createConvolver()
    convolver.buffer = irBuf
    convolverRef.current = convolver

    // Gain nodes for dry/wet mix
    const dryGain = ctx.createGain()
    const wetGain = ctx.createGain()
    dryGain.gain.value = 1 - mix / 100
    wetGain.gain.value = mix / 100
    dryGainRef.current = dryGain
    wetGainRef.current = wetGain

    // Source
    const source = ctx.createBufferSource()
    source.buffer = dryBuf
    sourceRef.current = source

    // Routing: source -> dryGain -> destination
    //          source -> convolver -> wetGain -> destination
    source.connect(dryGain)
    dryGain.connect(ctx.destination)

    source.connect(convolver)
    convolver.connect(wetGain)
    wetGain.connect(ctx.destination)

    source.onended = () => setPlaying(false)
    source.start()
    setPlaying(true)
  }, [mix, stop])

  return (
    <div>
      <div className="player">
        <div className="player__label">Impulse Response</div>
        {error && <p className="progress__error">{error}</p>}
        {!irLoaded && !error && (
          <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            Loading IR…
          </p>
        )}
        {irLoaded && (
          <p style={{ color: 'var(--green)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            IR loaded ({irBufferRef.current?.duration.toFixed(2)}s,{' '}
            {irBufferRef.current?.sampleRate} Hz)
          </p>
        )}
      </div>

      <div className="player" style={{ marginTop: '0.8rem' }}>
        <div className="player__label">Dry Signal</div>
        <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginBottom: '0.5rem' }}>
          {dryFile ? dryFile.name : DEMO_MESSAGE}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          onChange={handleDryFile}
          style={{ display: 'none' }}
        />
        <button
          className="upload-zone__clear"
          style={{ borderColor: 'var(--amber-500)', color: 'var(--amber-300)' }}
          onClick={() => inputRef.current?.click()}
        >
          Choose dry audio
        </button>
      </div>

      {/* Playback controls */}
      {dryFile && irLoaded && (
        <div className="player" style={{ marginTop: '0.8rem' }}>
          <div className="player__label">Playback</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.4rem' }}>
            <button
              className="process-btn"
              style={{ width: 'auto', padding: '0.5rem 1.5rem', fontSize: '0.8rem', marginTop: 0 }}
              onClick={playing ? stop : play}
            >
              {playing ? 'Stop' : 'Play'}
            </button>
            <div style={{ flex: 1 }}>
              <label
                className="param-field__label"
                style={{ marginBottom: '0.2rem', display: 'block' }}
              >
                Dry / Wet: {mix}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={mix}
                onChange={(e) => setMix(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--amber-400)' }}
              />
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6rem',
                  color: 'var(--text-muted)',
                }}
              >
                <span>Dry</span>
                <span>Wet</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Minimal ZIP parser — extract the first .wav file from a ZIP blob.
 * Handles only stored and deflated entries. Good enough for our single ir.wav.
 */
async function extractWavFromZip(blob) {
  const buf = await blob.arrayBuffer()
  const view = new DataView(buf)
  let offset = 0

  while (offset < buf.byteLength - 4) {
    const sig = view.getUint32(offset, true)
    if (sig !== 0x04034b50) break // PK\x03\x04

    const method = view.getUint16(offset + 8, true)
    const compSize = view.getUint32(offset + 18, true)
    const uncompSize = view.getUint32(offset + 22, true)
    const nameLen = view.getUint16(offset + 26, true)
    const extraLen = view.getUint16(offset + 28, true)
    const nameBytes = new Uint8Array(buf, offset + 30, nameLen)
    const name = new TextDecoder().decode(nameBytes)
    const dataStart = offset + 30 + nameLen + extraLen

    if (name.toLowerCase().endsWith('.wav')) {
      if (method === 0) {
        // Stored
        return new Blob([buf.slice(dataStart, dataStart + compSize)], {
          type: 'audio/wav',
        })
      } else if (method === 8) {
        // Deflated — use DecompressionStream
        const compressed = buf.slice(dataStart, dataStart + compSize)
        const ds = new DecompressionStream('deflate-raw')
        const writer = ds.writable.getWriter()
        writer.write(new Uint8Array(compressed))
        writer.close()
        const reader = ds.readable.getReader()
        const chunks = []
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value)
        }
        const totalLen = chunks.reduce((s, c) => s + c.byteLength, 0)
        const result = new Uint8Array(totalLen)
        let pos = 0
        for (const c of chunks) {
          result.set(c, pos)
          pos += c.byteLength
        }
        return new Blob([result], { type: 'audio/wav' })
      }
    }

    offset = dataStart + compSize
  }

  return null
}
