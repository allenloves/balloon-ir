// Minimal WAV decoder/encoder for PCM 16/24/32-bit and 32-bit float.
// Returns/accepts deinterleaved Float32Array channels in [-1, 1].

export function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  if (str(view, 0, 4) !== "RIFF" || str(view, 8, 4) !== "WAVE") {
    throw new Error("Not a RIFF/WAVE file");
  }

  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset < view.byteLength - 8) {
    const id = str(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      fmt = {
        format: view.getUint16(body + 0, true),         // 1 = PCM, 3 = float, 0xFFFE = extensible
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        byteRate: view.getUint32(body + 8, true),
        blockAlign: view.getUint16(body + 12, true),
        bitsPerSample: view.getUint16(body + 14, true),
      };
      // WAVE_FORMAT_EXTENSIBLE: actual format code lives at body+24..26
      if (fmt.format === 0xfffe && size >= 40) {
        fmt.format = view.getUint16(body + 24, true);
      }
    } else if (id === "data") {
      dataOffset = body;
      dataSize = size;
      break;
    }
    offset = body + size + (size & 1); // chunks are word-aligned
  }

  if (!fmt) throw new Error("Missing fmt chunk");
  if (dataOffset < 0) throw new Error("Missing data chunk");

  const { format, channels, sampleRate, bitsPerSample } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const frameCount = dataSize / (bytesPerSample * channels);

  const out = [];
  for (let c = 0; c < channels; c++) out.push(new Float32Array(frameCount));

  for (let i = 0; i < frameCount; i++) {
    for (let c = 0; c < channels; c++) {
      const p = dataOffset + (i * channels + c) * bytesPerSample;
      let s;
      if (format === 1) {
        if (bitsPerSample === 16) {
          s = view.getInt16(p, true) / 0x8000;
        } else if (bitsPerSample === 24) {
          // Little-endian 3-byte signed
          const b0 = view.getUint8(p);
          const b1 = view.getUint8(p + 1);
          const b2 = view.getUint8(p + 2);
          let v = b0 | (b1 << 8) | (b2 << 16);
          if (v & 0x800000) v |= 0xff000000; // sign-extend to 32 bits
          s = v / 0x800000;
        } else if (bitsPerSample === 32) {
          s = view.getInt32(p, true) / 0x80000000;
        } else if (bitsPerSample === 8) {
          s = (view.getUint8(p) - 128) / 128;
        } else {
          throw new Error(`Unsupported PCM bit depth: ${bitsPerSample}`);
        }
      } else if (format === 3 && bitsPerSample === 32) {
        s = view.getFloat32(p, true);
      } else if (format === 3 && bitsPerSample === 64) {
        s = view.getFloat64(p, true);
      } else {
        throw new Error(`Unsupported WAV format ${format}/${bitsPerSample}-bit`);
      }
      out[c][i] = s;
    }
  }

  return { channels: out, sampleRate, bitsPerSample, format };
}

// Encode deinterleaved Float32 channels as a PCM WAV. bitDepth: 16 | 24 | 32 (float).
export function encodeWav(channels, sampleRate, bitDepth = 24) {
  const numChannels = channels.length;
  const numFrames = channels[0].length;
  const isFloat = bitDepth === 32;
  const bytesPerSample = isFloat ? 4 : bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = numFrames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(view, 8, "WAVE");
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, isFloat ? 3 : 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeStr(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numChannels; c++) {
      const s = clamp(channels[c][i], -1, 1);
      if (isFloat) {
        view.setFloat32(offset, s, true);
        offset += 4;
      } else if (bitDepth === 16) {
        view.setInt16(offset, Math.round(s * 0x7fff), true);
        offset += 2;
      } else if (bitDepth === 24) {
        const v = Math.round(s * 0x7fffff);
        view.setUint8(offset, v & 0xff);
        view.setUint8(offset + 1, (v >> 8) & 0xff);
        view.setUint8(offset + 2, (v >> 16) & 0xff);
        offset += 3;
      } else {
        throw new Error(`Unsupported bit depth ${bitDepth}`);
      }
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function str(view, offset, len) {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

function writeStr(view, offset, s) {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}

function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}
