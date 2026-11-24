/**
 * Zero-dependency Int16Array Audio Splicer & Equal-Power Crossfade Engine for Ansitu (Web).
 *
 * Mirrors `quran_cleaner/processor.py` and `app.py` quality presets while storing
 * high-sample-rate audio in interleaved `Int16Array` buffers (50% less mobile RAM
 * than `Float32Array`, zero intermediate full-buffer copies).
 */

export const EDGE_FADE_IN_MS = 150;
export const EDGE_FADE_OUT_MS = 500;
export const INT16_MAX = 32767;
export const INT16_MIN = -32767;

/**
 * Exact output format labels from `app.py`.
 */
export const FORMAT_LABELS = Object.freeze({
  m4a: "M4A — جودة عالية (AAC High Quality)",
  mp3: "MP3 — متوافق مع كل الأجهزة (Universal)",
});

/**
 * Exact MIME types from `app.py`.
 */
export const MIME_TYPES = Object.freeze({
  m4a: "audio/mp4",
  mp3: "audio/mpeg",
  wav: "audio/wav",
});

/**
 * Exact 3 standard export quality presets defined in `app.py`.
 */
export const QUALITY_PRESETS = Object.freeze({
  "128": Object.freeze({
    id: "128",
    bitrateKbps: 128,
    label: "128 kbps (جودة يوتيوب - حجم خفيف)",
    isDefault: false,
    losslessOnly: false,
  }),
  "192": Object.freeze({
    id: "192",
    bitrateKbps: 192,
    label: "192 kbps (جودة عالية - الافتراضي)",
    isDefault: true,
    losslessOnly: false,
  }),
  "256": Object.freeze({
    id: "256",
    bitrateKbps: 256,
    label: "256 kbps (أقصى جودة أرشيفية)",
    isDefault: false,
    losslessOnly: false,
  }),
});

/**
 * Additional Lossless WAV preset unlocked in `app.py` when uploading `.wav` or `.flac`.
 */
export const LOSSLESS_WAV_PRESET = Object.freeze({
  id: "lossless_wav",
  bitrateKbps: null,
  format: "wav",
  mimeType: "audio/wav",
  label: "Lossless WAV (جودة أصلية بدون فقدان - PCM)",
  isDefault: false,
  losslessOnly: true,
});

export const DEFAULT_QUALITY_PRESET = "192";
export const LOSSLESS_EXTENSIONS = Object.freeze([".wav", ".flac"]);

/**
 * Return true when the input filename uses a lossless PCM/FLAC container (exact port of `is_lossless_source`).
 */
export function isLosslessSource(sourceName) {
  if (!sourceName || typeof sourceName !== "string") {
    return false;
  }
  const dotIdx = sourceName.lastIndexOf(".");
  if (dotIdx < 0) {
    return false;
  }
  const ext = sourceName.slice(dotIdx).toLowerCase();
  return LOSSLESS_EXTENSIONS.includes(ext);
}

/**
 * Return selectable quality preset IDs for a given source filename (exact port of `quality_options_for_source`).
 */
export function qualityOptionsForSource(sourceName) {
  const opts = ["128", "192", "256"];
  if (isLosslessSource(sourceName)) {
    opts.push("lossless_wav");
  }
  return opts;
}

/**
 * Clamp a floating-point or integer value cleanly into symmetric signed 16-bit PCM range `[-32767, 32767]`.
 */
export function clampInt16(val) {
  const rounded = Math.round(val);
  if (rounded > INT16_MAX) {
    return INT16_MAX;
  }
  if (rounded < INT16_MIN) {
    return INT16_MIN;
  }
  return rounded;
}

/**
 * Generate equal-power (cos/sin) crossfade curves over `length` samples.
 */
export function equalPowerCurves(length) {
  if (length <= 0) {
    return {
      fadeOut: new Float64Array(0),
      fadeIn: new Float64Array(0),
    };
  }
  const fadeOut = new Float64Array(length);
  const fadeIn = new Float64Array(length);
  if (length === 1) {
    // Match `np.linspace(0.0, np.pi / 2.0, 1, endpoint=True)` -> [0.0]
    fadeOut[0] = 1.0;
    fadeIn[0] = 0.0;
    return { fadeOut, fadeIn };
  }
  const step = (Math.PI * 0.5) / (length - 1);
  for (let i = 0; i < length; i++) {
    const theta = i * step;
    fadeOut[i] = Math.cos(theta);
    fadeIn[i] = Math.sin(theta);
  }
  return { fadeOut, fadeIn };
}

/**
 * Convert time segments to sample frame intervals `[startFrame, endFrame]`,
 * extending tails only when gaps require overlap headroom (exact port of `_build_sample_ranges`).
 */
export function buildSampleRanges(
  segments,
  sampleRate,
  totalFrames,
  fadeSamples,
  extendLast = false,
  extendTails = true
) {
  const ranges = [];
  const lastIdx = segments.length - 1;
  for (let idx = 0; idx < segments.length; idx++) {
    const seg = segments[idx];
    const sFrame = Math.max(0, Math.round(seg.start * sampleRate));
    let eFrame = Math.max(sFrame + 1, Math.round(seg.end * sampleRate));
    if (extendTails && (idx < lastIdx || extendLast) && fadeSamples > 0) {
      eFrame += fadeSamples;
      if (idx < lastIdx) {
        const nextStart = Math.max(0, Math.round(segments[idx + 1].start * sampleRate));
        eFrame = Math.min(eFrame, nextStart);
      }
    }
    if (totalFrames !== null && totalFrames !== undefined) {
      eFrame = Math.min(eFrame, totalFrames);
    }
    if (eFrame > sFrame) {
      ranges.push([sFrame, eFrame]);
    }
  }
  return ranges;
}

/**
 * Deterministic Mulberry32 PRNG for reproducible synthetic room-tone floor fallback.
 */
function createSeededRng(seed = 42) {
  let state = seed >>> 0;
  const nextUniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
  const nextNormal = () => {
    const u1 = Math.max(1e-12, nextUniform());
    const u2 = nextUniform();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };
  return { nextNormal };
}

/**
 * Create a smooth low-passed ambient noise fallback matching `floorDb` in interleaved `Int16Array`.
 */
function syntheticFloorBufferInt16(needed, channels, floorDb) {
  const targetDb = Math.min(-24.0, Math.max(-52.0, floorDb));
  const targetRms = 10.0 ** (targetDb / 20.0);
  const rng = createSeededRng(42);

  const rawLen = needed + 4;
  const raw = new Float64Array(rawLen * channels);
  for (let i = 0; i < raw.length; i++) {
    raw[i] = rng.nextNormal();
  }

  const soft = new Float64Array(needed * channels);
  const means = new Float64Array(channels);
  for (let f = 0; f < needed; f++) {
    const r0 = f * channels;
    const r1 = (f + 1) * channels;
    const r2 = (f + 2) * channels;
    const outOff = f * channels;
    for (let ch = 0; ch < channels; ch++) {
      const val = 0.25 * raw[r0 + ch] + 0.50 * raw[r1 + ch] + 0.25 * raw[r2 + ch];
      soft[outOff + ch] = val;
      means[ch] += val;
    }
  }
  for (let ch = 0; ch < channels; ch++) {
    means[ch] /= needed;
  }

  let sumSq = 0.0;
  for (let f = 0; f < needed; f++) {
    const outOff = f * channels;
    for (let ch = 0; ch < channels; ch++) {
      const v = soft[outOff + ch] - means[ch];
      soft[outOff + ch] = v;
      sumSq += v * v;
    }
  }
  const rms = Math.sqrt(sumSq / soft.length + 1e-12);
  const scale = (targetRms / rms) * INT16_MAX;

  const out = new Int16Array(needed * channels);
  for (let i = 0; i < soft.length; i++) {
    out[i] = clampInt16(soft[i] * scale);
  }
  return out;
}

/**
 * Build a native-rate room-tone bridge (vintage/live) or digital silence (studio) in `Int16Array`.
 */
export function buildRoomToneBufferInt16({
  pcm,
  sampleRate,
  channels = 1,
  vadResult,
  gapSamples,
  fadeSamples,
}) {
  if (gapSamples <= 0) {
    return null;
  }

  const needed = gapSamples + fadeSamples;
  const profile = vadResult?.profile;
  const useRoomTone = Boolean(vadResult?.use_room_tone);
  const floorP5 = profile?.noise_floor_p5_db ?? -60.0;
  const floorP10 = profile?.noise_floor_p10_db ?? floorP5;
  const gapWindow = vadResult?.quietest_gap_window ?? null;

  if (!useRoomTone || floorP5 <= -50.0) {
    return new Int16Array(needed * channels);
  }
  if (!gapWindow || !pcm || pcm.length === 0) {
    return syntheticFloorBufferInt16(needed, channels, floorP5);
  }

  const totalFrames = Math.floor(pcm.length / channels);
  const gStart = gapWindow[0];
  const gEnd = gapWindow[1];
  const dur = Math.max(0.20, gEnd - gStart);
  const sIdx = Math.max(0, Math.min(totalFrames - 1, Math.round(gStart * sampleRate)));
  const nRead = Math.min(totalFrames - sIdx, Math.max(1, Math.round(dur * sampleRate)));

  if (nRead < Math.max(64, Math.trunc(0.10 * sampleRate))) {
    return syntheticFloorBufferInt16(needed, channels, floorP5);
  }

  // Extract normalized Float64 grain for DC removal and floor capping.
  const grain = new Float64Array(nRead * channels);
  const means = new Float64Array(channels);
  for (let f = 0; f < nRead; f++) {
    const srcOff = (sIdx + f) * channels;
    const dstOff = f * channels;
    for (let ch = 0; ch < channels; ch++) {
      const v = pcm[srcOff + ch] / INT16_MAX;
      grain[dstOff + ch] = v;
      means[ch] += v;
    }
  }
  for (let ch = 0; ch < channels; ch++) {
    means[ch] /= nRead;
  }

  let sumSq = 0.0;
  for (let f = 0; f < nRead; f++) {
    const off = f * channels;
    for (let ch = 0; ch < channels; ch++) {
      const v = grain[off + ch] - means[ch];
      grain[off + ch] = v;
      sumSq += v * v;
    }
  }
  const grainRms = Math.sqrt(sumSq / grain.length + 1e-12);
  const grainDb = 20.0 * Math.log10(grainRms);
  const floorCapDb = Math.min(-22.0, Math.max(-48.0, floorP10 + 1.0));
  if (grainDb > floorCapDb) {
    const gain = 10.0 ** ((floorCapDb - grainDb) / 20.0);
    for (let i = 0; i < grain.length; i++) {
      grain[i] *= gain;
    }
  }

  if (nRead >= needed) {
    const out = new Int16Array(needed * channels);
    for (let i = 0; i < out.length; i++) {
      out[i] = clampInt16(grain[i] * INT16_MAX);
    }
    return out;
  }

  // Mirror and crossfade-loop when extracted grain is shorter than `needed`.
  const xf = Math.min(Math.floor(nRead / 4), Math.max(32, Math.trunc(0.02 * sampleRate)));
  const mirroredFrames = nRead * 2;
  const mirrored = new Float64Array(mirroredFrames * channels);
  mirrored.set(grain, 0);
  for (let f = 0; f < nRead; f++) {
    const srcOff = (nRead - 1 - f) * channels;
    const dstOff = (nRead + f) * channels;
    for (let ch = 0; ch < channels; ch++) {
      mirrored[dstOff + ch] = grain[srcOff + ch];
    }
  }

  const outFloat = new Float64Array(needed * channels);
  let pos = 0;
  const step = Math.max(1, mirroredFrames - xf);
  while (pos < needed) {
    const take = Math.min(mirroredFrames, needed - pos);
    if (pos === 0) {
      outFloat.set(mirrored.subarray(0, take * channels), 0);
    } else {
      const blendN = Math.min(xf, take);
      const { fadeOut, fadeIn } = equalPowerCurves(blendN);
      for (let f = 0; f < blendN; f++) {
        const outOff = (pos + f) * channels;
        const mirOff = f * channels;
        for (let ch = 0; ch < channels; ch++) {
          outFloat[outOff + ch] =
            outFloat[outOff + ch] * fadeOut[f] + mirrored[mirOff + ch] * fadeIn[f];
        }
      }
      if (take > blendN) {
        outFloat.set(
          mirrored.subarray(blendN * channels, take * channels),
          (pos + blendN) * channels
        );
      }
    }
    pos += pos + take < needed ? step : take;
  }

  const out = new Int16Array(needed * channels);
  for (let i = 0; i < outFloat.length; i++) {
    out[i] = clampInt16(outFloat[i] * INT16_MAX);
  }
  return out;
}

/**
 * Copy `frameCount` frames from `item` starting at `itemOffsetFrame` into `out` at `outOffsetFrame`.
 */
function copyItemFrames(out, outOffsetFrame, item, itemOffsetFrame, frameCount, channels) {
  if (frameCount <= 0) {
    return;
  }
  const srcStart = (item.startFrame + itemOffsetFrame) * channels;
  const srcEnd = srcStart + frameCount * channels;
  out.set(item.buffer.subarray(srcStart, srcEnd), outOffsetFrame * channels);
}

/**
 * Blend `blendFrames` overlapping frames from `prevTailItem` and `currItem` using equal-power cos/sin curves.
 */
function blendItemFrames(
  out,
  outOffsetFrame,
  prevItem,
  prevOffsetFrame,
  currItem,
  currOffsetFrame,
  blendFrames,
  channels,
  cachedCurves,
  fadeSamples
) {
  if (blendFrames <= 0) {
    return;
  }
  const { fadeOut, fadeIn } =
    blendFrames === fadeSamples && cachedCurves
      ? cachedCurves
      : equalPowerCurves(blendFrames);

  for (let f = 0; f < blendFrames; f++) {
    const pOff = (prevItem.startFrame + prevOffsetFrame + f) * channels;
    const cOff = (currItem.startFrame + currOffsetFrame + f) * channels;
    const oOff = (outOffsetFrame + f) * channels;
    const wOut = fadeOut[f];
    const wIn = fadeIn[f];
    for (let ch = 0; ch < channels; ch++) {
      out[oOff + ch] = clampInt16(prevItem.buffer[pOff + ch] * wOut + currItem.buffer[cOff + ch] * wIn);
    }
  }
}

/**
 * Core single-pass `Int16Array` stream splicer matching `CrossfadeSegmentStream` semantics.
 * Each item is `{ buffer: Int16Array, startFrame: number, length: number }`.
 */
function renderStreamItemsInt16({
  items,
  sampleRate,
  channels,
  crossfadeMs,
  fadeEdges = true,
  edgeFadeInMs = EDGE_FADE_IN_MS,
  edgeFadeOutMs = EDGE_FADE_OUT_MS,
}) {
  const nonEmpty = items.filter((it) => it && it.length > 0);
  if (nonEmpty.length === 0) {
    return {
      pcm: new Int16Array(0),
      frames: 0,
      duration: 0.0,
      sampleRate,
      channels,
    };
  }

  const fadeSamples = Math.max(0, Math.round((crossfadeMs / 1000.0) * sampleRate));

  // Fast zero-copy path for single untouched clip without edge fades (e.g., low_snr_guarded).
  if (nonEmpty.length === 1 && !fadeEdges) {
    const only = nonEmpty[0];
    const pcm = only.buffer.slice(
      only.startFrame * channels,
      (only.startFrame + only.length) * channels
    );
    return {
      pcm,
      frames: only.length,
      duration: only.length / sampleRate,
      sampleRate,
      channels,
    };
  }

  // Pass 1: compute exact output frame count matching CrossfadeSegmentStream state transitions.
  let outputFrames = 0;
  let heldTailLen = 0;
  const lastIdx = nonEmpty.length - 1;

  for (let idx = 0; idx < nonEmpty.length; idx++) {
    const len = nonEmpty[idx].length;
    const isLast = idx === lastIdx;

    if (!isLast) {
      if (heldTailLen > 0) {
        const n = Math.min(heldTailLen, len);
        outputFrames += n;
        if (heldTailLen > n) {
          outputFrames += heldTailLen - n;
        }
        const rem = len - n;
        if (rem > fadeSamples) {
          outputFrames += rem - fadeSamples;
          heldTailLen = fadeSamples;
        } else {
          heldTailLen = rem;
        }
      } else if (len > fadeSamples) {
        outputFrames += len - fadeSamples;
        heldTailLen = fadeSamples;
      } else {
        heldTailLen = len;
      }
    } else {
      if (heldTailLen > 0) {
        const n = Math.min(heldTailLen, len);
        outputFrames += n;
        if (heldTailLen > n) {
          outputFrames += heldTailLen - n;
        }
        if (len > n) {
          outputFrames += len - n;
        }
        heldTailLen = 0;
      } else {
        outputFrames += len;
      }
    }
  }

  // Pass 2: allocate exact Int16Array once and copy bodies / blend boundaries directly.
  const out = new Int16Array(outputFrames * channels);
  const cachedCurves = fadeSamples > 0 ? equalPowerCurves(fadeSamples) : null;
  let writeFrame = 0;
  let heldTailItem = null;
  let heldTailOffset = 0;
  heldTailLen = 0;

  for (let idx = 0; idx < nonEmpty.length; idx++) {
    const item = nonEmpty[idx];
    const len = item.length;
    const isLast = idx === lastIdx;

    if (!isLast) {
      let itemCursor = 0;
      if (heldTailLen > 0) {
        const n = Math.min(heldTailLen, len);
        blendItemFrames(
          out,
          writeFrame,
          heldTailItem,
          heldTailOffset,
          item,
          0,
          n,
          channels,
          cachedCurves,
          fadeSamples
        );
        writeFrame += n;
        if (heldTailLen > n) {
          copyItemFrames(out, writeFrame, heldTailItem, heldTailOffset + n, heldTailLen - n, channels);
          writeFrame += heldTailLen - n;
        }
        itemCursor = n;
        heldTailItem = null;
        heldTailOffset = 0;
        heldTailLen = 0;
      }

      const rem = len - itemCursor;
      if (rem > fadeSamples) {
        const emitLen = rem - fadeSamples;
        copyItemFrames(out, writeFrame, item, itemCursor, emitLen, channels);
        writeFrame += emitLen;
        heldTailItem = item;
        heldTailOffset = itemCursor + emitLen;
        heldTailLen = fadeSamples;
      } else if (rem > 0) {
        heldTailItem = item;
        heldTailOffset = itemCursor;
        heldTailLen = rem;
      }
    } else {
      let itemCursor = 0;
      if (heldTailLen > 0) {
        const n = Math.min(heldTailLen, len);
        blendItemFrames(
          out,
          writeFrame,
          heldTailItem,
          heldTailOffset,
          item,
          0,
          n,
          channels,
          cachedCurves,
          fadeSamples
        );
        writeFrame += n;
        if (heldTailLen > n) {
          copyItemFrames(out, writeFrame, heldTailItem, heldTailOffset + n, heldTailLen - n, channels);
          writeFrame += heldTailLen - n;
        }
        itemCursor = n;
        heldTailItem = null;
        heldTailOffset = 0;
        heldTailLen = 0;
      }
      if (len > itemCursor) {
        const rem = len - itemCursor;
        copyItemFrames(out, writeFrame, item, itemCursor, rem, channels);
        writeFrame += rem;
      }
    }
  }

  // Apply gentle output boundary fades in-place matching `CrossfadeSegmentStream`.
  if (fadeEdges && outputFrames > 0) {
    const fadeInSamples = Math.max(0, Math.round((edgeFadeInMs / 1000.0) * sampleRate));
    if (fadeInSamples > 0) {
      const nIn = Math.min(fadeInSamples, outputFrames);
      const scaleEnd = nIn / fadeInSamples;
      for (let f = 0; f < nIn; f++) {
        const gain = nIn > 1 ? (f / (nIn - 1)) * scaleEnd : 0.0;
        const off = f * channels;
        for (let ch = 0; ch < channels; ch++) {
          out[off + ch] = clampInt16(out[off + ch] * gain);
        }
      }
    }

    const fadeOutSamples = Math.max(0, Math.round((edgeFadeOutMs / 1000.0) * sampleRate));
    if (fadeOutSamples > 0) {
      const nOut = Math.min(fadeOutSamples, outputFrames);
      if (nOut > 1) {
        const startF = outputFrames - nOut;
        for (let f = 0; f < nOut; f++) {
          const gain = 1.0 - f / (nOut - 1);
          const off = (startF + f) * channels;
          for (let ch = 0; ch < channels; ch++) {
            out[off + ch] = clampInt16(out[off + ch] * gain);
          }
        }
      }
    }
  }

  return {
    pcm: out,
    frames: outputFrames,
    duration: outputFrames / sampleRate,
    sampleRate,
    channels,
  };
}

/**
 * Slice and equal-power crossfade segments directly from an interleaved `Int16Array`.
 */
export function spliceAudioInt16({
  pcm,
  sampleRate,
  channels = 1,
  segments,
  crossfadeMs = 0,
  gapMs = 0,
  fadeEdges = true,
  edgeFadeInMs = EDGE_FADE_IN_MS,
  edgeFadeOutMs = EDGE_FADE_OUT_MS,
  useRoomTone = false,
  quietestGapWindow = null,
  profile = null,
}) {
  if (!segments || segments.length === 0) {
    throw new Error("No speech segments provided for splicing.");
  }

  const totalFrames = Math.floor(pcm.length / channels);
  const fadeSamples = Math.max(0, Math.round((crossfadeMs / 1000.0) * sampleRate));
  const gapSamples = Math.max(0, Math.round((gapMs / 1000.0) * sampleRate));
  const sRanges = buildSampleRanges(
    segments,
    sampleRate,
    totalFrames,
    fadeSamples,
    false,
    gapSamples > 0
  );
  if (sRanges.length === 0) {
    throw new Error("No valid audio sample ranges remained after slicing.");
  }

  const needsGap = gapSamples > 0 && sRanges.length > 1;
  const gapBuffer = needsGap
    ? buildRoomToneBufferInt16({
        pcm,
        sampleRate,
        channels,
        vadResult: {
          use_room_tone: useRoomTone,
          quietest_gap_window: quietestGapWindow,
          profile: profile ?? { noise_floor_p5_db: -60.0, noise_floor_p10_db: -60.0 },
        },
        gapSamples,
        fadeSamples,
      })
    : null;

  const streamItems = [];
  for (let idx = 0; idx < sRanges.length; idx++) {
    const [sFrame, eFrame] = sRanges[idx];
    streamItems.push({
      buffer: pcm,
      startFrame: sFrame,
      length: eFrame - sFrame,
    });
    if (idx < sRanges.length - 1 && gapBuffer && gapBuffer.length > 0) {
      streamItems.push({
        buffer: gapBuffer,
        startFrame: 0,
        length: Math.floor(gapBuffer.length / channels),
      });
    }
  }

  return renderStreamItemsInt16({
    items: streamItems,
    sampleRate,
    channels,
    crossfadeMs,
    fadeEdges,
    edgeFadeInMs,
    edgeFadeOutMs,
  });
}

/**
 * Stitch and splice one or more VADResult range groups seamlessly in `Int16Array`
 * (exact port of `_build_range_stream_items` + `render_clean_ranges`).
 */
export function spliceRangesInt16({
  pcm,
  sampleRate,
  channels = 1,
  rangeResults,
  crossfadeMs = null,
  fadeEdges = null,
  edgeFadeInMs = EDGE_FADE_IN_MS,
  edgeFadeOutMs = EDGE_FADE_OUT_MS,
}) {
  if (!rangeResults || rangeResults.length === 0) {
    throw new Error("At least one range result is required.");
  }

  const effCrossfadeMs =
    crossfadeMs !== null && crossfadeMs !== undefined
      ? crossfadeMs
      : Math.max(...rangeResults.map((vr) => vr.resolved_crossfade_ms ?? 0));
  const effFadeEdges =
    fadeEdges !== null && fadeEdges !== undefined
      ? fadeEdges
      : !rangeResults.every((vr) => Boolean(vr.low_snr_guarded));

  const totalFrames = Math.floor(pcm.length / channels);
  const fadeSamples = Math.max(0, Math.round((effCrossfadeMs / 1000.0) * sampleRate));
  const perRange = [];
  const lastRangeIdx = rangeResults.length - 1;

  for (let rIdx = 0; rIdx < rangeResults.length; rIdx++) {
    const vr = rangeResults[rIdx];
    if (!vr.segments || vr.segments.length === 0) {
      throw new Error(`Range #${rIdx + 1} had no speech segments detected.`);
    }
    const isLastRange = rIdx === lastRangeIdx;
    const gapSamples = Math.max(0, Math.round(((vr.resolved_gap_ms ?? 0) / 1000.0) * sampleRate));
    const sRanges = buildSampleRanges(
      vr.segments,
      sampleRate,
      totalFrames,
      fadeSamples,
      !isLastRange,
      gapSamples > 0
    );
    if (sRanges.length === 0) {
      continue;
    }
    const needsGap = gapSamples > 0 && (sRanges.length > 1 || !isLastRange);
    const gapBuf = needsGap
      ? buildRoomToneBufferInt16({
          pcm,
          sampleRate,
          channels,
          vadResult: vr,
          gapSamples,
          fadeSamples,
        })
      : null;
    perRange.push({ sRanges, gapBuf });
  }

  if (perRange.length === 0) {
    throw new Error("No valid audio sample ranges remained after slicing.");
  }

  const streamItems = [];
  const lastGroupIdx = perRange.length - 1;
  for (let gIdx = 0; gIdx < perRange.length; gIdx++) {
    const { sRanges, gapBuf } = perRange[gIdx];
    const isLastGroup = gIdx === lastGroupIdx;
    const lastSegIdx = sRanges.length - 1;
    for (let sIdx = 0; sIdx < sRanges.length; sIdx++) {
      const [sFrame, eFrame] = sRanges[sIdx];
      const isLastOverall = isLastGroup && sIdx === lastSegIdx;
      streamItems.push({
        buffer: pcm,
        startFrame: sFrame,
        length: eFrame - sFrame,
      });
      if (!isLastOverall && gapBuf && gapBuf.length > 0) {
        streamItems.push({
          buffer: gapBuf,
          startFrame: 0,
          length: Math.floor(gapBuf.length / channels),
        });
      }
    }
  }

  return renderStreamItemsInt16({
    items: streamItems,
    sampleRate,
    channels,
    crossfadeMs: effCrossfadeMs,
    fadeEdges: effFadeEdges,
    edgeFadeInMs,
    edgeFadeOutMs,
  });
}

/**
 * Convert planar channel `Float32Array[]` or interleaved `Float32Array` to interleaved `Int16Array`.
 */
export function float32ToInt16(input, channels = 1) {
  if (Array.isArray(input)) {
    const chCount = input.length;
    const frames = chCount > 0 ? input[0].length : 0;
    const out = new Int16Array(frames * chCount);
    for (let f = 0; f < frames; f++) {
      const base = f * chCount;
      for (let ch = 0; ch < chCount; ch++) {
        const s = Math.max(-1.0, Math.min(1.0, input[ch][f]));
        out[base + ch] = clampInt16(s * INT16_MAX);
      }
    }
    return out;
  }

  const total = input.length;
  const out = new Int16Array(total);
  for (let i = 0; i < total; i++) {
    const s = Math.max(-1.0, Math.min(1.0, input[i]));
    out[i] = clampInt16(s * INT16_MAX);
  }
  return out;
}

/**
 * Convert interleaved `Int16Array` back to planar `Float32Array[]` per channel (e.g., for Web Audio preview).
 */
export function int16ToFloat32Channels(pcm, channels = 1) {
  const frames = Math.floor(pcm.length / channels);
  const planar = Array.from({ length: channels }, () => new Float32Array(frames));
  const inv = 1.0 / INT16_MAX;
  for (let f = 0; f < frames; f++) {
    const base = f * channels;
    for (let ch = 0; ch < channels; ch++) {
      planar[ch][f] = pcm[base + ch] * inv;
    }
  }
  return planar;
}

/**
 * Encode an interleaved `Int16Array` into a standard 44-byte RIFF/WAVE PCM-16 `ArrayBuffer`.
 */
export function encodeWavInt16(pcm, sampleRate, channels = 1) {
  const dataBytes = pcm.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeAscii = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // AudioFormat = 1 (PCM)
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true); // ByteRate
  view.setUint16(32, channels * 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);

  const dst = new Int16Array(buffer, 44, pcm.length);
  dst.set(pcm);
  return buffer;
}

const ARABIC_DIGIT_MAP = {
  "٠": "0",
  "١": "1",
  "٢": "2",
  "٣": "3",
  "٤": "4",
  "٥": "5",
  "٦": "6",
  "٧": "7",
  "٨": "8",
  "٩": "9",
  "۰": "0",
  "۱": "1",
  "۲": "2",
  "۳": "3",
  "۴": "4",
  "۵": "5",
  "۶": "6",
  "۷": "7",
  "۸": "8",
  "۹": "9",
};

/**
 * Convert Arabic-Indic / Eastern Arabic digits to ASCII and normalize
 * `.`, `,`, `-`, spaces, and fullwidth colons to `:`.
 */
export function normalizeTimestampSeparators(value) {
  return String(value ?? "")
    .replace(/[٠-٩۰-۹]/g, (ch) => ARABIC_DIGIT_MAP[ch] || ch)
    .replace(/[\s.,،٫\-：]+/g, ":")
    .replace(/:+/g, ":");
}

/**
 * Format 1-6 pure digits as a clock string (`0045` -> `00:45`, `130` -> `1:30`, `10530` -> `1:05:30`).
 */
export function formatTimestampDigits(digits) {
  const d = String(digits ?? "").replace(/\D/g, "");
  if (d.length <= 2) {
    return d;
  }
  if (d.length === 3) {
    return `${d.slice(0, 1)}:${d.slice(1)}`;
  }
  if (d.length === 4) {
    return `${d.slice(0, 2)}:${d.slice(2)}`;
  }
  if (d.length === 5) {
    return `${d.slice(0, 1)}:${d.slice(1, 3)}:${d.slice(3)}`;
  }
  return `${d.slice(0, 2)}:${d.slice(2, 4)}:${d.slice(4, 6)}`;
}

/**
 * Parse flexible Prayer Mode timestamps into non-negative seconds:
 * - Treats `.`, `,`, `-`, and spaces as `:` (`00.45` -> `00:45` -> 45s, NEVER 0.45s).
 * - Supports pure digits (`0045` -> 45s, `130` -> 90s, `10530` -> 3930s).
 * - Supports `MM:SS` and `HH:MM:SS`.
 */
export function parseTimestamp(value) {
  const raw = normalizeTimestampSeparators(value).trim();
  if (!raw) {
    throw new Error("INVALID_TIMESTAMP");
  }

  if (!raw.includes(":")) {
    if (!/^\d+$/.test(raw) || raw.length > 6) {
      throw new Error("INVALID_TIMESTAMP");
    }
    if (raw.length <= 2) {
      return Number(raw);
    }
    const formatted = formatTimestampDigits(raw);
    return parseTimestamp(formatted);
  }

  const parts = raw.split(":");
  if (parts.length !== 2 && parts.length !== 3) {
    throw new Error("INVALID_TIMESTAMP");
  }
  if (parts.some((p) => !/^\d+$/.test(p))) {
    throw new Error("INVALID_TIMESTAMP");
  }

  const nums = parts.map((p) => Number(p));
  if (nums.length === 2) {
    const [mins, secs] = nums;
    if (mins < 0 || secs < 0 || secs >= 60) {
      throw new Error("INVALID_TIMESTAMP");
    }
    return mins * 60 + secs;
  }

  const [hours, mins, secs] = nums;
  if (hours < 0 || mins < 0 || secs < 0 || mins >= 60 || secs >= 60) {
    throw new Error("INVALID_TIMESTAMP");
  }
  return hours * 3600 + mins * 60 + secs;
}

/**
 * Canonicalize a timestamp input on blur without altering invalid user text:
 * `00.45` -> `00:45`, `0045` -> `00:45`, `130` -> `01:30`, `10530` -> `1:05:30`.
 */
export function formatTimestampOnBlur(value) {
  const normalized = normalizeTimestampSeparators(value).trim();
  if (!normalized) {
    return "";
  }
  let secs;
  try {
    secs = parseTimestamp(normalized);
  } catch {
    return normalized;
  }
  const pad = (n) => String(n).padStart(2, "0");
  const hours = Math.floor(secs / 3600);
  const rem = secs % 3600;
  const mins = Math.floor(rem / 60);
  const s = rem % 60;
  if (hours > 0) {
    const hStr = hours >= 10 || normalized.startsWith("0") ? pad(hours) : String(hours);
    return `${hStr}:${pad(mins)}:${pad(s)}`;
  }
  return `${pad(mins)}:${pad(s)}`;
}

/**
 * Resolve user-supplied Prayer Mode ranges against `totalDuration`.
 *
 * - `rangeAction === "keep"` (default): validates and clamps each requested range in order.
 * - `rangeAction === "remove"`: merges overlapping/adjacent exclusion ranges, clamps to
 *   `[0, totalDuration]`, and returns the complement intervals to keep. Throws
 *   `ALL_AUDIO_REMOVED` if nothing remains.
 */
export function resolveEffectiveRanges(ranges, totalDuration, rangeAction = "keep") {
  const dur = Number(totalDuration);
  if (!Number.isFinite(dur) || dur <= 0) {
    throw new Error("INVALID_DURATION");
  }
  if (!Array.isArray(ranges) || ranges.length === 0) {
    return [{ startS: 0.0, endS: dur }];
  }

  const normalized = [];
  for (const r of ranges) {
    const rawStart =
      r.startS !== null && r.startS !== undefined ? Number(r.startS) : 0.0;
    const rawEnd =
      r.endS !== null && r.endS !== undefined ? Number(r.endS) : dur;
    const clipStart = Math.max(0.0, rawStart);
    const clipEnd = Math.min(rawEnd, dur);

    if (clipStart >= dur) {
      throw new Error("RANGE_EXCEEDS_DURATION");
    }
    if (clipStart >= clipEnd) {
      throw new Error("INVALID_RANGE_ORDER");
    }
    normalized.push({ startS: clipStart, endS: clipEnd });
  }

  if (rangeAction !== "remove") {
    return normalized;
  }

  // Merge overlapping or adjacent exclusion ranges
  const sorted = normalized
    .slice()
    .sort((a, b) => a.startS - b.startS || a.endS - b.endS);
  const merged = [];
  for (const seg of sorted) {
    if (merged.length === 0 || seg.startS > merged[merged.length - 1].endS + 1e-6) {
      merged.push({ startS: seg.startS, endS: seg.endS });
    } else {
      merged[merged.length - 1].endS = Math.max(
        merged[merged.length - 1].endS,
        seg.endS
      );
    }
  }

  // Compute complement intervals against [0, dur]
  const kept = [];
  let cursor = 0.0;
  for (const ex of merged) {
    if (ex.startS - cursor > 1e-3) {
      kept.push({ startS: cursor, endS: ex.startS });
    }
    cursor = Math.max(cursor, ex.endS);
  }
  if (dur - cursor > 1e-3) {
    kept.push({ startS: cursor, endS: dur });
  }

  if (kept.length === 0) {
    throw new Error("ALL_AUDIO_REMOVED");
  }
  return kept;
}

/**
 * Decide how to proceed after VAD analysis across `effectiveRanges`.
 *
 * - Normal Mode (or Prayer Mode with no user ranges): preserves exact safety-guard
 *   abort on `low_snr_guarded` and throws `NO_SPEECH_DETECTED` if any range is empty.
 * - Prayer Mode with user ranges: NEVER discards the user's montage.
 *   - `silenceTrimStatus === "skipped"` (`silenceTrimSkipped === true`): ONLY when ALL
 *     ranges skipped automatic silence trimming (so ONLY user ranges are kept/removed).
 *   - `silenceTrimStatus === "partial"` (`silenceTrimSkipped === false`): some clean
 *     ranges had pauses trimmed while noisy/quiet ranges (>= 1.2s) were kept intact.
 *   - `silenceTrimStatus === "trimmed"` (`silenceTrimSkipped === false`): all ranges
 *     analyzed cleanly.
 */
export function applyPrayerModeFallback(
  rangeResults,
  effectiveRanges,
  { isPrayerMode = false, hasUserRanges = false } = {}
) {
  const preserveMontage = Boolean(isPrayerMode && hasUserRanges);

  if (!preserveMontage) {
    const allGuarded = rangeResults.every((r) => Boolean(r.low_snr_guarded));
    if (allGuarded) {
      return {
        action: "abort_low_snr",
        rangeResults,
        silenceTrimSkipped: false,
        silenceTrimStatus: "skipped",
        skippedRangesCount: rangeResults.length,
        totalRangesCount: rangeResults.length,
      };
    }
    if (rangeResults.some((r) => !r.segments || r.segments.length === 0)) {
      throw new Error("NO_SPEECH_DETECTED");
    }
    return {
      action: "splice",
      rangeResults,
      silenceTrimSkipped: false,
      silenceTrimStatus: "trimmed",
      skippedRangesCount: 0,
      totalRangesCount: rangeResults.length,
    };
  }

  // Sub-1.2s residual edge slivers cannot have >= 1.2s silences cut by VAD anyway;
  // keep them intact without misreporting a noisy-section notice when other ranges are clean.
  const MIN_TRIMMABLE_RANGE_S = 1.2;
  let skippedAnyCount = 0;
  let skippedSignificantCount = 0;
  let cleanCount = 0;
  const fallbackCrossfadeMs = effectiveRanges.length > 1 ? 80 : 0;

  const adjusted = rangeResults.map((res, idx) => {
    const eff = effectiveRanges[idx];
    const dur = Math.max(0.0, eff.endS - eff.startS);
    const isGuarded = Boolean(res.low_snr_guarded);
    const isEmpty = !res.segments || res.segments.length === 0;
    if (!isGuarded && !isEmpty) {
      cleanCount += 1;
      return res;
    }
    skippedAnyCount += 1;
    if (effectiveRanges.length === 1 || dur >= MIN_TRIMMABLE_RANGE_S) {
      skippedSignificantCount += 1;
    }
    return {
      ...res,
      segments: [
        {
          start: eff.startS,
          end: eff.endS,
          duration: dur,
        },
      ],
      resolved_crossfade_ms: Math.max(
        res.resolved_crossfade_ms ?? 0,
        fallbackCrossfadeMs
      ),
      resolved_gap_ms: 0,
      use_room_tone: false,
      direct_overlap: false,
    };
  });

  const allSkipped = skippedAnyCount === rangeResults.length;
  let silenceTrimStatus = "trimmed";
  if (allSkipped) {
    silenceTrimStatus = "skipped";
  } else if (skippedSignificantCount > 0 && cleanCount > 0) {
    silenceTrimStatus = "partial";
  }

  return {
    action: "splice",
    rangeResults: adjusted,
    silenceTrimSkipped: allSkipped,
    silenceTrimStatus,
    skippedRangesCount: skippedSignificantCount,
    totalRangesCount: rangeResults.length,
  };
}
