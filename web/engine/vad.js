/**
 * Pure Energy DSP Voice Activity Detection & Auto-Profiler for Ansitu (Web).
 *
 * Zero-dependency ES Module port of `quran_cleaner/vad.py` preserving:
 * - 32ms (512-sample) non-overlapping RMS framing at 16 kHz
 * - Studio (< -55 dBFS p5) vs Vintage/Live auto-profiling & dense-recitation reclassification
 * - Low-SNR Safety Gate (< 12.0 dB SNR -> keep full audio untouched)
 * - Hysteresis energy island detection + whispered Quranic letter boundary protection
 * - Studio high-frequency breath rejection & vintage unpitched HF noise-blip guard
 * - Ultra-safe crowd murmur cluster stripping (<= 3.5s, >= 10..14 dB below reciter median)
 * - Dynamic pad_end (1100/1300ms vintage/live, 300ms studio) & >= 1.2s effective cut guard
 */

export const VAD_SAMPLE_RATE = 16000;
export const FRAME_SAMPLES = 512; // 32ms windows at 16kHz
export const MIN_SPEECH_MS = 200;
export const STUDIO_MIN_SPEECH_MS = 220;
export const LOW_SNR_GUARD_DB = 12.0;
export const MIN_VINTAGE_EFFECTIVE_CUT_S = 1.2;

export const DEFAULT_VAD_CONFIG = Object.freeze({
  threshold: 0.20,
  padStartMs: null,
  padEndMs: null,
  minSilenceMs: null,
  crossfadeMs: null,
  gapMs: null,
});

const _WHISPER_EXTEND_MAX_FRAMES = 6; // 192ms max extension on soft whispered onset/tail

/**
 * Normalize VAD config supporting both camelCase and snake_case option keys.
 */
function normalizeConfig(config = {}) {
  const src = config ?? {};
  const pick = (camel, snake) =>
    src[camel] !== undefined ? src[camel] : src[snake] !== undefined ? src[snake] : null;
  return {
    threshold: src.threshold !== undefined && src.threshold !== null ? Number(src.threshold) : 0.20,
    padStartMs: pick("padStartMs", "pad_start_ms"),
    padEndMs: pick("padEndMs", "pad_end_ms"),
    minSilenceMs: pick("minSilenceMs", "min_silence_ms"),
    crossfadeMs: pick("crossfadeMs", "crossfade_ms"),
    gapMs: pick("gapMs", "gap_ms"),
  };
}

/**
 * Round a floating-point value to `decimals` decimal places without `-0`.
 */
export function roundTo(value, decimals) {
  const factor = 10 ** decimals;
  const rounded = Math.round(value * factor) / factor;
  return Object.is(rounded, -0) ? 0 : rounded;
}

/**
 * Compute linear-interpolation quantile matching `numpy.percentile`.
 */
export function percentile(values, pct) {
  const n = values.length;
  if (n === 0) {
    return 0.0;
  }
  const sorted = Float64Array.from(values);
  sorted.sort();
  if (n === 1) {
    return sorted[0];
  }
  const idx = (pct / 100.0) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/**
 * Compute duration-weighted step quantile matching `_weighted_percentile` in `vad.py`.
 */
export function weightedPercentile(values, weights, pct) {
  const n = values.length;
  if (n === 0) {
    return 0.0;
  }
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    order[i] = i;
  }
  order.sort((a, b) => values[a] - values[b]);

  const cumWeights = new Float64Array(n);
  let total = 0.0;
  for (let i = 0; i < n; i++) {
    total += weights[order[i]];
    cumWeights[i] = total;
  }
  if (total <= 0.0) {
    return percentile(values, pct);
  }

  const target = pct / 100.0;
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (cumWeights[mid] / total < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const idx = Math.min(Math.max(lo, 0), n - 1);
  return values[order[idx]];
}

/**
 * Apply a 2nd-order Butterworth 80Hz high-pass filter (matching FFmpeg `highpass=f=80`)
 * for Stage 2 browser-decoded 16kHz analysis audio without modifying the original stream.
 */
export function applyHighPass80Hz(audio16k, sampleRate = VAD_SAMPLE_RATE) {
  const n = audio16k.length;
  const out = new Float32Array(n);
  if (n === 0) {
    return out;
  }

  const w0 = (2.0 * Math.PI * 80.0) / sampleRate;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2.0 * Math.SQRT1_2);
  const a0 = 1.0 + alpha;
  const b0 = ((1.0 + cosW0) * 0.5) / a0;
  const b1 = -(1.0 + cosW0) / a0;
  const b2 = b0;
  const a1 = (-2.0 * cosW0) / a0;
  const a2 = (1.0 - alpha) / a0;

  let z1 = 0.0;
  let z2 = 0.0;
  for (let i = 0; i < n; i++) {
    const x = audio16k[i];
    const y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    out[i] = y;
  }
  return out;
}

/**
 * Compute mean DC offset in Float64 without allocating an extra copy of the full audio buffer.
 */
export function computeDcMean(audio16k) {
  const n = audio16k.length;
  if (n === 0) {
    return 0.0;
  }
  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    sum += audio16k[i];
  }
  return sum / n;
}

/**
 * Compute 32ms (512-sample) non-overlapping frame RMS levels in dBFS.
 */
export function computeFrameDbs(audio16k, dcMean = 0.0) {
  const nFrames = Math.floor(audio16k.length / FRAME_SAMPLES);
  if (nFrames === 0) {
    return new Float64Array(0);
  }
  const frameDbs = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const base = f * FRAME_SAMPLES;
    let sumSq = 0.0;
    for (let j = 0; j < FRAME_SAMPLES; j++) {
      const v = audio16k[base + j] - dcMean;
      sumSq += v * v;
    }
    frameDbs[f] = 20.0 * Math.log10(Math.sqrt(sumSq / FRAME_SAMPLES + 1e-12));
  }
  return frameDbs;
}

/**
 * Compute 32ms frame RMS dBFS values and overall segment linear RMS dBFS.
 */
export function segmentFrameDbs(audio16k, dcMean, startIdx, endIdx) {
  const s = Math.max(0, startIdx);
  const e = Math.min(audio16k.length, endIdx);
  const len = e - s;
  if (len <= 0) {
    return { frameDbs: new Float64Array([-120.0]), overallDb: -120.0 };
  }

  let totalSumSq = 0.0;
  for (let i = s; i < e; i++) {
    const v = audio16k[i] - dcMean;
    totalSumSq += v * v;
  }
  const overallDb = 20.0 * Math.log10(Math.sqrt(totalSumSq / len + 1e-12));

  const nFull = Math.floor(len / FRAME_SAMPLES);
  if (nFull === 0) {
    return { frameDbs: new Float64Array([overallDb]), overallDb };
  }

  const remLen = len - nFull * FRAME_SAMPLES;
  const hasRem = remLen >= (FRAME_SAMPLES >> 2);
  const frameDbs = new Float64Array(hasRem ? nFull + 1 : nFull);

  for (let f = 0; f < nFull; f++) {
    const base = s + f * FRAME_SAMPLES;
    let sumSq = 0.0;
    for (let j = 0; j < FRAME_SAMPLES; j++) {
      const v = audio16k[base + j] - dcMean;
      sumSq += v * v;
    }
    frameDbs[f] = 20.0 * Math.log10(Math.sqrt(sumSq / FRAME_SAMPLES + 1e-12));
  }

  if (hasRem) {
    const base = s + nFull * FRAME_SAMPLES;
    let sumSq = 0.0;
    for (let j = 0; j < remLen; j++) {
      const v = audio16k[base + j] - dcMean;
      sumSq += v * v;
    }
    frameDbs[nFull] = 20.0 * Math.log10(Math.sqrt(sumSq / remLen + 1e-12));
  }

  return { frameDbs, overallDb };
}

/**
 * In-place iterative Cooley-Tukey radix-2 FFT for real/imaginary Float64Arrays of power-of-2 length.
 */
function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const stepAngle = (-2.0 * Math.PI) / size;
    const wStepRe = Math.cos(stepAngle);
    const wStepIm = Math.sin(stepAngle);
    for (let start = 0; start < n; start += size) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let k = 0; k < half; k++) {
        const even = start + k;
        const odd = even + half;
        const tRe = wRe * re[odd] - wIm * im[odd];
        const tIm = wRe * im[odd] + wIm * re[odd];
        re[odd] = re[even] - tRe;
        im[odd] = im[even] - tIm;
        re[even] += tRe;
        im[even] += tIm;
        const nextRe = wRe * wStepRe - wIm * wStepIm;
        wIm = wRe * wStepIm + wIm * wStepRe;
        wRe = nextRe;
      }
    }
  }
}

/**
 * Return ratio of spectral energy >= 2.8 kHz relative to the 80–7800 Hz speech band.
 */
export function hfBreathRatio(audio16k, dcMean, startIdx, endIdx) {
  const s = Math.max(0, startIdx);
  const e = Math.min(audio16k.length, endIdx);
  const len = e - s;
  if (len < 64) {
    return 0.0;
  }

  let fftSize = 1;
  while (fftSize < len) {
    fftSize <<= 1;
  }

  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  const denom = len > 1 ? len - 1 : 1;
  for (let i = 0; i < len; i++) {
    const win = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / denom));
    re[i] = (audio16k[s + i] - dcMean) * win;
  }

  fftRadix2(re, im);

  const half = fftSize >> 1;
  const binHz = VAD_SAMPLE_RATE / fftSize;
  let total = 1e-12;
  let hf = 0.0;
  for (let k = 0; k <= half; k++) {
    const freq = k * binHz;
    if (freq >= 80.0 && freq <= 7800.0) {
      const magSq = re[k] * re[k] + im[k] * im[k];
      total += magSq;
      if (freq >= 2800.0) {
        hf += magSq;
      }
    }
  }
  return hf / total;
}

/**
 * Classify recording into 'studio' (low noise floor < -55 dBFS) vs 'vintage/live'.
 */
export function profileAudio(audio16k, frameDbs = null, dcMean = 0.0) {
  const fdbs = frameDbs ?? computeFrameDbs(audio16k, dcMean);
  if (fdbs.length === 0) {
    return {
      kind: "vintage/live",
      noise_floor_p5_db: -45.0,
      noise_floor_p10_db: -45.0,
      p90_db: -15.0,
      snr_db: 30.0,
    };
  }

  const nonZero = [];
  for (let i = 0; i < fdbs.length; i++) {
    if (fdbs[i] > -85.0) {
      nonZero.push(fdbs[i]);
    }
  }
  if (nonZero.length === 0) {
    return {
      kind: "studio",
      noise_floor_p5_db: -80.0,
      noise_floor_p10_db: -80.0,
      p90_db: -80.0,
      snr_db: 0.0,
    };
  }

  const nfP5 = percentile(nonZero, 5);
  const nfP10 = percentile(nonZero, 10);
  const p90 = percentile(nonZero, 90);
  const snr = Math.max(0.0, p90 - nfP10);
  const kind = nfP5 < -55.0 ? "studio" : "vintage/live";

  return {
    kind,
    noise_floor_p5_db: roundTo(nfP5, 2),
    noise_floor_p10_db: roundTo(nfP10, 2),
    p90_db: roundTo(p90, 2),
    snr_db: roundTo(snr, 2),
  };
}

/**
 * Sort and merge overlapping or closely spaced sample intervals [start, end].
 */
export function mergeIntervals(intervals, bridgeSamples = 0) {
  if (intervals.length === 0) {
    return [];
  }
  const ordered = intervals
    .map(([s, e]) => [s, e])
    .sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));

  const merged = [ordered[0]];
  for (let i = 1; i < ordered.length; i++) {
    const [sIdx, eIdx] = ordered[i];
    const prev = merged[merged.length - 1];
    if (sIdx <= prev[1] + bridgeSamples) {
      prev[1] = Math.max(prev[1], eIdx);
    } else {
      merged.push([sIdx, eIdx]);
    }
  }
  return merged;
}

/**
 * Apply asymmetric start/end padding and merge sub-crossfade or sub-threshold gaps.
 */
export function applyPaddingAndMerge(
  segments,
  padStartS,
  padEndS,
  minBound,
  maxBound,
  minSpliceGapS,
  minEffectiveCutS = 0.0
) {
  if (segments.length === 0) {
    return [];
  }

  const isArr = Array.isArray(padEndS);
  const padded = segments.map((seg, idx) => {
    const pEnd = isArr ? padEndS[idx] : Number(padEndS);
    return {
      start: Math.max(minBound, seg.start - padStartS),
      end: Math.min(maxBound, seg.end + pEnd),
    };
  });

  const merged = [padded[0]];
  for (let i = 1; i < padded.length; i++) {
    const seg = padded[i];
    const prev = merged[merged.length - 1];
    const gapBetween = seg.start - prev.end;
    if (
      gapBetween <= minSpliceGapS ||
      (minEffectiveCutS > 0.0 && gapBetween < minEffectiveCutS)
    ) {
      prev.end = Math.max(prev.end, seg.end);
    } else {
      merged.push(seg);
    }
  }
  return merged;
}

/**
 * Derive adaptive min_silence, gap, crossfade, padding, overlap, and room-tone flags.
 */
export function resolveProfileParams(profile, config = {}) {
  const cfg = normalizeConfig(config);
  const isStudio = profile.kind === "studio";
  const isHighAmbient =
    !isStudio && (profile.noise_floor_p5_db > -30.0 || profile.snr_db < 18.0);

  let autoMinSil;
  let autoGap;
  let autoXfade;
  let autoPadStart;
  let autoPadEnd;

  if (isStudio) {
    autoMinSil = 700;
    autoGap = 450;
    autoXfade = 20;
    autoPadStart = 200;
    autoPadEnd = 300;
  } else {
    autoMinSil = isHighAmbient ? 1250 : 900;
    autoGap = 0;
    autoXfade = 400;
    autoPadStart = 650;
    autoPadEnd = 1100;
  }

  const effMinSil = cfg.minSilenceMs !== null ? cfg.minSilenceMs : autoMinSil;
  const effGap = cfg.gapMs !== null ? cfg.gapMs : autoGap;
  const effXfade = cfg.crossfadeMs !== null ? cfg.crossfadeMs : autoXfade;
  const effPadStart = cfg.padStartMs !== null ? cfg.padStartMs : autoPadStart;
  const effPadEnd = cfg.padEndMs !== null ? cfg.padEndMs : autoPadEnd;
  const directOverlap = !isStudio && effGap === 0;
  const useRoomTone = !isStudio && effGap > 0 && profile.noise_floor_p5_db > -50.0;

  return {
    effMinSil,
    effGap,
    effXfade,
    effPadStart,
    effPadEnd,
    directOverlap,
    useRoomTone,
  };
}

/**
 * Detect melodic/sustained vocal islands via SNR-aware hysteresis RMS gate
 * plus whispered Quranic letter boundary preservation.
 */
export function detectEnergyIslands(audioLen, profile, islandBridgeMs, frameDbs) {
  const nFrames = frameDbs.length;
  if (nFrames === 0) {
    return [];
  }

  const spread = profile.snr_db;
  const isStudio = profile.kind === "studio";

  let onsetDb;
  let sustainDb;
  let minIslandDur;

  if (isStudio) {
    if (spread < 15.0) {
      return [];
    }
    onsetDb = Math.max(profile.noise_floor_p5_db + 16.0, profile.p90_db - 26.0);
    sustainDb = Math.max(profile.noise_floor_p5_db + 12.0, onsetDb - 4.0);
    minIslandDur = 0.40;
  } else {
    if (spread < 6.0) {
      return [];
    }
    if (spread >= 22.0) {
      onsetDb = Math.max(profile.noise_floor_p10_db + 15.0, profile.p90_db - 25.0);
      sustainDb = onsetDb - 4.0;
    } else {
      onsetDb = Math.max(profile.noise_floor_p10_db + 0.52 * spread, profile.p90_db - 25.0);
      sustainDb = onsetDb - Math.max(2.5, Math.min(3.5, 0.18 * spread));
    }
    minIslandDur = 0.35;
  }

  const whisperFloorDb = isStudio
    ? Math.min(sustainDb, profile.noise_floor_p5_db + 8.0)
    : Math.min(sustainDb, profile.noise_floor_p10_db + Math.max(4.5, Math.min(8.0, spread * 0.25)));

  const confirmedFrameIslands = [];
  let inIsland = false;
  let startF = 0;

  for (let idx = 0; idx < nFrames; idx++) {
    const valDb = frameDbs[idx];
    if (!inIsland && valDb >= onsetDb) {
      startF = idx;
      inIsland = true;
    } else if (inIsland && valDb < sustainDb) {
      const durS = ((idx - startF) * FRAME_SAMPLES) / VAD_SAMPLE_RATE;
      let peakDb = -Infinity;
      for (let k = startF; k < idx; k++) {
        if (frameDbs[k] > peakDb) {
          peakDb = frameDbs[k];
        }
      }
      if (
        durS >= minIslandDur ||
        (!isStudio && peakDb >= onsetDb + 4.0) ||
        (isStudio && durS >= STUDIO_MIN_SPEECH_MS / 1000.0 && peakDb >= onsetDb + 4.0)
      ) {
        confirmedFrameIslands.push([startF, idx, false]);
      }
      inIsland = false;
    }
  }

  if (inIsland) {
    const durS = ((nFrames - startF) * FRAME_SAMPLES) / VAD_SAMPLE_RATE;
    let peakDb = -Infinity;
    for (let k = startF; k < nFrames; k++) {
      if (frameDbs[k] > peakDb) {
        peakDb = frameDbs[k];
      }
    }
    if (
      durS >= minIslandDur ||
      (!isStudio && peakDb >= onsetDb + 4.0) ||
      (isStudio && durS >= STUDIO_MIN_SPEECH_MS / 1000.0 && peakDb >= onsetDb + 4.0)
    ) {
      confirmedFrameIslands.push([startF, nFrames, true]);
    }
  }

  // Extend confirmed island boundaries across adjacent multi-frame (>=64ms) whispered Quranic consonants.
  const rawIslands = confirmedFrameIslands.map(([sF, eF, reachedEnd]) => {
    let stepsBack = 0;
    while (
      sF - stepsBack > 0 &&
      stepsBack < _WHISPER_EXTEND_MAX_FRAMES &&
      frameDbs[sF - 1 - stepsBack] >= whisperFloorDb
    ) {
      stepsBack++;
    }
    const s = stepsBack >= 2 ? sF - stepsBack : sF;

    let stepsFwd = 0;
    while (
      eF + stepsFwd < nFrames &&
      stepsFwd < _WHISPER_EXTEND_MAX_FRAMES &&
      frameDbs[eF + stepsFwd] >= whisperFloorDb
    ) {
      stepsFwd++;
    }
    const e = stepsFwd >= 2 ? eF + stepsFwd : eF;

    const sSample = s * FRAME_SAMPLES;
    const eSample =
      reachedEnd || e >= nFrames
        ? Math.min(audioLen, e === nFrames && reachedEnd ? audioLen : e * FRAME_SAMPLES)
        : e * FRAME_SAMPLES;
    return [sSample, eSample];
  });

  const bridgeSamples = Math.round((islandBridgeMs / 1000.0) * VAD_SAMPLE_RATE);
  const bridged = mergeIntervals(rawIslands, bridgeSamples);
  const minDurMs = isStudio ? STUDIO_MIN_SPEECH_MS : MIN_SPEECH_MS;
  const minSamples = Math.round((minDurMs / 1000.0) * VAD_SAMPLE_RATE);

  return bridged.filter(([s, e]) => e - s >= minSamples);
}

/**
 * Reject short close-mic inhalation breaths on clean studio recordings.
 */
export function filterStudioBreaths(audio16k, dcMean, intervals) {
  if (intervals.length === 0) {
    return [];
  }

  const minSamples = Math.round((STUDIO_MIN_SPEECH_MS / 1000.0) * VAD_SAMPLE_RATE);
  const stats = [];
  for (const [sIdx, eIdx] of intervals) {
    if (eIdx - sIdx < minSamples) {
      continue;
    }
    const { overallDb } = segmentFrameDbs(audio16k, dcMean, sIdx, eIdx);
    stats.push({ sIdx, eIdx, rmsDb: overallDb });
  }
  if (stats.length === 0) {
    return [];
  }

  const refDb = percentile(
    stats.map((x) => x.rmsDb),
    75
  );
  const kept = [];
  for (const { sIdx, eIdx, rmsDb } of stats) {
    const durS = (eIdx - sIdx) / VAD_SAMPLE_RATE;
    let isBreath = false;
    if (durS < 0.90 && rmsDb < refDb - 10.0) {
      const hfRatio = hfBreathRatio(audio16k, dcMean, sIdx, eIdx);
      isBreath = hfRatio >= 0.22;
    }
    if (!isBreath) {
      kept.push([sIdx, eIdx]);
    }
  }
  return kept;
}

/**
 * Pure-DSP vintage unpitched high-frequency noise blip guard (conservative fallback for
 * unconfirmed cough/rustle bursts without risking whispered speech).
 */
export function filterUnpitchedNoiseBlips(audio16k, dcMean, intervals) {
  if (intervals.length < 3) {
    return intervals;
  }

  const stats = intervals.map(([sIdx, eIdx]) => {
    const { frameDbs, overallDb } = segmentFrameDbs(audio16k, dcMean, sIdx, eIdx);
    let peakDb = -Infinity;
    for (let i = 0; i < frameDbs.length; i++) {
      if (frameDbs[i] > peakDb) {
        peakDb = frameDbs[i];
      }
    }
    const durS = Math.max(1, eIdx - sIdx) / VAD_SAMPLE_RATE;
    return { sIdx, eIdx, overallDb, peakDb, durS };
  });

  const medDb = weightedPercentile(
    stats.map((x) => x.overallDb),
    stats.map((x) => x.durS),
    50.0
  );

  const kept = [];
  for (const { sIdx, eIdx, overallDb, peakDb, durS } of stats) {
    let isNoiseBlip = false;
    if (durS < 1.0 && overallDb < medDb - 11.5 && peakDb < medDb - 6.0) {
      const hfRatio = hfBreathRatio(audio16k, dcMean, sIdx, eIdx);
      isNoiseBlip = hfRatio >= 0.45;
    }
    if (!isNoiseBlip) {
      kept.push([sIdx, eIdx]);
    }
  }
  return kept;
}

/**
 * Strip short isolated crowd bursts/clusters (<= 3.5s) >= 10..14 dB below reciter core.
 */
export function filterCrowdMurmurClusters(segsInfo, snrDb, clipStartS, clipEndS) {
  const n = segsInfo.length;
  if (n < 3) {
    return { kept: segsInfo, strippedCount: 0 };
  }

  const coreVals = segsInfo.map((x) => x.coreDb);
  const durs = segsInfo.map((x) => Math.max(0.05, x.end - x.start));
  const reciterCoreMed = weightedPercentile(coreVals, durs, 50.0);
  const marginDb = Math.max(10.0, Math.min(14.0, snrDb * 0.40));
  const cutoffDb = reciterCoreMed - marginDb;

  const stripIndices = new Set();
  let idx = 0;
  while (idx < n) {
    if (segsInfo[idx].coreDb > cutoffDb) {
      idx++;
      continue;
    }

    let j = idx;
    while (
      j + 1 < n &&
      segsInfo[j + 1].coreDb <= cutoffDb &&
      segsInfo[j + 1].start - segsInfo[j].end < 1.8
    ) {
      j++;
    }

    const clusterSpan = segsInfo[j].end - segsInfo[idx].start;
    const gapBefore =
      idx === 0 ? segsInfo[idx].start - clipStartS : segsInfo[idx].start - segsInfo[idx - 1].end;
    const gapAfter =
      j === n - 1 ? clipEndS - segsInfo[j].end : segsInfo[j + 1].start - segsInfo[j].end;

    if (clusterSpan <= 3.5 && gapBefore >= 1.5 && gapAfter >= 1.8) {
      let totalFrames = 0;
      for (let k = idx; k <= j; k++) {
        totalFrames += segsInfo[k].frameDbs.length;
      }
      const combinedFdbs = new Float64Array(totalFrames);
      let off = 0;
      for (let k = idx; k <= j; k++) {
        combinedFdbs.set(segsInfo[k].frameDbs, off);
        off += segsInfo[k].frameDbs.length;
      }
      const clusterCoreDb = percentile(combinedFdbs, 75);
      if (clusterCoreDb <= cutoffDb) {
        for (let k = idx; k <= j; k++) {
          stripIndices.add(k);
        }
      }
    }

    idx = j + 1;
  }

  if (stripIndices.size >= n) {
    return { kept: segsInfo, strippedCount: 0 };
  }

  const kept = segsInfo.filter((_, k) => !stripIndices.has(k));
  return { kept, strippedCount: stripIndices.size };
}

/**
 * Extend pad_end to 1300ms on loud vintage/live endings (tail core_db >= reciter p75).
 */
export function computeSegmentPadEndsS(
  audio16k,
  dcMean,
  segments,
  profile,
  config,
  basePadEndMs,
  clipStartS
) {
  const cfg = normalizeConfig(config);
  const baseS = basePadEndMs / 1000.0;
  if (cfg.padEndMs !== null || profile.kind === "studio" || segments.length === 0) {
    return new Array(segments.length).fill(baseS);
  }

  const tailWindowSamples = Math.round(0.80 * VAD_SAMPLE_RATE);
  const segCores = new Array(segments.length);
  const segDurs = new Array(segments.length);
  const tailCores = new Array(segments.length);

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const sIdx = Math.max(
      0,
      Math.min(audio16k.length, Math.round((seg.start - clipStartS) * VAD_SAMPLE_RATE))
    );
    const eIdx = Math.max(
      sIdx,
      Math.min(audio16k.length, Math.round((seg.end - clipStartS) * VAD_SAMPLE_RATE))
    );
    const { frameDbs: segFdbs } = segmentFrameDbs(audio16k, dcMean, sIdx, eIdx);
    segCores[i] = percentile(segFdbs, 75);
    segDurs[i] = Math.max(0.05, seg.end - seg.start);

    const tailS = Math.max(sIdx, eIdx - tailWindowSamples);
    const { frameDbs: tailFdbs } = segmentFrameDbs(audio16k, dcMean, tailS, eIdx);
    tailCores[i] = percentile(tailFdbs, 75);
  }

  const reciterP75 = weightedPercentile(segCores, segDurs, 75.0);
  const loudS = Math.max(baseS, 1.300);
  return tailCores.map((tc) => (tc >= reciterP75 ? loudS : baseS));
}

/**
 * Sample a clean ambient loop window from the recording's quietest floor frames (p5–p15).
 */
export function findQuietestGapWindow(
  audio16k,
  dcMean,
  keptSegments,
  clipStartS,
  clipEndS,
  targetDurationS = 0.6,
  profile = null,
  frameDbs = null
) {
  const clipDur = clipEndS - clipStartS;
  if (keptSegments.length === 0 || clipDur <= 0.3) {
    return null;
  }

  const gaps = [];
  let cursor = clipStartS;
  for (const seg of keptSegments) {
    if (seg.start - cursor >= 0.25) {
      gaps.push([cursor, seg.start]);
    }
    cursor = Math.max(cursor, seg.end);
  }
  if (clipEndS - cursor >= 0.25) {
    gaps.push([cursor, clipEndS]);
  }

  let floorLo = null;
  let floorHi = null;
  let fdbs = frameDbs;
  if (profile !== null) {
    if (fdbs === null) {
      fdbs = computeFrameDbs(audio16k, dcMean);
    }
    const nonZero = [];
    for (let i = 0; i < fdbs.length; i++) {
      if (fdbs[i] > -85.0) {
        nonZero.push(fdbs[i]);
      }
    }
    const p15 = nonZero.length > 0 ? percentile(nonZero, 15) : profile.noise_floor_p10_db;
    floorLo = profile.noise_floor_p5_db - 3.0;
    floorHi = Math.max(profile.noise_floor_p10_db + 3.0, p15 + 2.0);
  }

  const winSamples = Math.max(
    Math.trunc(0.25 * VAD_SAMPLE_RATE),
    Math.round(targetDurationS * VAD_SAMPLE_RATE)
  );
  let bestAny = null;
  let bestFloor = null;

  const consider = (cand) => {
    if (bestAny === null || cand[2] < bestAny[2]) {
      bestAny = cand;
    }
    if (floorLo !== null && floorHi !== null && cand[2] >= floorLo && cand[2] <= floorHi) {
      if (bestFloor === null || cand[2] < bestFloor[2]) {
        bestFloor = cand;
      }
    }
  };

  for (const [gStart, gEnd] of gaps) {
    const innerS = gEnd - gStart > 0.5 ? gStart + 0.10 : gStart;
    const innerE = gEnd - gStart > 0.5 ? gEnd - 0.10 : gEnd;
    const sIdx = Math.max(0, Math.round((innerS - clipStartS) * VAD_SAMPLE_RATE));
    const eIdx = Math.min(audio16k.length, Math.round((innerE - clipStartS) * VAD_SAMPLE_RATE));
    const spanLen = eIdx - sIdx;
    if (spanLen < Math.trunc(0.2 * VAD_SAMPLE_RATE)) {
      continue;
    }

    if (spanLen <= winSamples) {
      let sumSq = 0.0;
      for (let i = sIdx; i < eIdx; i++) {
        const v = audio16k[i] - dcMean;
        sumSq += v * v;
      }
      const rmsDb = 20.0 * Math.log10(Math.sqrt(sumSq / spanLen + 1e-12));
      consider([
        clipStartS + sIdx / VAD_SAMPLE_RATE,
        clipStartS + eIdx / VAD_SAMPLE_RATE,
        rmsDb,
      ]);
      continue;
    }

    const step = Math.max(1600, Math.floor(winSamples / 4));
    for (let off = 0; off <= spanLen - winSamples; off += step) {
      let sumSq = 0.0;
      const base = sIdx + off;
      for (let i = 0; i < winSamples; i++) {
        const v = audio16k[base + i] - dcMean;
        sumSq += v * v;
      }
      const rmsDb = 20.0 * Math.log10(Math.sqrt(sumSq / winSamples + 1e-12));
      consider([
        clipStartS + base / VAD_SAMPLE_RATE,
        clipStartS + (base + winSamples) / VAD_SAMPLE_RATE,
        rmsDb,
      ]);
    }
  }

  if (bestFloor === null && floorLo !== null && floorHi !== null) {
    if (fdbs === null) {
      fdbs = computeFrameDbs(audio16k, dcMean);
    }
    const winFrames = Math.max(8, Math.round(winSamples / FRAME_SAMPLES));
    if (fdbs.length >= winFrames) {
      const stepF = Math.max(4, Math.floor(winFrames / 2));
      for (let f0 = 0; f0 <= fdbs.length - winFrames; f0 += stepF) {
        let sumDb = 0.0;
        let maxDb = -Infinity;
        for (let k = 0; k < winFrames; k++) {
          const val = fdbs[f0 + k];
          sumDb += val;
          if (val > maxDb) {
            maxDb = val;
          }
        }
        const meanDb = sumDb / winFrames;
        if (meanDb >= floorLo && meanDb <= floorHi && maxDb <= floorHi + 3.0) {
          const wStart = clipStartS + (f0 * FRAME_SAMPLES) / VAD_SAMPLE_RATE;
          const wEnd = clipStartS + ((f0 + winFrames) * FRAME_SAMPLES) / VAD_SAMPLE_RATE;
          if (bestFloor === null || meanDb < bestFloor[2]) {
            bestFloor = [wStart, wEnd, meanDb];
          }
        }
      }
    }
  }

  return bestFloor || bestAny;
}

/**
 * Detect vocal segments via auto-profiled Pure Energy DSP pipeline (exact port of `vad.py`).
 *
 * @param {Float32Array|Float64Array} audio16k - 16 kHz mono audio signal.
 * @param {object} [config] - Optional VADConfig overrides.
 * @param {object} [options] - `{ exactDuration, clipStartS, progressCb }`.
 */
export function detectSpeechSegments(audio16k, config = {}, options = {}) {
  const cfg = normalizeConfig(config);
  const exactDuration =
    options.exactDuration !== undefined ? options.exactDuration : null;
  const clipStartS = options.clipStartS !== undefined ? Number(options.clipStartS) : 0.0;
  const progressCb = typeof options.progressCb === "function" ? options.progressCb : null;

  const vadDuration = audio16k.length / VAD_SAMPLE_RATE;
  const clipDur = exactDuration !== null ? Number(exactDuration) : vadDuration;
  const clipEndS = clipStartS + clipDur;

  const dcMean = computeDcMean(audio16k);
  const frameDbs = computeFrameDbs(audio16k, dcMean);
  let profile = profileAudio(audio16k, frameDbs, dcMean);

  if (profile.snr_db < LOW_SNR_GUARD_DB) {
    if (progressCb) {
      progressCb(1.0);
    }
    const { frameDbs: segFdbs, overallDb } = segmentFrameDbs(
      audio16k,
      dcMean,
      0,
      audio16k.length
    );
    const coreDb = percentile(segFdbs, 75);
    const segStart = roundTo(clipStartS, 4);
    const segEnd = roundTo(clipEndS, 4);
    return {
      segments: [
        {
          start: segStart,
          end: segEnd,
          duration: roundTo(Math.max(0.0, segEnd - segStart), 4),
          rms_db: roundTo(overallDb, 2),
          core_db: roundTo(coreDb, 2),
        },
      ],
      total_duration: clipDur,
      profile,
      raw_segments_count: 1,
      filtered_segments_count: 1,
      quietest_gap_window: null,
      resolved_pad_start_ms: 0,
      resolved_pad_end_ms: 0,
      resolved_min_silence_ms: 0,
      resolved_crossfade_ms: 0,
      resolved_gap_ms: 0,
      use_room_tone: false,
      direct_overlap: false,
      low_snr_guarded: true,
      stray_shouts_stripped: 0,
    };
  }

  let resolved = resolveProfileParams(profile, cfg);
  let energyIntervals = detectEnergyIslands(
    audio16k.length,
    profile,
    resolved.effMinSil,
    frameDbs
  );
  if (progressCb) {
    progressCb(0.65);
  }

  // Dense studio recitations with short inter-verse breaths (<700ms) push frame p5
  // into vocal tails; inspect low-tail p0.2 when detected speech covers >=90% of the audio.
  if (profile.kind !== "studio" && audio16k.length > 0) {
    let coveredSamples = 0;
    for (const [s, e] of energyIntervals) {
      coveredSamples += e - s;
    }
    const cov = coveredSamples / audio16k.length;
    const nonZero = [];
    for (let i = 0; i < frameDbs.length; i++) {
      if (frameDbs[i] > -85.0) {
        nonZero.push(frameDbs[i]);
      }
    }
    if (cov >= 0.90 && nonZero.length > 0 && percentile(nonZero, 0.2) <= -50.0) {
      profile = {
        ...profile,
        kind: "studio",
      };
      resolved = resolveProfileParams(profile, cfg);
      if (cfg.minSilenceMs === null) {
        energyIntervals = detectEnergyIslands(
          audio16k.length,
          profile,
          resolved.effMinSil,
          frameDbs
        );
      }
    }
  }

  const bridgeSamples = Math.round((resolved.effMinSil / 1000.0) * VAD_SAMPLE_RATE);
  let rawSamples = mergeIntervals(energyIntervals, bridgeSamples);

  if (profile.kind === "studio") {
    rawSamples = filterStudioBreaths(audio16k, dcMean, rawSamples);
  } else {
    rawSamples = filterUnpitchedNoiseBlips(audio16k, dcMean, rawSamples);
  }

  if (rawSamples.length === 0) {
    if (progressCb) {
      progressCb(1.0);
    }
    return {
      segments: [],
      total_duration: clipDur,
      profile,
      raw_segments_count: 0,
      filtered_segments_count: 0,
      quietest_gap_window: null,
      resolved_pad_start_ms: resolved.effPadStart,
      resolved_pad_end_ms: resolved.effPadEnd,
      resolved_min_silence_ms: resolved.effMinSil,
      resolved_crossfade_ms: resolved.effXfade,
      resolved_gap_ms: resolved.effGap,
      use_room_tone: resolved.useRoomTone,
      direct_overlap: resolved.directOverlap,
      low_snr_guarded: false,
      stray_shouts_stripped: 0,
    };
  }

  const rawCount = rawSamples.length;
  let strayStrippedCount = 0;

  let segsRaw = rawSamples.map(([sIdx, eIdx]) => {
    const { frameDbs: segFdbs } = segmentFrameDbs(audio16k, dcMean, sIdx, eIdx);
    const coreDb = percentile(segFdbs, 75);
    return {
      start: clipStartS + sIdx / VAD_SAMPLE_RATE,
      end: clipStartS + eIdx / VAD_SAMPLE_RATE,
      coreDb,
      frameDbs: segFdbs,
    };
  });

  if (profile.kind === "vintage/live") {
    const filtered = filterCrowdMurmurClusters(
      segsRaw,
      profile.snr_db,
      clipStartS,
      clipEndS
    );
    segsRaw = filtered.kept;
    strayStrippedCount = filtered.strippedCount;
  }

  const bridged = segsRaw.map((x) => ({ start: x.start, end: x.end }));
  const filteredCount = bridged.length;
  if (progressCb) {
    progressCb(1.0);
  }

  const padEndsS = computeSegmentPadEndsS(
    audio16k,
    dcMean,
    bridged,
    profile,
    cfg,
    resolved.effPadEnd,
    clipStartS
  );
  const minEffectiveCutS =
    profile.kind === "studio" ? 0.0 : MIN_VINTAGE_EFFECTIVE_CUT_S;
  const padded = applyPaddingAndMerge(
    bridged,
    resolved.effPadStart / 1000.0,
    padEndsS,
    clipStartS,
    clipEndS,
    resolved.effXfade / 1000.0,
    minEffectiveCutS
  );

  const finalSegments = padded.map((seg) => {
    const sIdx = Math.max(
      0,
      Math.min(audio16k.length, Math.round((seg.start - clipStartS) * VAD_SAMPLE_RATE))
    );
    const eIdx = Math.max(
      sIdx,
      Math.min(audio16k.length, Math.round((seg.end - clipStartS) * VAD_SAMPLE_RATE))
    );
    const { frameDbs: segFdbs, overallDb } = segmentFrameDbs(
      audio16k,
      dcMean,
      sIdx,
      eIdx
    );
    const coreDb = percentile(segFdbs, 75);
    const sRound = roundTo(seg.start, 4);
    const eRound = roundTo(seg.end, 4);
    return {
      start: sRound,
      end: eRound,
      duration: roundTo(Math.max(0.0, eRound - sRound), 4),
      rms_db: roundTo(overallDb, 2),
      core_db: roundTo(coreDb, 2),
    };
  });

  const quietGap = resolved.useRoomTone
    ? findQuietestGapWindow(
        audio16k,
        dcMean,
        finalSegments,
        clipStartS,
        clipEndS,
        Math.max(0.5, (resolved.effGap + resolved.effXfade) / 1000.0),
        profile,
        frameDbs
      )
    : null;

  return {
    segments: finalSegments,
    total_duration: clipDur,
    profile,
    raw_segments_count: rawCount,
    filtered_segments_count: filteredCount,
    quietest_gap_window: quietGap,
    resolved_pad_start_ms: resolved.effPadStart,
    resolved_pad_end_ms: resolved.effPadEnd,
    resolved_min_silence_ms: resolved.effMinSil,
    resolved_crossfade_ms: resolved.effXfade,
    resolved_gap_ms: resolved.effGap,
    use_room_tone: resolved.useRoomTone,
    direct_overlap: resolved.directOverlap,
    low_snr_guarded: false,
    stray_shouts_stripped: strayStrippedCount,
  };
}
