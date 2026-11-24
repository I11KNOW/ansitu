/**
 * Dedicated Module Web Worker for ansitu.
 *
 * Pipeline inside the worker (zero UI blocking, instant abort via `worker.terminate()`):
 *   1. Load `ffmpeg.wasm` (`@ffmpeg/core-mt` when `self.crossOriginIsolated`, else `@ffmpeg/core`).
 *   2. Decode input media to native PCM-16 WAV via FFmpeg VFS, parse into `Int16Array`.
 *   3. Downmix & resample to 16 kHz mono + 80 Hz Butterworth high-pass (`applyHighPass80Hz`).
 *   4. Run `detectSpeechSegments` (`./engine/vad.js`) across user ranges (or full file).
 *      If `low_snr_guarded`, stop before splicing/re-encoding and report immediately.
 *   5. Splice retained segments in `Int16Array` via `spliceRangesInt16` (`./engine/splicer.js`).
 *   6. Encode WAV/PCM output (lossless) or transcode via `ffmpeg.wasm` to M4A/MP3 and transfer buffer.
 */

import {
  FFMPEG_CORE_MT_BASE,
  FFMPEG_CORE_ST_BASE,
} from "./config.js";
import {
  VAD_SAMPLE_RATE,
  applyHighPass80Hz,
  detectSpeechSegments,
} from "./engine/vad.js";
import {
  EDGE_FADE_IN_MS,
  EDGE_FADE_OUT_MS,
  INTER_RANGE_MICRO_FADE_MS,
  MIME_TYPES,
  QUALITY_PRESETS,
  applyPrayerModeFallback,
  encodeWavInt16,
  isLosslessSource,
  resolveEffectiveRanges,
  spliceRangesInt16,
} from "./engine/splicer.js";

let ffmpegInstance = null;
let loadedMode = null;

async function toBlobURL(url, mimeType) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  const buf = await response.arrayBuffer();
  return URL.createObjectURL(new Blob([buf], { type: mimeType }));
}

async function loadCoreFromBase(baseUrl, multiThreaded) {
  const fetchTasks = [
    toBlobURL(`${baseUrl}/ffmpeg-core.js`, "text/javascript"),
    toBlobURL(`${baseUrl}/ffmpeg-core.wasm`, "application/wasm"),
  ];
  if (multiThreaded) {
    fetchTasks.push(
      toBlobURL(`${baseUrl}/ffmpeg-core.worker.js`, "text/javascript")
    );
  }
  const [coreURL, wasmURL, workerURL] = await Promise.all(fetchTasks);

  const mod = await import(/* @vite-ignore */ coreURL);
  const createFFmpegCore = mod.default;
  if (typeof createFFmpegCore !== "function") {
    throw new Error("Invalid ffmpeg-core module export.");
  }

  const hashPayload = workerURL ? { wasmURL, workerURL } : { wasmURL };
  const mainScriptUrlOrBlob = `${coreURL}#${btoa(JSON.stringify(hashPayload))}`;

  return createFFmpegCore({ mainScriptUrlOrBlob });
}

async function ensureFFmpegLoaded() {
  if (ffmpegInstance) {
    return ffmpegInstance;
  }

  const canUseMT = Boolean(self.crossOriginIsolated);
  if (canUseMT) {
    try {
      ffmpegInstance = await loadCoreFromBase(FFMPEG_CORE_MT_BASE, true);
      loadedMode = "mt";
      return ffmpegInstance;
    } catch {
      // Fall back cleanly to single-threaded core if pthreads fail to initialize
    }
  }

  ffmpegInstance = await loadCoreFromBase(FFMPEG_CORE_ST_BASE, false);
  loadedMode = "st";
  return ffmpegInstance;
}

function execFFmpeg(core, args, onProgress = null) {
  const logs = [];
  core.setLogger(({ message }) => {
    if (message) {
      logs.push(message);
      if (logs.length > 40) {
        logs.shift();
      }
    }
  });
  core.setProgress(({ progress }) => {
    if (onProgress && Number.isFinite(progress)) {
      onProgress(Math.max(0, Math.min(1, progress)));
    }
  });
  core.setTimeout(-1);
  core.exec(...args);
  const ret = core.ret;
  core.reset();
  if (ret !== 0) {
    const tail = logs.slice(-8).join("\n");
    throw new Error(`FFmpeg exited with code ${ret}${tail ? `: ${tail}` : ""}`);
  }
}

function deleteFile(core, filePath) {
  try {
    core.FS.unlink(filePath);
  } catch {
    // Ignore if file already unlinked
  }
}

const _ALLOWED_META_KEYS = new Set([
  "title",
  "artist",
  "album",
  "album_artist",
  "date",
  "genre",
  "comment",
]);

/**
 * Keep only clean text metadata tags (skipping `encoder`, stale `TLEN`, and `[CHAPTER]` sections).
 */
function filterSafeFfmetadata(rawText) {
  const lines = String(rawText || "").split(/\r?\n/);
  const kept = [];
  let keepingMultiline = false;
  for (const line of lines) {
    if (!keepingMultiline) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(";")) {
        continue;
      }
      if (trimmed.startsWith("[")) {
        break;
      }
      const eqIdx = line.indexOf("=");
      if (eqIdx <= 0) {
        continue;
      }
      const key = line.slice(0, eqIdx).trim().toLowerCase();
      if (_ALLOWED_META_KEYS.has(key)) {
        kept.push(line);
        keepingMultiline = line.endsWith("\\");
      }
    } else {
      kept.push(line);
      keepingMultiline = line.endsWith("\\");
    }
  }
  return kept.length > 0 ? `;FFMETADATA1\n${kept.join("\n")}\n` : null;
}

/**
 * Parse a standard PCM-16 RIFF/WAVE file into interleaved `Int16Array` without copying
 * unless the `data` chunk is odd-byte aligned.
 */
function parseWavInt16(wavBytes) {
  const view = new DataView(
    wavBytes.buffer,
    wavBytes.byteOffset,
    wavBytes.byteLength
  );
  const ascii = (offset, len) => {
    let s = "";
    for (let i = 0; i < len; i++) {
      s += String.fromCharCode(view.getUint8(offset + i));
    }
    return s;
  };

  if (wavBytes.byteLength < 44 || ascii(0, 4) !== "RIFF" || ascii(8, 4) !== "WAVE") {
    throw new Error("DECODE_FAILED");
  }

  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;

  let offset = 12;
  while (offset + 8 <= wavBytes.byteLength) {
    const chunkId = ascii(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const payloadStart = offset + 8;

    if (chunkId === "fmt " && chunkSize >= 16 && payloadStart + 16 <= wavBytes.byteLength) {
      channels = view.getUint16(payloadStart + 2, true);
      sampleRate = view.getUint32(payloadStart + 4, true);
      bitsPerSample = view.getUint16(payloadStart + 14, true);
    } else if (chunkId === "data") {
      dataOffset = payloadStart;
      dataSize = Math.min(chunkSize, wavBytes.byteLength - payloadStart);
      break;
    }

    // RIFF chunks are padded to even byte boundaries
    offset = payloadStart + chunkSize + (chunkSize & 1);
  }

  if (!sampleRate || !channels || bitsPerSample !== 16 || dataOffset < 0 || dataSize <= 0) {
    throw new Error("DECODE_FAILED");
  }

  const sampleCount = Math.floor(dataSize / 2);
  const byteStart = wavBytes.byteOffset + dataOffset;
  let pcm;
  if (byteStart % 2 === 0) {
    pcm = new Int16Array(wavBytes.buffer, byteStart, sampleCount);
  } else {
    const aligned = new Uint8Array(sampleCount * 2);
    aligned.set(wavBytes.subarray(dataOffset, dataOffset + sampleCount * 2));
    pcm = new Int16Array(aligned.buffer);
  }

  const frames = Math.floor(sampleCount / channels);
  return { pcm, sampleRate, channels, frames };
}

/**
 * Downmix interleaved native-rate `Int16Array` to 16 kHz mono `Float32Array`
 * and apply the 80 Hz high-pass filter matching Python's `extract_vad_audio`.
 */
function buildVadAudio16k(pcm, sampleRate, channels, totalFrames, gainComp = 1.0) {
  const targetLen = Math.max(
    1,
    Math.round((totalFrames * VAD_SAMPLE_RATE) / sampleRate)
  );
  const mono16k = new Float32Array(targetLen);
  const invScale = gainComp / (32768.0 * channels);

  if (sampleRate === VAD_SAMPLE_RATE) {
    for (let f = 0; f < targetLen && f < totalFrames; f++) {
      let sum = 0;
      const base = f * channels;
      for (let ch = 0; ch < channels; ch++) {
        sum += pcm[base + ch];
      }
      mono16k[f] = sum * invScale;
    }
  } else if (sampleRate > VAD_SAMPLE_RATE) {
    const ratio = sampleRate / VAD_SAMPLE_RATE;
    for (let i = 0; i < targetLen; i++) {
      const fStart = Math.floor(i * ratio);
      const fEnd = Math.min(totalFrames, Math.max(fStart + 1, Math.ceil((i + 1) * ratio)));
      let acc = 0.0;
      for (let f = fStart; f < fEnd; f++) {
        const base = f * channels;
        for (let ch = 0; ch < channels; ch++) {
          acc += pcm[base + ch];
        }
      }
      mono16k[i] = (acc / (fEnd - fStart)) * invScale;
    }
  } else {
    const ratio = sampleRate / VAD_SAMPLE_RATE;
    for (let i = 0; i < targetLen; i++) {
      const srcPos = i * ratio;
      const f0 = Math.min(totalFrames - 1, Math.floor(srcPos));
      const f1 = Math.min(totalFrames - 1, f0 + 1);
      const frac = srcPos - f0;
      let s0 = 0;
      let s1 = 0;
      const b0 = f0 * channels;
      const b1 = f1 * channels;
      for (let ch = 0; ch < channels; ch++) {
        s0 += pcm[b0 + ch];
        s1 += pcm[b1 + ch];
      }
      mono16k[i] = ((s0 * (1.0 - frac) + s1 * frac) * invScale);
    }
  }

  return applyHighPass80Hz(mono16k, VAD_SAMPLE_RATE);
}

function postProgress(stage, pct) {
  self.postMessage({
    type: "progress",
    stage,
    pct: Math.max(0, Math.min(100, Math.round(pct))),
    coreMode: loadedMode,
  });
}

function getFileExtension(name) {
  const m = String(name || "").toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : "mp3";
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || msg.type !== "process") {
    return;
  }

  const {
    fileName,
    outFormat = "m4a",
    quality = "192",
    ranges = [],
    mode = "standard",
    rangeAction = "keep",
  } = msg;

  let core = null;
  let inFile = null;

  try {
    postProgress("loading", 3);
    core = await ensureFFmpegLoaded();

    postProgress("decoding", 10);
    const inExt = getFileExtension(fileName);
    inFile = `input.${inExt}`;
    let inputBytes = new Uint8Array(msg.fileBuffer);
    core.FS.writeFile(inFile, inputBytes);
    inputBytes = null;
    msg.fileBuffer = null;

    // Apply -6 dB decode headroom for lossy inputs so hot (>0 dBFS) masters never clip in Int16Array.
    const useHeadroom = !isLosslessSource(fileName) && quality !== "lossless_wav";

    let hasMetadata = false;
    try {
      execFFmpeg(core, ["-y", "-i", inFile, "-f", "ffmetadata", "meta_raw.txt"]);
      const rawMeta = new TextDecoder("utf-8").decode(core.FS.readFile("meta_raw.txt"));
      deleteFile(core, "meta_raw.txt");
      const filteredMeta = filterSafeFfmetadata(rawMeta);
      if (filteredMeta) {
        core.FS.writeFile("meta.txt", new TextEncoder().encode(filteredMeta));
        hasMetadata = true;
      }
    } catch {
      deleteFile(core, "meta_raw.txt");
      hasMetadata = false;
    }

    const decodeArgs = useHeadroom
      ? ["-y", "-i", inFile, "-vn", "-af", "volume=0.5", "-c:a", "pcm_s16le", "-f", "wav", "native.wav"]
      : ["-y", "-i", inFile, "-vn", "-c:a", "pcm_s16le", "-f", "wav", "native.wav"];

    execFFmpeg(
      core,
      decodeArgs,
      (p) => postProgress("decoding", 10 + p * 22)
    );
    deleteFile(core, inFile);
    inFile = null;

    let wavBytes = core.FS.readFile("native.wav");
    deleteFile(core, "native.wav");

    let decoded = parseWavInt16(wavBytes);
    wavBytes = null;
    let nativePcm = decoded.pcm;
    const { sampleRate, channels, frames } = decoded;
    decoded = null;

    const totalDuration = frames / sampleRate;
    self.postMessage({
      type: "meta",
      duration: totalDuration,
      sampleRate,
      channels,
      coreMode: loadedMode,
    });

    postProgress("analyzing", 35);
    let vadAudio16k = buildVadAudio16k(
      nativePcm,
      sampleRate,
      channels,
      frames,
      useHeadroom ? 2.0 : 1.0
    );

    const hasUserRanges = Array.isArray(ranges) && ranges.length > 0;
    const isPrayerMode = mode === "prayer";
    const effectiveRanges = resolveEffectiveRanges(
      hasUserRanges ? ranges : [],
      totalDuration,
      isPrayerMode ? rangeAction : "keep"
    );

    const rangeResults = [];
    for (let rIdx = 0; rIdx < effectiveRanges.length; rIdx++) {
      const r = effectiveRanges[rIdx];
      const clipStart = r.startS;
      const clipEnd = r.endS;
      const clipDuration = clipEnd - clipStart;
      const s16k = Math.max(0, Math.round(clipStart * VAD_SAMPLE_RATE));
      const e16k = Math.min(
        vadAudio16k.length,
        Math.max(s16k + 1, Math.round(clipEnd * VAD_SAMPLE_RATE))
      );
      const slice16k = vadAudio16k.subarray(s16k, e16k);

      const rangeBasePct = 36 + (rIdx / effectiveRanges.length) * 28;
      const rangeSpanPct = 28 / effectiveRanges.length;

      const vadRes = detectSpeechSegments(
        slice16k,
        {},
        {
          exactDuration: clipDuration,
          clipStartS: clipStart,
          progressCb: (frac) =>
            postProgress("analyzing", rangeBasePct + frac * rangeSpanPct),
        }
      );
      rangeResults.push(vadRes);
    }

    // Release 16 kHz analysis buffer immediately
    vadAudio16k = null;

    const decision = applyPrayerModeFallback(rangeResults, effectiveRanges, {
      isPrayerMode,
      hasUserRanges,
    });

    if (decision.action === "abort_low_snr") {
      nativePcm = null;
      deleteFile(core, "meta.txt");
      self.postMessage({
        type: "low_snr_guarded",
        originalDuration: totalDuration,
        profile: rangeResults[0].profile,
      });
      return;
    }

    const silenceTrimSkipped = decision.silenceTrimSkipped;
    const silenceTrimStatus = decision.silenceTrimStatus;

    postProgress("splicing", 66);
    const trimmedStart = isPrayerMode && hasUserRanges && effectiveRanges[0].startS > 0.05;
    const trimmedEnd =
      isPrayerMode &&
      hasUserRanges &&
      effectiveRanges[effectiveRanges.length - 1].endS < totalDuration - 0.05;
    let spliced = spliceRangesInt16({
      pcm: nativePcm,
      sampleRate,
      channels,
      rangeResults: decision.rangeResults,
      edgeFadeInMs: trimmedStart ? INTER_RANGE_MICRO_FADE_MS : EDGE_FADE_IN_MS,
      edgeFadeOutMs: trimmedEnd ? INTER_RANGE_MICRO_FADE_MS : EDGE_FADE_OUT_MS,
    });
    // Aggressively release large native PCM buffer before creating output WAV
    nativePcm = null;

    const originalDuration = totalDuration;
    const finalDuration = spliced.duration;

    postProgress("exporting", 80);
    let wavOutBuf = encodeWavInt16(spliced.pcm, sampleRate, channels);
    spliced = null;

    if (quality === "lossless_wav") {
      deleteFile(core, "meta.txt");
      postProgress("exporting", 100);
      self.postMessage(
        {
          type: "done",
          buffer: wavOutBuf,
          mimeType: MIME_TYPES.wav,
          ext: "wav",
          quality: "lossless_wav",
          originalDuration,
          finalDuration,
          silenceTrimSkipped,
          silenceTrimStatus,
          coreMode: loadedMode,
        },
        [wavOutBuf]
      );
      return;
    }

    const effFmt = outFormat === "mp3" ? "mp3" : "m4a";
    const bitrateKbps = QUALITY_PRESETS[quality]?.bitrateKbps ?? 192;
    const outFile = `output.${effFmt}`;

    core.FS.writeFile("spliced.wav", new Uint8Array(wavOutBuf));
    wavOutBuf = null;

    const encodeArgs = ["-y", "-i", "spliced.wav"];
    if (hasMetadata) {
      encodeArgs.push("-i", "meta.txt", "-map", "0:a", "-map_metadata", "1");
    }
    if (useHeadroom) {
      encodeArgs.push("-af", "volume=2.0");
    }
    if (effFmt === "mp3") {
      encodeArgs.push("-c:a", "libmp3lame", "-b:a", `${bitrateKbps}k`, outFile);
    } else {
      encodeArgs.push(
        "-c:a",
        "aac",
        "-b:a",
        `${bitrateKbps}k`,
        "-movflags",
        "+faststart",
        outFile
      );
    }

    execFFmpeg(core, encodeArgs, (p) => postProgress("exporting", 82 + p * 17));
    deleteFile(core, "spliced.wav");
    deleteFile(core, "meta.txt");

    let encodedBytes = core.FS.readFile(outFile);
    deleteFile(core, outFile);

    const exactBuf =
      encodedBytes.byteOffset === 0 &&
      encodedBytes.byteLength === encodedBytes.buffer.byteLength
        ? encodedBytes.buffer
        : encodedBytes.buffer.slice(
            encodedBytes.byteOffset,
            encodedBytes.byteOffset + encodedBytes.byteLength
          );
    encodedBytes = null;

    postProgress("exporting", 100);
    self.postMessage(
      {
        type: "done",
        buffer: exactBuf,
        mimeType: MIME_TYPES[effFmt],
        ext: effFmt,
        quality: String(bitrateKbps),
        originalDuration,
        finalDuration,
        silenceTrimSkipped,
        silenceTrimStatus,
        coreMode: loadedMode,
      },
      [exactBuf]
    );
  } catch (err) {
    if (core && inFile) {
      deleteFile(core, inFile);
    }
    if (core) {
      deleteFile(core, "native.wav");
      deleteFile(core, "spliced.wav");
      deleteFile(core, "meta_raw.txt");
      deleteFile(core, "meta.txt");
    }
    self.postMessage({
      type: "error",
      code: err?.message || "UNKNOWN_ERROR",
    });
  }
};
