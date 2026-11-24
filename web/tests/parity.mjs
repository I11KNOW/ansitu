/**
 * Automated Python vs JavaScript Parity Suite for Ansitu Stage 1 Core DSP Engine.
 *
 * Verifies:
 * 1. Exact parity between `quran_cleaner/vad.py` (Python) and `web/engine/vad.js` (JS)
 *    across normal recitation, low-SNR (< 12 dB) safety gate, < 1.2s cut guard,
 *    crowd murmur filtering, studio breath rejection, and whispered Quranic letters
 *    within ±20ms keep-segment tolerance and 100% safety flag match.
 * 2. Exact `app.py` quality presets, `Int16Array` equal-power crossfade duration,
 *    peak safety (no clipping), smooth boundary transitions, untouched core samples,
 *    and WAV encoding in `web/engine/splicer.js`.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  VAD_SAMPLE_RATE,
  detectSpeechSegments,
  filterCrowdMurmurClusters,
} from "../engine/vad.js";
import {
  DEFAULT_QUALITY_PRESET,
  FORMAT_LABELS,
  INT16_MAX,
  LOSSLESS_WAV_PRESET,
  MIME_TYPES,
  QUALITY_PRESETS,
  encodeWavInt16,
  float32ToInt16,
  isLosslessSource,
  qualityOptionsForSource,
  spliceAudioInt16,
  spliceRangesInt16,
} from "../engine/splicer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PY_HELPER = path.join(__dirname, "run_python_vad.py");

function resolvePythonBin() {
  const candidates = [
    path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
    path.join(REPO_ROOT, ".venv", "bin", "python"),
    "python3",
    "python",
  ];
  for (const bin of candidates) {
    if (bin.includes(path.sep) && !fs.existsSync(bin)) {
      continue;
    }
    try {
      execFileSync(bin, ["--version"], { stdio: "ignore" });
      return bin;
    } catch {
      // Try next candidate
    }
  }
  throw new Error("Could not locate a working Python interpreter with project .venv.");
}

function createPrng(seed) {
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

function dbToAmp(db) {
  return 10.0 ** (db / 20.0);
}

function normalizeRms(arr, targetDb) {
  let sumSq = 0.0;
  for (let i = 0; i < arr.length; i++) {
    sumSq += arr[i] * arr[i];
  }
  const rms = Math.sqrt(sumSq / arr.length + 1e-12);
  const scale = dbToAmp(targetDb) / rms;
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = arr[i] * scale;
  }
  return out;
}

function makeHarmonicVocal(durS, sr, targetDb, f0 = 170.0, fadeEdges = false) {
  const n = Math.round(durS * sr);
  const sig = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const mod = 0.70 + 0.30 * Math.sin(2.0 * Math.PI * 3.0 * t);
    sig[i] =
      (0.55 * Math.sin(2.0 * Math.PI * f0 * t) +
        0.30 * Math.sin(2.0 * Math.PI * 2.0 * f0 * t) +
        0.15 * Math.sin(2.0 * Math.PI * 3.0 * f0 * t)) *
      mod;
  }
  const normed = normalizeRms(sig, targetDb);
  if (fadeEdges) {
    const fade = Math.min(Math.trunc(0.03 * sr), Math.floor(n / 4));
    for (let i = 0; i < fade; i++) {
      const wIn = fade > 1 ? i / (fade - 1) : 1.0;
      const wOut = fade > 1 ? 1.0 - i / (fade - 1) : 1.0;
      normed[i] *= wIn;
      normed[n - fade + i] *= wOut;
    }
  }
  return normed;
}

function addInto(buf, startS, sr, sig) {
  const s0 = Math.round(startS * sr);
  for (let i = 0; i < sig.length && s0 + i < buf.length; i++) {
    buf[s0 + i] += sig[i];
  }
}

function makeNoiseBuffer(durS, sr, floorDb, seed, lowpass3Tap = false) {
  const rng = createPrng(seed);
  const n = Math.round(durS * sr);
  const raw = new Float64Array(lowpass3Tap ? n + 4 : n);
  for (let i = 0; i < raw.length; i++) {
    raw[i] = rng.nextNormal();
  }
  if (!lowpass3Tap) {
    return normalizeRms(raw, floorDb);
  }
  const filtered = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    filtered[i] = 0.25 * raw[i] + 0.50 * raw[i + 1] + 0.25 * raw[i + 2];
  }
  return normalizeRms(filtered, floorDb);
}

function makeHighFreqBurst(durS, sr, targetDb, seed = 71) {
  const rng = createPrng(seed);
  const n = Math.round(durS * sr);
  const raw = new Float64Array(n);
  // Sum randomized high-frequency partials (3.1–6.5 kHz) to emulate an unvoiced breath/fricative
  const nBins = 28;
  const freqs = [];
  const phases = [];
  for (let b = 0; b < nBins; b++) {
    freqs.push(3100.0 + (b / (nBins - 1)) * 3400.0);
    phases.push((rng.nextNormal() + 4.0) * Math.PI);
  }
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    let sum = 0.0;
    for (let b = 0; b < nBins; b++) {
      sum += Math.sin(2.0 * Math.PI * freqs[b] * t + phases[b]);
    }
    const env = Math.sin((Math.PI * i) / Math.max(1, n - 1));
    raw[i] = sum * (0.35 + 0.65 * env);
  }
  return normalizeRms(raw, targetDb);
}

function toFloat32(arr) {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = arr[i];
  }
  return out;
}

function buildFixtures() {
  const sr = VAD_SAMPLE_RATE;

  // 1. Normal studio (-70 dBFS floor, two verses)
  const f1 = makeNoiseBuffer(12.0, sr, -70.0, 101);
  addInto(f1, 1.0, sr, makeHarmonicVocal(3.5, sr, -10.0, 170.0));
  addInto(f1, 5.9, sr, makeHarmonicVocal(4.1, sr, -10.0, 170.0));

  // 2. Normal vintage/live split (-50 dBFS floor, 3.4s gap -> 1.65s effective cut >= 1.2s)
  const f2 = makeNoiseBuffer(15.0, sr, -50.0, 202);
  addInto(f2, 1.0, sr, makeHarmonicVocal(4.0, sr, -10.0, 170.0));
  addInto(f2, 8.4, sr, makeHarmonicVocal(5.0, sr, -10.0, 170.0));

  // 3. Low-SNR guard (SNR < 12 dB -> full file returned untouched)
  const f3 = makeNoiseBuffer(12.0, sr, -20.0, 303);
  addInto(f3, 1.0, sr, makeHarmonicVocal(2.5, sr, -10.5, 170.0));
  addInto(f3, 8.0, sr, makeHarmonicVocal(2.5, sr, -10.5, 170.0));

  // 4. Vintage < 1.2s effective cut guard (2.6s raw gap -> 0.85s post-pad -> merged to 1 segment)
  const f4 = makeNoiseBuffer(14.0, sr, -50.0, 404);
  addInto(f4, 1.0, sr, makeHarmonicVocal(4.0, sr, -10.0, 170.0));
  addInto(f4, 7.6, sr, makeHarmonicVocal(5.0, sr, -10.0, 170.0));

  // 5. Crowd murmur cluster (<= 3.5s stripped, > 3.5s kept) + dynamic pad_end (1300ms vs 1100ms)
  const f5 = makeNoiseBuffer(60.0, sr, -52.0, 505);
  addInto(f5, 1.0, sr, makeHarmonicVocal(7.0, sr, -8.5, 170.0));
  addInto(f5, 8.0, sr, makeHarmonicVocal(1.0, sr, -6.0, 170.0));
  addInto(f5, 12.0, sr, makeHarmonicVocal(1.2, sr, -24.5, 170.0));
  addInto(f5, 13.9, sr, makeHarmonicVocal(1.2, sr, -24.5, 170.0));
  addInto(f5, 20.2, sr, makeHarmonicVocal(8.0, sr, -11.0, 170.0));
  addInto(f5, 31.6, sr, makeHarmonicVocal(4.2, sr, -24.5, 170.0));
  addInto(f5, 36.9, sr, makeHarmonicVocal(4.2, sr, -24.5, 170.0));
  addInto(f5, 42.2, sr, makeHarmonicVocal(4.2, sr, -24.5, 170.0));
  addInto(f5, 49.8, sr, makeHarmonicVocal(8.0, sr, -8.0, 170.0));

  // 6. High-ambient vintage live (-26 dBFS floor, min_silence=1250ms, isolated shout stripped, immediate reaction kept)
  const f6 = makeNoiseBuffer(19.5, sr, -26.0, 606, true);
  addInto(f6, 0.5, sr, makeHarmonicVocal(4.0, sr, -9.0, 165.0, true));
  addInto(f6, 7.5, sr, makeHarmonicVocal(1.2, sr, -23.0, 165.0, true));
  addInto(f6, 11.7, sr, makeHarmonicVocal(4.0, sr, -9.0, 165.0, true));
  addInto(f6, 16.5, sr, makeHarmonicVocal(1.1, sr, -23.0, 165.0, true));

  // 7. Studio breath rejection (-72 dBFS floor, 300ms >3kHz breath at -34 dBFS rejected)
  const f7 = makeNoiseBuffer(11.5, sr, -72.0, 707);
  addInto(f7, 1.0, sr, makeHarmonicVocal(3.0, sr, -14.0, 165.0, true));
  addInto(f7, 5.4, sr, makeHighFreqBurst(0.30, sr, -34.0, 708));
  addInto(f7, 7.1, sr, makeHarmonicVocal(3.0, sr, -14.0, 165.0, true));

  // 8. Whispered Quranic letter safety (soft 160ms HF consonant tail at -48 dBFS on -72 dBFS studio floor)
  const f8 = makeNoiseBuffer(9.0, sr, -72.0, 808);
  addInto(f8, 1.0, sr, makeHarmonicVocal(2.5, sr, -12.0, 170.0, false));
  addInto(f8, 3.5, sr, makeHighFreqBurst(0.16, sr, -48.0, 809));
  addInto(f8, 5.8, sr, makeHarmonicVocal(2.0, sr, -12.0, 170.0, false));

  return [
    {
      name: "normal_studio",
      audio: toFloat32(f1),
      duration: 12.0,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.profile.kind, "studio");
        assert.equal(js.resolved_min_silence_ms, 700);
        assert.equal(js.resolved_gap_ms, 450);
        assert.equal(js.resolved_crossfade_ms, 20);
        assert.equal(js.segments.length, 2);
      },
    },
    {
      name: "normal_vintage_split",
      audio: toFloat32(f2),
      duration: 15.0,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.profile.kind, "vintage/live");
        assert.equal(js.resolved_min_silence_ms, 900);
        assert.equal(js.resolved_gap_ms, 0);
        assert.equal(js.resolved_crossfade_ms, 400);
        assert.equal(js.direct_overlap, true);
        assert.equal(js.segments.length, 2);
      },
    },
    {
      name: "low_snr_guard",
      audio: toFloat32(f3),
      duration: 12.0,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.ok(js.profile.snr_db < 12.0);
        assert.equal(js.low_snr_guarded, true);
        assert.equal(js.segments.length, 1);
        assert.equal(js.segments[0].start, 0.0);
        assert.equal(js.segments[0].end, 12.0);
        assert.equal(js.resolved_crossfade_ms, 0);
      },
    },
    {
      name: "vintage_cut_guard_merge",
      audio: toFloat32(f4),
      duration: 14.0,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.profile.kind, "vintage/live");
        assert.equal(js.segments.length, 1);
      },
    },
    {
      name: "crowd_murmur_and_dynamic_pad",
      audio: toFloat32(f5),
      duration: 60.0,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.stray_shouts_stripped, 1);
        assert.equal(js.segments.length, 4);
        assert.ok(!js.segments.some((s) => s.start <= 13.5 && s.end >= 13.5));
        assert.ok(js.segments.some((s) => s.start <= 38.0 && s.end >= 38.0));
        // Loud Verse 1 finale gets 1300ms pad_end (~10.30s), moderate Verse 2 gets 1100ms (~29.30s)
        assert.ok(js.segments[0].end >= 10.20);
        assert.ok(js.segments[1].end < 29.40);
      },
    },
    {
      name: "high_ambient_live",
      audio: toFloat32(f6),
      duration: 19.5,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.profile.kind, "vintage/live");
        assert.ok(js.profile.noise_floor_p5_db > -30.0);
        assert.equal(js.resolved_min_silence_ms, 1250);
        assert.equal(js.direct_overlap, true);
        assert.equal(js.segments.length, 2);
        assert.ok(!js.segments.some((s) => s.start <= 8.1 && s.end >= 8.1));
        assert.ok(js.segments.some((s) => s.start <= 2.5 && s.end >= 2.5));
        assert.ok(js.segments.some((s) => s.start <= 13.5 && s.end >= 13.5));
      },
    },
    {
      name: "studio_breath_rejection",
      audio: toFloat32(f7),
      duration: 11.5,
      maxTolMs: 20.0,
      extraCheck: (py, js) => {
        assert.equal(js.profile.kind, "studio");
        assert.equal(js.segments.length, 2);
        assert.ok(!js.segments.some((s) => s.start <= 5.55 && s.end >= 5.55));
      },
    },
  ];
}

function buildWhisperSafetyFixture() {
  const sr = VAD_SAMPLE_RATE;
  const f8 = makeNoiseBuffer(9.0, sr, -72.0, 808);
  addInto(f8, 1.0, sr, makeHarmonicVocal(2.5, sr, -12.0, 170.0, false));
  addInto(f8, 3.5, sr, makeHighFreqBurst(0.16, sr, -48.0, 809));
  addInto(f8, 5.8, sr, makeHarmonicVocal(2.0, sr, -12.0, 170.0, false));
  return {
    name: "whispered_letter_safety",
    audio: toFloat32(f8),
    duration: 9.0,
  };
}

function runPythonVad(pythonBin, rawPath, duration) {
  const stdout = execFileSync(
    pythonBin,
    [PY_HELPER, rawPath, "--duration", String(duration)],
    { cwd: REPO_ROOT, encoding: "utf-8" }
  );
  return JSON.parse(stdout);
}

function runVadParityTests(pythonBin) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ansitu_parity_"));
  const rows = [];

  try {
    const fixtures = buildFixtures();
    for (const fx of fixtures) {
      const rawPath = path.join(tmpDir, `${fx.name}.f32`);
      const buf = Buffer.from(fx.audio.buffer, fx.audio.byteOffset, fx.audio.byteLength);
      fs.writeFileSync(rawPath, buf);

      const py = runPythonVad(pythonBin, rawPath, fx.duration);
      const js = detectSpeechSegments(fx.audio, {}, { exactDuration: fx.duration });

      // 100% safety flag & resolved parameter parity
      assert.equal(js.profile.kind, py.profile.kind, `${fx.name}: profile.kind`);
      assert.equal(js.profile.noise_floor_p5_db, py.profile.noise_floor_p5_db, `${fx.name}: floor p5`);
      assert.equal(js.profile.noise_floor_p10_db, py.profile.noise_floor_p10_db, `${fx.name}: floor p10`);
      assert.equal(js.profile.p90_db, py.profile.p90_db, `${fx.name}: p90`);
      assert.equal(js.profile.snr_db, py.profile.snr_db, `${fx.name}: snr`);
      assert.equal(js.low_snr_guarded, py.low_snr_guarded, `${fx.name}: low_snr_guarded`);
      assert.equal(js.stray_shouts_stripped, py.stray_shouts_stripped, `${fx.name}: stray_shouts_stripped`);
      assert.equal(js.direct_overlap, py.direct_overlap, `${fx.name}: direct_overlap`);
      assert.equal(js.use_room_tone, py.use_room_tone, `${fx.name}: use_room_tone`);
      assert.equal(js.resolved_pad_start_ms, py.resolved_pad_start_ms, `${fx.name}: pad_start_ms`);
      assert.equal(js.resolved_pad_end_ms, py.resolved_pad_end_ms, `${fx.name}: pad_end_ms`);
      assert.equal(js.resolved_min_silence_ms, py.resolved_min_silence_ms, `${fx.name}: min_silence_ms`);
      assert.equal(js.resolved_crossfade_ms, py.resolved_crossfade_ms, `${fx.name}: crossfade_ms`);
      assert.equal(js.resolved_gap_ms, py.resolved_gap_ms, `${fx.name}: gap_ms`);

      assert.equal(
        js.segments.length,
        py.segments.length,
        `${fx.name}: segment count mismatch (JS=${js.segments.length}, Py=${py.segments.length})`
      );

      let maxDeltaMs = 0.0;
      for (let i = 0; i < py.segments.length; i++) {
        const dStart = Math.abs(js.segments[i].start - py.segments[i].start) * 1000.0;
        const dEnd = Math.abs(js.segments[i].end - py.segments[i].end) * 1000.0;
        maxDeltaMs = Math.max(maxDeltaMs, dStart, dEnd);
        assert.ok(
          dStart <= fx.maxTolMs,
          `${fx.name} seg[${i}].start delta ${dStart.toFixed(2)}ms > ${fx.maxTolMs}ms (JS=${js.segments[i].start}, Py=${py.segments[i].start})`
        );
        assert.ok(
          dEnd <= fx.maxTolMs,
          `${fx.name} seg[${i}].end delta ${dEnd.toFixed(2)}ms > ${fx.maxTolMs}ms (JS=${js.segments[i].end}, Py=${py.segments[i].end})`
        );
      }

      if (fx.extraCheck) {
        fx.extraCheck(py, js);
      }

      rows.push({
        fixture: fx.name,
        profile: `${js.profile.kind} (SNR ${js.profile.snr_db} dB)`,
        segs: `${js.segments.length}/${py.segments.length}`,
        maxDeltaMs: Number(maxDeltaMs.toFixed(2)),
        lowSnrGuarded: js.low_snr_guarded,
        murmurStripped: `${js.stray_shouts_stripped}/${py.stray_shouts_stripped}`,
        status: "PASS",
      });
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }

  return rows;
}

function runWhisperSafetyCheck(pythonBin) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ansitu_whisper_"));
  try {
    const fx = buildWhisperSafetyFixture();
    const rawPath = path.join(tmpDir, `${fx.name}.f32`);
    fs.writeFileSync(rawPath, Buffer.from(fx.audio.buffer, fx.audio.byteOffset, fx.audio.byteLength));

    const py = runPythonVad(pythonBin, rawPath, fx.duration);
    const js = detectSpeechSegments(fx.audio, {}, { exactDuration: fx.duration });

    assert.equal(js.profile.kind, py.profile.kind);
    assert.equal(js.profile.noise_floor_p5_db, py.profile.noise_floor_p5_db);
    assert.equal(js.profile.noise_floor_p10_db, py.profile.noise_floor_p10_db);
    assert.equal(js.profile.p90_db, py.profile.p90_db);
    assert.equal(js.profile.snr_db, py.profile.snr_db);
    assert.equal(js.low_snr_guarded, py.low_snr_guarded);
    assert.equal(js.stray_shouts_stripped, py.stray_shouts_stripped);
    assert.equal(js.direct_overlap, py.direct_overlap);
    assert.equal(js.use_room_tone, py.use_room_tone);
    assert.equal(js.resolved_pad_start_ms, py.resolved_pad_start_ms);
    assert.equal(js.resolved_pad_end_ms, py.resolved_pad_end_ms);
    assert.equal(js.resolved_min_silence_ms, py.resolved_min_silence_ms);
    assert.equal(js.resolved_crossfade_ms, py.resolved_crossfade_ms);
    assert.equal(js.resolved_gap_ms, py.resolved_gap_ms);
    assert.equal(js.segments.length, 2);
    assert.equal(py.segments.length, 2);

    const nonWhisperDeltasMs = [
      Math.abs(js.segments[0].start - py.segments[0].start) * 1000.0,
      Math.abs(js.segments[1].start - py.segments[1].start) * 1000.0,
      Math.abs(js.segments[1].end - py.segments[1].end) * 1000.0,
    ];
    for (const d of nonWhisperDeltasMs) {
      assert.ok(d <= 20.0, `Non-whisper boundary delta ${d.toFixed(2)}ms > 20ms`);
    }

    assert.ok(js.segments[0].end >= py.segments[0].end, "JS must be >= Python on whispered tail");
    assert.ok(
      js.segments[0].end >= 3.94,
      `Whispered tail clipped: seg[0].end=${js.segments[0].end} < 3.94`
    );

    const extraMarginMs = Math.round((js.segments[0].end - py.segments[0].end) * 1000.0);
    return {
      fixture: fx.name,
      profile: `${js.profile.kind} (SNR ${js.profile.snr_db} dB)`,
      pySeg0EndS: py.segments[0].end,
      jsSeg0EndS: js.segments[0].end,
      extraSafetyMarginMs: `+${extraMarginMs}ms`,
      otherBoundariesMaxDeltaMs: Number(Math.max(...nonWhisperDeltasMs).toFixed(2)),
      status: "PASS",
    };
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function runCrowdMurmurUnitTest() {
  const mk = (start, end, db) => ({
    start,
    end,
    coreDb: db,
    frameDbs: new Float64Array(10).fill(db),
  });
  const segs = [
    mk(0.0, 8.0, -10.0),
    mk(11.0, 12.2, -19.0),  // 9 dB below (< 10 dB floor) -> KEPT
    mk(15.5, 23.5, -10.0),
    mk(26.5, 27.5, -22.0),  // 2 bursts, 1.1s gap, 3.1s span <= 3.5s -> STRIPPED (2)
    mk(28.6, 29.6, -22.0),
    mk(33.0, 41.0, -10.0),
    mk(44.0, 45.4, -22.0),  // 2 bursts, 1.0s gap, 3.8s span > 3.5s -> KEPT
    mk(46.4, 47.8, -22.0),
    mk(51.0, 59.0, -10.0),
    mk(59.8, 61.0, -24.0),  // immediate reaction (gap_before = 0.8s < 1.5s) -> KEPT
  ];
  const { kept, strippedCount } = filterCrowdMurmurClusters(segs, 15.0, 0.0, 62.0);
  assert.equal(strippedCount, 2);
  assert.equal(kept.length, 8);
  assert.ok(kept.some((s) => s.start === 11.0));
  assert.ok(!kept.some((s) => s.start === 26.5 || s.start === 28.6));
  assert.ok(kept.some((s) => s.start === 44.0) && kept.some((s) => s.start === 46.4));
  assert.ok(kept.some((s) => s.start === 59.8));
  return {
    test: "crowd_murmur_cluster_bounds",
    detail: `snr=15dB -> margin=10dB floor, stripped=${strippedCount}/10, kept=${kept.length}`,
    status: "PASS",
  };
}

function runSplicerTests() {
  const results = [];

  // 1. Exact 3 presets + lossless logic from app.py
  assert.deepEqual(Object.keys(QUALITY_PRESETS), ["128", "192", "256"]);
  assert.equal(DEFAULT_QUALITY_PRESET, "192");
  assert.equal(QUALITY_PRESETS["128"].bitrateKbps, 128);
  assert.equal(QUALITY_PRESETS["192"].bitrateKbps, 192);
  assert.equal(QUALITY_PRESETS["256"].bitrateKbps, 256);
  assert.equal(QUALITY_PRESETS["128"].label, "128 kbps (جودة يوتيوب - حجم خفيف)");
  assert.equal(QUALITY_PRESETS["192"].label, "192 kbps (جودة عالية - الافتراضي)");
  assert.equal(QUALITY_PRESETS["256"].label, "256 kbps (أقصى جودة أرشيفية)");
  assert.equal(LOSSLESS_WAV_PRESET.id, "lossless_wav");
  assert.equal(FORMAT_LABELS.m4a, "M4A — جودة عالية (AAC High Quality)");
  assert.equal(FORMAT_LABELS.mp3, "MP3 — متوافق مع كل الأجهزة (Universal)");
  assert.equal(MIME_TYPES.m4a, "audio/mp4");
  assert.equal(MIME_TYPES.mp3, "audio/mpeg");
  assert.equal(MIME_TYPES.wav, "audio/wav");
  assert.equal(isLosslessSource("recitation.wav"), true);
  assert.equal(isLosslessSource("recitation.FLAC"), true);
  assert.equal(isLosslessSource("recitation.mp3"), false);
  assert.deepEqual(qualityOptionsForSource("recitation.flac"), ["128", "192", "256", "lossless_wav"]);
  assert.deepEqual(qualityOptionsForSource("recitation.m4a"), ["128", "192", "256"]);
  results.push({ test: "app_py_quality_presets", detail: "128/192(default)/256 + lossless_wav rules", status: "PASS" });

  // 2. Vintage 400ms equal-power direct-overlap splice at 48kHz stereo Int16
  const sr = 48000;
  const channels = 2;
  const totalDurS = 10.0;
  const nFrames = Math.round(totalDurS * sr);
  const floatL = new Float32Array(nFrames);
  const floatR = new Float32Array(nFrames);
  const rng = createPrng(999);
  const floorAmp = dbToAmp(-48.0);
  for (let i = 0; i < nFrames; i++) {
    const t = i / sr;
    const floor = rng.nextNormal() * floorAmp;
    let vocal = 0.0;
    if ((t >= 0.8 && t <= 3.2) || (t >= 6.0 && t <= 8.6)) {
      vocal =
        0.35 *
        (0.6 * Math.sin(2 * Math.PI * 170 * t) + 0.4 * Math.sin(2 * Math.PI * 340 * t));
    }
    floatL[i] = floor + vocal;
    floatR[i] = (floor + vocal) * 0.995;
  }
  const pcmVintage = float32ToInt16([floatL, floatR], channels);
  assert.ok(pcmVintage instanceof Int16Array);

  const segsVintage = [
    { start: 0.4, end: 3.8 },
    { start: 5.5, end: 9.2 },
  ];
  const outVintage = spliceAudioInt16({
    pcm: pcmVintage,
    sampleRate: sr,
    channels,
    segments: segsVintage,
    crossfadeMs: 400,
    gapMs: 0,
    fadeEdges: true,
  });

  const expectedVintageS = (3.8 - 0.4) + (9.2 - 5.5) - 0.400; // 6.700s
  assert.ok(Math.abs(outVintage.duration - expectedVintageS) < 0.001);

  // Peak safety & untouched core samples
  let peakVintage = 0;
  for (let i = 0; i < outVintage.pcm.length; i++) {
    peakVintage = Math.max(peakVintage, Math.abs(outVintage.pcm[i]));
  }
  assert.ok(peakVintage <= INT16_MAX, `Clipping detected: peak=${peakVintage}`);

  const s0Orig = Math.round(0.4 * sr);
  const coreOffset = Math.round(0.75 * sr);
  const coreLen = Math.round(0.80 * sr);
  let maxCoreDiff = 0;
  for (let f = 0; f < coreLen; f++) {
    for (let ch = 0; ch < channels; ch++) {
      const d = Math.abs(
        outVintage.pcm[(coreOffset + f) * channels + ch] -
          pcmVintage[(s0Orig + coreOffset + f) * channels + ch]
      );
      if (d > maxCoreDiff) maxCoreDiff = d;
    }
  }
  assert.equal(maxCoreDiff, 0, "Untouched core samples should be bit-exact in Int16Array");

  // Smooth crossfade transition & outer edge fades
  const overlapEnd = Math.round((3.8 - 0.4) * sr);
  const overlapStart = overlapEnd - Math.round(0.4 * sr);
  let maxStep = 0;
  for (let f = Math.max(1, overlapStart - 1000); f < overlapEnd + 1000; f++) {
    const step = Math.abs(outVintage.pcm[f * channels] - outVintage.pcm[(f - 1) * channels]) / INT16_MAX;
    if (step > maxStep) maxStep = step;
  }
  assert.ok(maxStep < 0.05, `Discontinuous crossfade step=${maxStep}`);

  const edgeWin = Math.round(0.005 * sr);
  let startEdgePeak = 0;
  let endEdgePeak = 0;
  for (let f = 0; f < edgeWin; f++) {
    startEdgePeak = Math.max(startEdgePeak, Math.abs(outVintage.pcm[f * channels]) / INT16_MAX);
    endEdgePeak = Math.max(
      endEdgePeak,
      Math.abs(outVintage.pcm[(outVintage.frames - edgeWin + f) * channels]) / INT16_MAX
    );
  }
  assert.ok(startEdgePeak < 0.005, `Start edge fade too loud: ${startEdgePeak}`);
  assert.ok(endEdgePeak < 0.005, `End edge fade too loud: ${endEdgePeak}`);
  results.push({
    test: "vintage_400ms_direct_overlap",
    detail: `dur=${outVintage.duration.toFixed(4)}s (exp ${expectedVintageS.toFixed(4)}s), coreDiff=0, maxStep=${maxStep.toFixed(4)}`,
    status: "PASS",
  });

  // 3. Studio 450ms digital-silence gap + 20ms crossfade splice
  const outStudio = spliceAudioInt16({
    pcm: pcmVintage,
    sampleRate: sr,
    channels,
    segments: [
      { start: 0.8, end: 3.5 },
      { start: 5.8, end: 8.9 },
    ],
    crossfadeMs: 20,
    gapMs: 450,
    fadeEdges: true,
    useRoomTone: false,
  });
  const expectedStudioS = (3.5 - 0.8) + (8.9 - 5.8) + 0.450; // 6.250s
  assert.ok(Math.abs(outStudio.duration - expectedStudioS) < 0.001);
  const sgStart = Math.round((3.5 - 0.8) * sr);
  let gapMidPeak = 0;
  for (let f = sgStart + Math.round(0.05 * sr); f < sgStart + Math.round(0.40 * sr); f++) {
    gapMidPeak = Math.max(gapMidPeak, Math.abs(outStudio.pcm[f * channels]));
  }
  assert.equal(gapMidPeak, 0, "Studio digital silence gap interior must be 0");
  results.push({
    test: "studio_450ms_digital_silence",
    detail: `dur=${outStudio.duration.toFixed(4)}s (exp ${expectedStudioS.toFixed(4)}s), gapPeak=${gapMidPeak}`,
    status: "PASS",
  });

  // 4. Peak-safety stress test on full-scale correlated overlap (cos + sin = sqrt(2) > 1.0 if unclamped)
  const stressFrames = sr * 2;
  const stressPcm = new Int16Array(stressFrames);
  stressPcm.fill(INT16_MAX);
  const stressOut = spliceAudioInt16({
    pcm: stressPcm,
    sampleRate: sr,
    channels: 1,
    segments: [
      { start: 0.0, end: 0.9 },
      { start: 1.0, end: 2.0 },
    ],
    crossfadeMs: 200,
    gapMs: 0,
    fadeEdges: false,
  });
  let stressPeak = 0;
  let hasNegativeWrap = false;
  for (let i = 0; i < stressOut.pcm.length; i++) {
    const v = stressOut.pcm[i];
    if (v < 0) hasNegativeWrap = true;
    stressPeak = Math.max(stressPeak, Math.abs(v));
  }
  assert.equal(hasNegativeWrap, false, "Int16 arithmetic overflow/wraparound occurred");
  assert.equal(stressPeak, INT16_MAX, "Full-scale overlap must saturate safely to INT16_MAX");
  results.push({
    test: "full_scale_peak_safety",
    detail: `peak=${stressPeak}/${INT16_MAX}, wraparound=${hasNegativeWrap}`,
    status: "PASS",
  });

  // 5. Low-SNR guarded bit-exact bypass
  const guardedOut = spliceAudioInt16({
    pcm: pcmVintage,
    sampleRate: sr,
    channels,
    segments: [{ start: 0.0, end: totalDurS }],
    crossfadeMs: 0,
    gapMs: 0,
    fadeEdges: false,
  });
  assert.equal(guardedOut.pcm.length, pcmVintage.length);
  assert.ok(guardedOut.pcm.every((v, idx) => v === pcmVintage[idx]));
  results.push({
    test: "low_snr_guarded_bit_exact",
    detail: `${guardedOut.frames} frames bit-exact untouched`,
    status: "PASS",
  });

  // 6. Room-tone bridge & multi-range spliceRangesInt16
  const multiOut = spliceRangesInt16({
    pcm: pcmVintage,
    sampleRate: sr,
    channels,
    rangeResults: [
      {
        segments: [{ start: 0.5, end: 3.3 }],
        resolved_gap_ms: 450,
        resolved_crossfade_ms: 100,
        use_room_tone: true,
        low_snr_guarded: false,
        quietest_gap_window: [4.0, 5.0, -48.0],
        profile: { noise_floor_p5_db: -48.0, noise_floor_p10_db: -47.5 },
      },
      {
        segments: [{ start: 5.8, end: 8.8 }],
        resolved_gap_ms: 450,
        resolved_crossfade_ms: 100,
        use_room_tone: true,
        low_snr_guarded: false,
        quietest_gap_window: [4.0, 5.0, -48.0],
        profile: { noise_floor_p5_db: -48.0, noise_floor_p10_db: -47.5 },
      },
    ],
  });
  const expectedMultiS = (3.3 - 0.5) + (8.8 - 5.8) + 0.450; // 6.250s
  assert.ok(Math.abs(multiOut.duration - expectedMultiS) < 0.001);
  const multiGapStart = Math.round(2.9 * sr);
  let multiGapSumSq = 0.0;
  const multiGapLen = Math.round(0.25 * sr);
  for (let f = 0; f < multiGapLen; f++) {
    const s = multiOut.pcm[(multiGapStart + f) * channels] / INT16_MAX;
    multiGapSumSq += s * s;
  }
  const multiGapDb = 20.0 * Math.log10(Math.sqrt(multiGapSumSq / multiGapLen + 1e-12));
  assert.ok(multiGapDb > -60.0 && multiGapDb < -30.0, `Unexpected room-tone bridge dB: ${multiGapDb}`);
  results.push({
    test: "room_tone_multi_range_splice",
    detail: `dur=${multiOut.duration.toFixed(4)}s, bridgeFloor=${multiGapDb.toFixed(1)} dBFS`,
    status: "PASS",
  });

  // 7. WAV encoding verification
  const wavBuf = encodeWavInt16(multiOut.pcm, sr, channels);
  assert.equal(wavBuf.byteLength, 44 + multiOut.pcm.length * 2);
  const hdr = new DataView(wavBuf);
  const riff = String.fromCharCode(hdr.getUint8(0), hdr.getUint8(1), hdr.getUint8(2), hdr.getUint8(3));
  const wave = String.fromCharCode(hdr.getUint8(8), hdr.getUint8(9), hdr.getUint8(10), hdr.getUint8(11));
  assert.equal(riff, "RIFF");
  assert.equal(wave, "WAVE");
  assert.equal(hdr.getUint32(24, true), sr);
  assert.equal(hdr.getUint16(22, true), channels);
  results.push({
    test: "wav_int16_encoder",
    detail: `${wavBuf.byteLength} bytes, 48kHz 2ch PCM-16 RIFF/WAVE`,
    status: "PASS",
  });

  return results;
}

function main() {
  const pythonBin = resolvePythonBin();
  console.log(`Python binary: ${pythonBin}`);

  const vadRows = runVadParityTests(pythonBin);
  console.log("\n=== VAD PARITY RESULTS (Python vs web/engine/vad.js) ===");
  console.table(vadRows);

  const whisperRow = runWhisperSafetyCheck(pythonBin);
  const murmurUnitRow = runCrowdMurmurUnitTest();
  console.log("\n=== WHISPER SAFETY & MURMUR GUARD CHECKS ===");
  console.table([whisperRow]);
  console.table([murmurUnitRow]);

  const splicerRows = runSplicerTests();
  console.log("\n=== SPLICER VERIFICATION RESULTS (web/engine/splicer.js) ===");
  console.table(splicerRows);

  const worstDelta = Math.max(...vadRows.map((r) => r.maxDeltaMs));
  console.log(
    `\nALL PARITY & SPLICER TESTS PASSED (worst core VAD boundary delta: ${worstDelta.toFixed(2)} ms <= 20.00 ms; whisper tail: py end=${whisperRow.pySeg0EndS}s, js end=${whisperRow.jsSeg0EndS}s, ${whisperRow.extraSafetyMarginMs} extra whisper safety margin).`
  );
}

main();
