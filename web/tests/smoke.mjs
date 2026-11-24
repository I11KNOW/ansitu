/**
 * Automated Smoke & Compliance Verification for ansitu Stage 2 Web App.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { WEB3FORMS_ACCESS_KEY } from "../config.js";
import { createStaticServer } from "../dev-server.mjs";
import {
  MONTAGE_DECLICK_MS,
  buildMontageInt16,
  decidePrayerPostVad,
  formatTimestampDigits,
  formatTimestampOnBlur,
  resolveEffectiveRanges,
  spliceRangesInt16,
} from "../engine/splicer.js";
import {
  TRANSLATIONS,
  buildOutputFilename,
  formatClock,
  parseTimestamp,
} from "../i18n.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(__dirname, "..");

function checkHeadersFile() {
  const raw = fs.readFileSync(path.join(WEB_ROOT, "_headers"), "utf-8");
  assert.match(raw, /Cross-Origin-Opener-Policy:\s*same-origin/);
  assert.match(raw, /Cross-Origin-Embedder-Policy:\s*require-corp/);
}

function checkManifestAndAssets() {
  const manifestPath = path.join(WEB_ROOT, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
  assert.equal(manifest.dir, "rtl");
  assert.equal(manifest.lang, "ar");
  assert.ok(Array.isArray(manifest.icons) && manifest.icons.length >= 2);

  for (const icon of manifest.icons) {
    const abs = path.resolve(WEB_ROOT, icon.src);
    assert.ok(fs.existsSync(abs), `Missing icon referenced by manifest: ${icon.src}`);
    if (abs.endsWith(".png")) {
      const hdr = fs.readFileSync(abs).subarray(0, 8);
      assert.deepEqual(
        Array.from(hdr),
        [137, 80, 78, 71, 13, 10, 26, 10],
        `Invalid PNG header: ${icon.src}`
      );
    }
  }

  for (const rel of ["icons/apple-touch-icon.png", "assets/og-preview.png"]) {
    const abs = path.join(WEB_ROOT, rel);
    assert.ok(fs.existsSync(abs), `Missing asset: ${rel}`);
    const hdr = fs.readFileSync(abs).subarray(0, 8);
    assert.deepEqual(Array.from(hdr), [137, 80, 78, 71, 13, 10, 26, 10]);
  }
}

function checkI18nAndHtmlRules() {
  const ar = TRANSLATIONS.ar;
  const en = TRANSLATIONS.en;

  assert.equal(ar.dir, "rtl");
  assert.equal(ar.pageTitle, "أَنصِتُوا");
  assert.equal(ar.brand, "أَنصِتُوا");
  assert.equal(ar.heroTitle, "تلاوات خاشعة، بلا سكتات طويلة");
  assert.equal(
    ar.heroSubtitle,
    "احذف السكتات الطويلة بين الآيات تلقائياً، مع الحفاظ على كل حرف من التلاوة"
  );
  assert.ok(!ar.heroSubtitle.includes("—"), "Hero subtitle must not contain em-dash");
  assert.equal(ar.pickFile, "اختر ملفاً صوتياً");
  assert.ok(!ar.pickFile.includes("ارفع"), "File picker must avoid 'ارفع'");
  assert.equal(ar.dropHint, "أو اسحب الملف وأفلته هنا");
  assert.equal(ar.feedbackLink, "ملاحظات");
  assert.equal(ar.startCleaning, "تنقية التلاوة");
  assert.equal(ar.tabStandard, "تلاوة عادية");
  assert.equal(ar.tabPrayer, "تلاوة صلاة");
  assert.equal(ar.formatOptions.mp3, "\u2066MP3\u2069 (الأكثر انتشاراً)");
  assert.equal(ar.formatOptions.m4a, "\u2066M4A\u2069 (الأنقى)");
  assert.equal(ar.qualityOptions["192"], "\u2066192 kbps\u2069 (عالية)");
  assert.equal(ar.qualityOptions.lossless_wav, "\u2066WAV\u2069 (بدون فقدان)");

  assert.equal(en.dir, "ltr");
  assert.equal(en.pageTitle, "ansitu");
  assert.equal(en.brand, "ansitu");
  assert.equal(
    en.heroSubtitle,
    "Automatically remove long pauses between verses while preserving every letter of the recitation."
  );
  assert.equal(en.dropHint, "Or drag and drop an audio file here");
  assert.equal(en.feedbackLink, "Feedback");
  assert.equal(en.startCleaning, "Clean recitation");
  assert.equal(en.tabStandard, "Standard Recitation");
  assert.equal(en.tabPrayer, "Prayer Recitation");
  assert.equal(en.formatOptions.mp3, "MP3 (Universal)");
  assert.equal(en.formatOptions.m4a, "M4A (Cleaner)");
  assert.equal(en.qualityOptions["192"], "192 kbps (High)");
  assert.equal(en.qualityOptions.lossless_wav, "WAV (Lossless)");

  assert.equal(formatClock(0), "00:00");
  assert.equal(formatClock(3182), "53:02");
  assert.equal(formatClock(3723), "01:02:03");

  assert.equal(parseTimestamp("05:00"), 300);
  assert.equal(parseTimestamp("01:02:03"), 3723);
  assert.equal(parseTimestamp("00.45"), 45);
  assert.equal(parseTimestamp("00,45"), 45);
  assert.equal(parseTimestamp("00-45"), 45);
  assert.equal(parseTimestamp("00 45"), 45);
  assert.equal(parseTimestamp("0045"), 45);
  assert.equal(parseTimestamp("130"), 90);
  assert.equal(parseTimestamp("10530"), 3930);
  assert.equal(parseTimestamp("٠٠.٤٥"), 45);
  assert.equal(formatTimestampDigits("0045"), "00:45");
  assert.equal(formatTimestampDigits("130"), "1:30");
  assert.equal(formatTimestampDigits("10530"), "1:05:30");
  assert.equal(formatTimestampOnBlur("00.45"), "00:45");
  assert.equal(formatTimestampOnBlur("0045"), "00:45");
  assert.equal(formatTimestampOnBlur("130"), "01:30");
  assert.equal(formatTimestampOnBlur("10530"), "1:05:30");
  assert.throws(() => parseTimestamp("bad:time"));
  assert.throws(() => parseTimestamp("01:75"));
  assert.throws(() => parseTimestamp("175"));

  assert.equal(buildOutputFilename("bahtimi_taha.mp3", "m4a"), "bahtimi_taha_ansitu.m4a");
  assert.equal(buildOutputFilename("surah.flac", "wav", "lossless_wav"), "surah_ansitu_lossless.wav");
  assert.equal(
    buildOutputFilename("المنشاوي جودة عالية HD(MP3_160K).mp3", "m4a", "192"),
    "المنشاوي جودة عالية HD_ansitu_192k.m4a"
  );
  assert.equal(
    buildOutputFilename("الاحزاب جودة عالية(MP3_1.mp3", "mp3", "128"),
    "الاحزاب جودة عالية_ansitu_128k.mp3"
  );
  assert.equal(
    buildOutputFilename("الانعام جودة ع(MP3__1.mp3", "m4a", "256"),
    "الانعام جودة ع_ansitu_256k.m4a"
  );
  assert.equal(
    buildOutputFilename("bahtimi (M4A_128K).mp3", "m4a", "192"),
    "bahtimi_ansitu_192k.m4a"
  );
  assert.equal(
    buildOutputFilename("surah (128k).flac", "wav", "lossless_wav"),
    "surah_ansitu_lossless.wav"
  );

  assert.equal(WEB3FORMS_ACCESS_KEY, "0577cc8f-5c12-49b2-8fb9-f1e2c9bf49d2");

  const html = fs.readFileSync(path.join(WEB_ROOT, "index.html"), "utf-8");
  const i18nSrc = fs.readFileSync(path.join(WEB_ROOT, "i18n.js"), "utf-8");
  assert.ok(!html.includes("—"), "index.html must not contain em-dash");
  assert.ok(!i18nSrc.includes("—"), "i18n.js must not contain em-dash");
  assert.ok(html.includes("<title>أَنصِتُوا</title>"));
  assert.ok(!/<audio\b/i.test(html), "Result view must NOT contain an audio preview player");
  assert.ok(html.includes('href="https://github.com/I11KNOW/ansitu"'));
  assert.ok(html.includes('id="feedback-dialog"'));
  assert.ok(html.includes('id="view-low-snr"'));
  assert.ok(html.includes('id="tab-standard"'));
  assert.ok(html.includes('id="tab-prayer"'));
  assert.ok(html.includes('id="range-action-remove"'));
  assert.ok(html.includes('id="range-action-keep"'));
  assert.ok(html.includes('id="result-notice"'));
  assert.equal(ar.rangeActionRemove, "حذف هذه المقاطع");
  assert.equal(ar.rangeActionKeep, "الاحتفاظ بهذه المقاطع فقط");
  assert.equal(en.rangeActionRemove, "Remove these ranges");
  assert.equal(en.rangeActionKeep, "Keep only these ranges");
  assert.ok(ar.resultSilenceSkippedNotice.length > 20);
  assert.ok(en.resultSilenceSkippedNotice.length > 20);
  assert.ok(ar.resultSilencePartialNotice.length > 20);
  assert.ok(en.resultSilencePartialNotice.length > 20);
  assert.ok(ar.trimHintRemove.includes("ثانية"), "AR remove hint must include ~1s margin tip");
  assert.ok(ar.trimHintKeep.includes("ثانية"), "AR keep hint must include ~1s margin tip");
  assert.ok(en.trimHintRemove.includes("1 second"), "EN remove hint must include ~1s margin tip");
  assert.ok(en.trimHintKeep.includes("1 second"), "EN keep hint must include ~1s margin tip");
  assert.ok(ar.errors.allAudioRemoved.length > 10);
  assert.ok(en.errors.allAudioRemoved.length > 10);
  assert.ok(html.includes("تنقية التلاوة"));
  assert.ok(!html.includes('id="trim-toggle"'));
  assert.ok(!html.includes('id="sample-btn"'), "index.html must not contain #sample-btn");
  assert.ok(!html.includes('id="sample-wrap"'), "index.html must not contain #sample-wrap");
  assert.ok(html.includes('viewBox="5 7 48 50"'));
  assert.ok(html.includes('dir="ltr"'));

  const svg = fs.readFileSync(path.join(WEB_ROOT, "icons", "crescent.svg"), "utf-8");
  assert.ok(svg.includes('viewBox="5 7 48 50"'), "crescent.svg must use tightened viewBox");

  const workerSrc = fs.readFileSync(path.join(WEB_ROOT, "worker.js"), "utf-8");
  assert.ok(workerSrc.includes("volume=0.5"), "worker.js must apply -6 dB decode headroom");
  assert.ok(workerSrc.includes("volume=2.0"), "worker.js must restore +6 dB gain on export");
  assert.ok(workerSrc.includes("ffmetadata"), "worker.js must preserve text metadata via ffmetadata");
  assert.ok(workerSrc.includes("isLosslessSource"), "worker.js must preserve bit-exact lossless WAV");
  assert.ok(workerSrc.includes("deleteFile("), "worker.js must explicitly unlink MEMFS temp files");
  assert.ok(!workerSrc.includes("-compression_level"), "worker.js must NOT set -compression_level");

  const swSrc = fs.readFileSync(path.join(WEB_ROOT, "sw.js"), "utf-8");
  const shellMatch = swSrc.match(/const SHELL_ASSETS = \[([\s\S]*?)\];/);
  assert.ok(shellMatch, "sw.js must define SHELL_ASSETS");
  const shellBlock = shellMatch[1];
  assert.ok(!/\.wasm\b/i.test(shellBlock), "sw.js SHELL_ASSETS must not pre-cache WASM binaries");
  assert.ok(!/\.onnx\b/i.test(shellBlock), "sw.js SHELL_ASSETS must not pre-cache ONNX models");
  assert.ok(!shellBlock.includes("og-preview.png"), "sw.js SHELL_ASSETS must not pre-cache og-preview.png");
  assert.ok(!/fonts\.googleapis\.com/i.test(html), "index.html should not block paint with external font stylesheets");
}

function checkPrayerModeLogic() {
  // 1. Keep mode (existing behavior)
  assert.deepEqual(
    resolveEffectiveRanges([{ startS: 10, endS: 50 }, { startS: 120, endS: 400 }], 300, "keep"),
    [{ startS: 10, endS: 50 }, { startS: 120, endS: 300 }]
  );
  assert.deepEqual(resolveEffectiveRanges([], 300, "remove"), [{ startS: 0, endS: 300 }]);

  // 2. Remove mode: single middle cut (e.g., Ruku'/Sujud)
  assert.deepEqual(
    resolveEffectiveRanges([{ startS: 60, endS: 120 }], 300, "remove"),
    [{ startS: 0, endS: 60 }, { startS: 120, endS: 300 }]
  );

  // 3. Remove mode: leading cut & trailing cut
  assert.deepEqual(
    resolveEffectiveRanges([{ startS: null, endS: 45 }], 300, "remove"),
    [{ startS: 45, endS: 300 }]
  );
  assert.deepEqual(
    resolveEffectiveRanges([{ startS: 250, endS: null }], 300, "remove"),
    [{ startS: 0, endS: 250 }]
  );

  // 4. Remove mode: unsorted, overlapping, adjacent, and > duration clamping
  assert.deepEqual(
    resolveEffectiveRanges(
      [
        { startS: 100, endS: 150 },
        { startS: 80, endS: 120 },
        { startS: 150, endS: 180 },
        { startS: 270, endS: 999 },
      ],
      300,
      "remove"
    ),
    [{ startS: 0, endS: 80 }, { startS: 180, endS: 270 }]
  );

  // 5. Remove mode: removing entire file throws ALL_AUDIO_REMOVED
  assert.throws(
    () => resolveEffectiveRanges([{ startS: 0, endS: 300 }], 300, "remove"),
    /ALL_AUDIO_REMOVED/
  );
  assert.throws(
    () =>
      resolveEffectiveRanges(
        [{ startS: 0, endS: 150 }, { startS: 150, endS: 300 }],
        300,
        "remove"
      ),
    /ALL_AUDIO_REMOVED/
  );

  // 6. Step 1 Pure Montage on MONO PCM: exact straight concatenation (NO 400ms gap, NO crossfade)
  //    with 5 ms de-click micro-fade at raw cut boundaries
  const monoSr = 8000;
  const monoDurS = 8;
  const monoPcm = new Int16Array(monoSr * monoDurS);
  // [0..2s]: +12000; [2..5s] (removed): -25000; [5..8s]: +20000
  for (let i = 0; i < monoSr * 2; i++) monoPcm[i] = 12000;
  for (let i = monoSr * 2; i < monoSr * 5; i++) monoPcm[i] = -25000;
  for (let i = monoSr * 5; i < monoSr * 8; i++) monoPcm[i] = 20000;

  const monoKept = resolveEffectiveRanges([{ startS: 2, endS: 5 }], monoDurS, "remove");
  const monoMontage = buildMontageInt16({
    pcm: monoPcm,
    sampleRate: monoSr,
    channels: 1,
    keptRanges: monoKept,
    declickMs: MONTAGE_DECLICK_MS,
  });
  // Exact straight montage duration: 2.0s + 3.0s = 5.000s (zero synthetic 400ms gap!)
  assert.ok(
    Math.abs(monoMontage.duration - 5.0) < 0.001,
    `Expected pure 5.000s mono montage, got ${monoMontage.duration}`
  );
  // Excluded region (-25000) must NEVER appear
  for (let i = 0; i < monoMontage.pcm.length; i++) {
    assert.ok(monoMontage.pcm[i] >= 0, `Excluded mono audio leaked at ${i}: ${monoMontage.pcm[i]}`);
  }
  // Seam at frame 2.0s: zero digital pop (0 -> 0 across the cut) and full +20000 by 6 ms after seam!
  const monoSeam = monoSr * 2;
  assert.equal(monoMontage.pcm[monoSeam - 1], 0);
  assert.equal(monoMontage.pcm[monoSeam], 0);
  assert.equal(monoMontage.pcm[monoSeam + Math.round(0.006 * monoSr)], 20000);
  // Untouched outer start (frame 0) and outer end (last frame) remain full amplitude (no unnecessary fade)
  assert.equal(monoMontage.pcm[0], 12000);
  assert.equal(monoMontage.pcm[monoMontage.frames - 1], 20000);

  // 7. Step 1 Pure Montage on STEREO PCM: exact interleaved L/R alignment, zero channel swap,
  //    zero synthetic gap, zero amplitude spikes/thumps at the seam
  const stSr = 16000;
  const stDurS = 7;
  const stChannels = 2;
  const stPcm = new Int16Array(stSr * stDurS * stChannels);
  for (let f = 0; f < stSr * stDurS; f++) {
    const off = f * stChannels;
    if (f < stSr * 2) {
      // Range 1 [0..2s]: L = +14000, R = -11000
      stPcm[off] = 14000;
      stPcm[off + 1] = -11000;
    } else if (f < stSr * 5) {
      // Excluded [2..5s]: L = -30000, R = +30000 (must never bleed in)
      stPcm[off] = -30000;
      stPcm[off + 1] = 30000;
    } else {
      // Range 2 [5..7s]: L = +18000, R = -21000
      stPcm[off] = 18000;
      stPcm[off + 1] = -21000;
    }
  }
  const stKept = resolveEffectiveRanges([{ startS: 2, endS: 5 }], stDurS, "remove");
  const stMontage = buildMontageInt16({
    pcm: stPcm,
    sampleRate: stSr,
    channels: stChannels,
    keptRanges: stKept,
    declickMs: MONTAGE_DECLICK_MS,
  });
  assert.equal(stMontage.frames, stSr * 4);
  assert.equal(stMontage.pcm.length, stSr * 4 * stChannels);
  assert.ok(Math.abs(stMontage.duration - 4.0) < 0.001);

  // Verify strict L/R channel alignment across entire stereo output (L >= 0, R <= 0 everywhere!)
  for (let f = 0; f < stMontage.frames; f++) {
    const l = stMontage.pcm[f * 2];
    const r = stMontage.pcm[f * 2 + 1];
    assert.ok(l >= 0 && l <= 18000, `Stereo L channel misaligned/spiked at frame ${f}: ${l}`);
    assert.ok(r <= 0 && r >= -21000, `Stereo R channel misaligned/spiked at frame ${f}: ${r}`);
  }
  // Verify smooth 5 ms seam de-click (zero step jump at seamFrame, full L/R by 6 ms)
  const stSeam = stSr * 2;
  assert.equal(stMontage.pcm[(stSeam - 1) * 2], 0);
  assert.equal(stMontage.pcm[(stSeam - 1) * 2 + 1], 0);
  assert.equal(stMontage.pcm[stSeam * 2], 0);
  assert.equal(stMontage.pcm[stSeam * 2 + 1], 0);
  const afterSeam = stSeam + Math.round(0.006 * stSr);
  assert.equal(stMontage.pcm[afterSeam * 2], 18000);
  assert.equal(stMontage.pcm[afterSeam * 2 + 1], -21000);
  // Max sample-to-sample step across the seam window must be smooth (< 1000), zero bang/thump
  const seamWin = Math.round(0.01 * stSr);
  for (let f = stSeam - seamWin; f < stSeam + seamWin; f++) {
    const dl = Math.abs(stMontage.pcm[f * 2] - stMontage.pcm[(f - 1) * 2]);
    const dr = Math.abs(stMontage.pcm[f * 2 + 1] - stMontage.pcm[(f - 1) * 2 + 1]);
    assert.ok(dl < 1000 && dr < 1000, `Sharp discontinuity at stereo seam frame ${f}: dl=${dl}, dr=${dr}`);
  }

  // 8. Step 2 Single-Stream VAD Decision: fallback to Step 1 pure montage on low_snr_guarded,
  //    empty segments, or no long silences found
  const noisyDecision = decidePrayerPostVad(
    { low_snr_guarded: true, segments: [{ start: 0, end: 4, duration: 4 }] },
    4.0
  );
  assert.deepEqual(noisyDecision, {
    skipSilenceTrim: true,
    silenceTrimSkipped: true,
    silenceTrimStatus: "skipped",
  });
  // Saved duration when skipped equals exact difference between original file and Step 1 pure montage
  assert.ok(Math.abs((stDurS - stMontage.duration) - 3.0) < 0.001);

  const emptyDecision = decidePrayerPostVad({ low_snr_guarded: false, segments: [] }, 4.0);
  assert.equal(emptyDecision.skipSilenceTrim, true);
  assert.equal(emptyDecision.silenceTrimSkipped, true);

  const noLongSilencesDecision = decidePrayerPostVad(
    { low_snr_guarded: false, segments: [{ start: 0.0, end: 4.0, duration: 4.0 }] },
    4.0
  );
  assert.equal(noLongSilencesDecision.skipSilenceTrim, true);
  assert.equal(noLongSilencesDecision.silenceTrimSkipped, true);

  // 9. Step 2 Single-Stream Silence Trimming on the Step 1 montage (clean audio with long silence)
  const cleanVadOnMontage = {
    low_snr_guarded: false,
    segments: [
      { start: 0.0, end: 1.5, duration: 1.5 },
      { start: 2.5, end: 4.0, duration: 1.5 },
    ],
    resolved_crossfade_ms: 0,
    resolved_gap_ms: 0,
    use_room_tone: false,
  };
  const trimmedDecision = decidePrayerPostVad(cleanVadOnMontage, stMontage.duration);
  assert.deepEqual(trimmedDecision, {
    skipSilenceTrim: false,
    silenceTrimSkipped: false,
    silenceTrimStatus: "trimmed",
  });
  const finalStereo = spliceRangesInt16({
    pcm: stMontage.pcm,
    sampleRate: stSr,
    channels: stChannels,
    rangeResults: [cleanVadOnMontage],
    fadeEdges: false,
  });
  // Original 7.0s -> Step 1 montage 4.0s (-3.0s) -> Step 2 VAD trimmed 3.0s (-1.0s) => 4.0s total saved
  assert.ok(Math.abs(finalStereo.duration - 3.0) < 0.002);
  assert.ok(Math.abs((stDurS - finalStereo.duration) - 4.0) < 0.002);
}

async function checkDevServer() {
  const server = createStaticServer(WEB_ROOT);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    const routes = [
      ["/", "text/html"],
      ["/styles.css", "text/css"],
      ["/app.js", "text/javascript"],
      ["/worker.js", "text/javascript"],
      ["/engine/vad.js", "text/javascript"],
      ["/engine/splicer.js", "text/javascript"],
      ["/manifest.json", "application/json"],
      ["/sw.js", "text/javascript"],
      ["/icons/crescent.svg", "image/svg+xml"],
      ["/icons/icon-192.png", "image/png"],
      ["/assets/og-preview.png", "image/png"],
    ];

    for (const [route, expectedMime] of routes) {
      const res = await fetch(`${baseUrl}${route}`);
      assert.equal(res.status, 200, `${route} returned HTTP ${res.status}`);
      assert.equal(res.headers.get("cross-origin-opener-policy"), "same-origin");
      assert.equal(res.headers.get("cross-origin-embedder-policy"), "require-corp");
      assert.ok(
        (res.headers.get("content-type") || "").includes(expectedMime),
        `${route} unexpected MIME: ${res.headers.get("content-type")}`
      );
    }

    const headMissing = await fetch(`${baseUrl}/assets/does-not-exist.mp3`, { method: "HEAD" });
    assert.equal(headMissing.status, 404);

    const traversal = await fetch(`${baseUrl}/../app.py`);
    assert.ok(traversal.status === 403 || traversal.status === 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

async function main() {
  checkHeadersFile();
  checkManifestAndAssets();
  checkI18nAndHtmlRules();
  checkPrayerModeLogic();
  await checkDevServer();
  console.log("ALL STAGE 2 SMOKE TESTS PASSED.");
}

main();
