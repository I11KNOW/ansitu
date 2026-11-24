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
  applyPrayerModeFallback,
  resolveEffectiveRanges,
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
  assert.throws(() => parseTimestamp("bad:time"));
  assert.throws(() => parseTimestamp("01:75"));

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

  // 6. Normal Mode: low_snr_guarded aborts cleanly (unchanged)
  const guardedRes = [
    {
      segments: [{ start: 0, end: 120, duration: 120 }],
      total_duration: 120,
      low_snr_guarded: true,
      resolved_crossfade_ms: 0,
      resolved_gap_ms: 0,
    },
  ];
  const normalDecision = applyPrayerModeFallback(
    guardedRes,
    [{ startS: 0, endS: 120 }],
    { isPrayerMode: false, hasUserRanges: false }
  );
  assert.equal(normalDecision.action, "abort_low_snr");
  assert.equal(normalDecision.silenceTrimSkipped, false);

  // 7. Prayer Mode with user ranges: NEVER discards montage on low_snr_guarded or empty VAD
  const prayerRanges = [
    { startS: 0, endS: 60 },
    { startS: 120, endS: 300 },
  ];
  const prayerVad = [
    {
      segments: [{ start: 0, end: 60, duration: 60 }],
      total_duration: 60,
      low_snr_guarded: true,
      resolved_crossfade_ms: 0,
      resolved_gap_ms: 0,
    },
    {
      segments: [],
      total_duration: 180,
      low_snr_guarded: false,
      resolved_crossfade_ms: 0,
      resolved_gap_ms: 0,
    },
  ];
  const prayerDecision = applyPrayerModeFallback(prayerVad, prayerRanges, {
    isPrayerMode: true,
    hasUserRanges: true,
  });
  assert.equal(prayerDecision.action, "splice");
  assert.equal(prayerDecision.silenceTrimSkipped, true);
  assert.equal(prayerDecision.rangeResults.length, 2);
  assert.deepEqual(prayerDecision.rangeResults[0].segments, [
    { start: 0, end: 60, duration: 60 },
  ]);
  assert.deepEqual(prayerDecision.rangeResults[1].segments, [
    { start: 120, end: 300, duration: 180 },
  ]);
  assert.equal(prayerDecision.rangeResults[0].resolved_crossfade_ms, 80);

  // 8. Prayer Mode with clean speech segments does NOT show skipped notice
  const cleanPrayerVad = [
    {
      segments: [{ start: 1, end: 59, duration: 58 }],
      total_duration: 60,
      low_snr_guarded: false,
      resolved_crossfade_ms: 100,
      resolved_gap_ms: 100,
    },
  ];
  const cleanDecision = applyPrayerModeFallback(
    cleanPrayerVad,
    [{ startS: 0, endS: 60 }],
    { isPrayerMode: true, hasUserRanges: true }
  );
  assert.equal(cleanDecision.action, "splice");
  assert.equal(cleanDecision.silenceTrimSkipped, false);
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
