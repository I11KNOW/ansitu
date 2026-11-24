# أَنصِتُوا — ansitu

**Live Web App (PWA):** [https://ansitu.pages.dev](https://ansitu.pages.dev)

A calm, bilingual (Arabic RTL / English LTR) client-side web app that automatically removes long pauses between Quranic verses while preserving every letter, breath, and natural hall ambience of the recitation.

---

## Web App (`web/`) — 100% Client-Side PWA

Everything runs locally inside the user's browser — **no server uploads, no backend processing, and full offline support** once installed as a PWA.

### Privacy-First Architecture
- **Pure JS Energy DSP VAD (`web/engine/vad.js`)**: Zero-dependency ES module port of the acoustic profiler and voice activity engine — engineered for both studio and vintage live mosque recordings without heavy neural/ONNX downloads.
- **Dedicated Web Worker (`web/worker.js`)**: Keeps the UI responsive during decoding, analysis, splicing, and encoding, with instant cancellation and automatic WASM memory reclamation when each job finishes.
- **Lazy-Loaded `ffmpeg-core.wasm`**: Not pre-cached or fetched on initial page load; loaded on demand only when the user starts processing (multi-threaded `@ffmpeg/core-mt` when `crossOriginIsolated`, with single-threaded fallback) and runtime-cached by the Service Worker after first use.
- **Featherweight Initial Load**: **~90 KB uncompressed (~30 KB gzipped)** initial payload with zero external font stylesheets or render-blocking scripts, backed by an offline PWA Service Worker (`web/sw.js`).
- **Native-Rate `Int16Array` Splicing & Headroom Preservation (`web/engine/splicer.js`)**: Splices at the original sample rate in `Int16Array` (50% less mobile RAM than `Float32Array`), applies `-6 dB` decode headroom for lossy inputs to prevent clipping hot masters, and preserves basic container text metadata (`title`, `artist`, `album`).
- **Export Quality Presets**: `128 kbps`, `192 kbps` *(default)*, `256 kbps` in `M4A` or `MP3`, plus automatic `Lossless WAV` for `.wav` / `.flac` sources.

### Two Recitation Modes
1. **Normal Recitation (`تلاوة عادية`)**:
   - Auto-profiles studio vs. vintage/live recordings, preserves natural Mujawwad breathing cadence and reverb tails, and enforces the **Low-SNR safety gate (`SNR < 12 dB`)** — pausing automatic silence removal when background noise is too high so zero Quranic letters are ever clipped.
2. **Prayer Mode (`تلاوة صلاة`)**:
   - Includes a **Remove vs. Keep ranges** toggle (`حذف هذه المقاطع` / `الاحتفاظ بهذه المقاطع فقط`) with mobile-friendly numeric timestamp inputs (`00:45`, `00.45`, `0045`, `130`, `10530`).
   - Uses a clean **2-step pipeline**:
     1. **Step 1 — Pure Montage (`buildMontageInt16`)**: Resolves kept ranges and concatenates the decoded PCM slices directly in order with strict mono/stereo frame alignment and a 5 ms de-click micro-fade at raw cut boundaries (zero overlapping crossfade, zero synthetic gap).
     2. **Step 2 — Single-Stream Silence Removal**: Feeds the concatenated montage into the exact same single-stream Normal Mode VAD + splicer pipeline. If `low_snr_guarded` triggers or no long silences are found, exports the Step 1 pure montage directly with a gentle notice so the user's prayer montage is never discarded.

### Local Development & Verification

Requires Node.js 18+.

```bash
# Start local dev server with COOP/COEP headers (http://127.0.0.1:4173)
node web/dev-server.mjs

# Run automated Stage 2 smoke & Prayer Mode unit tests
node web/tests/smoke.mjs

# Run Python-vs-JS DSP parity verification (requires Python venv below)
node web/tests/parity.mjs
```

---

## Legacy Python CLI & Streamlit App

The original Python engine (`quran_cleaner/`), CLI (`quran_trim.py`), and Streamlit interface (`app.py`) remain in this repository for desktop/CLI batch workflows, video inputs, and external URL downloads (`yt-dlp`).

### Setup

Requires `ffmpeg` and `ffprobe` on `PATH` (`packages.txt` installs `ffmpeg` automatically on Debian/Ubuntu/Streamlit Cloud).

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` uses PyTorch's CPU-only wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`) so cloud deployments stay lightweight.

### Streamlit Web App

```powershell
.\.venv\Scripts\streamlit run app.py
```

Supports uploading audio/video files (`mp3`, `wav`, `m4a`, `mp4`, `mkv`, `webm`, `opus`, `flac`) or pasting external media URLs (`http://` / `https://` via `yt-dlp`), selecting `128` / `192` / `256 kbps` or `Lossless WAV`, and clipping optional `Start` / `End` ranges.

### Python CLI Usage

```powershell
# Default ultra-safe automatic mode (auto-profiled)
.\.venv\Scripts\python quran_trim.py recitation.wav

# Extract and clean a specific range (e.g., first 5 minutes)
.\.venv\Scripts\python quran_trim.py tv_program.mp4 --start 00:00 --end 05:00

# Download best audio from any supported URL (YouTube, SoundCloud, Archive.org, direct MP3/M4A)
.\.venv\Scripts\python quran_trim.py "https://archive.org/details/example_recitation"
```

Outputs are saved next to the input as `<filename>_clean.<ext>` and `timestamps.json` (configurable via `-o` and `--timestamps`).

### Python Unit Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
