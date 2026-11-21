# أنصتوا — Ansitu (`quran-trim`)

A fast, modular Python CLI and bilingual (Arabic-first RTL / English) Streamlit web app to automatically remove long silences, breaths, and isolated crowd murmurs from Quran recitation audio and video files — engineered for vintage live recordings to preserve analog tape warmth, room ambience, and full dynamic range without dynamic compression or denoising.

## Features & Architecture

- **Dual-Rate Zero-Loss Pipeline**: Decodes a temporary 16kHz mono stream (with an 80Hz high-pass filter for VAD analysis only) for hybrid Silero VAD + vocal-band energy detection, then maps exact timestamps back to slice and concatenate the **original high-sample-rate audio/video stream** (`soundfile` for `.wav -> .wav`, streaming FFmpeg for `.mp3`, `.m4a`, `.mp4`, `.mkv`, `.webm`, `.opus`, `.flac`).
- **100%-Safety Auto-Profiler & Vintage Reverb Overlap**:
  - **Low-SNR Safety Gate (`SNR < 12.0 dB`)**: Automatically pauses silence removal when background noise is too high relative to the reciter (`snr_db < 12.0`), returning the full audio untouched (`low_snr_guarded: true`) with a clear bilingual notice to protect Quranic letters.
  - **Studio** recordings reject close-mic inhalation breaths via high-frequency spectral balance and relax sustain down to `p90 - 26 dB` so soft 1960s studio passages stay intact (`700ms` min silence / `450ms` digital silence gap / `20ms` crossfade).
  - On **all `vintage/live`** recordings, uses a calm Mujawwad breathing cadence (`650ms` before speech, `1100ms` after speech or `1300ms` when ending `core_db >= reciter_p75`) to capture real decaying mosque reverb and pre-breath hall ambience, enforces a **`>= 1.2s` minimum post-padding effective cut guard** (merging shorter gaps instead of splicing for `0.3s–1.1s`), and overlaps retained segments with a `400ms` equal-power (sin/cos) crossfade (`gap = 0`, yielding `~1.35s`–`1.55s` natural pause) plus a gentle `150ms` start fade-in and `500ms` end fade-out.
  - **Ultra-Safe Crowd Murmur Filter (`vintage/live`)**: Requires `margin_db = max(10.0, min(14.0, snr_db * 0.40))` (`>= 10 dB` below reciter median) and `max_cluster_span <= 3.5s` inside long pauses (`gap_before >= 1.5s` and `gap_after >= 1.8s`), so soft/quiet verses are never dropped.
- **Single Ultra-Safe Mode, Multi-Range Clipping & Quality Selection**:
  - **Single Ultra-Safe Automatic Mode**: Auto-profiled hybrid Silero VAD (`--threshold 0.20`) + energy gate — no aggressive cutting modes exposed.
  - **Multi-Range Clipping & External URLs**: Clean and seamlessly stitch one or more `Start` / `End` (`MM:SS`) time ranges from long recordings or prayers (via uploaded files or external `http://` / `https://` links such as YouTube, SoundCloud, Archive.org, or direct audio URLs via `yt-dlp`), exporting at `128 kbps`, `192 kbps` *(default)*, `256 kbps`, or `Lossless WAV` *(when uploading `.wav` or `.flac`)*.

## Setup

Requires `ffmpeg` and `ffprobe` on `PATH` (`packages.txt` installs `ffmpeg` automatically on Debian/Ubuntu/Streamlit Cloud).

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` uses PyTorch's CPU-only wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`) so cloud deployments stay lightweight.

## Web App Usage (أنصتوا | Ansitu)

Launch the bilingual Streamlit app locally:

```powershell
.\.venv\Scripts\streamlit run app.py
```

In **أنصتوا — Ansitu** you can:
1. **📁 رفع ملف صوتي أو فيديو | Upload Audio / Video File** (`mp3`, `wav`, `m4a`, `mp4`, `mkv`, `webm`, `opus`, `flac`) or switch to **🔗 رابط خارجي | External Media URL** (`http://` / `https://` links from YouTube, SoundCloud, Archive.org, or direct media URLs).
2. Select output format (`m4a` / `mp3`), quality (`128 kbps`, `192 kbps`, `256 kbps`, plus `Lossless WAV` for `.wav`/`.flac` uploads), and add one or more optional `Start` / `End` clip ranges (`➕ أضف مقطعاً آخر | + Add Range`) to extract and stitch desired passages cleanly.
3. Monitor live progress, preview the cleaned recitation inline, inspect metrics, and download the output + `timestamps.json`.

### Streamlit Community Cloud Deployment

1. Push this repository to GitHub (`app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`).
2. Create an app on [share.streamlit.io](https://share.streamlit.io) pointing to `app.py`.

## CLI Usage

```powershell
# Default ultra-safe automatic mode (auto-profiled)
.\.venv\Scripts\python quran_trim.py recitation.wav

# Extract and clean a specific range (e.g., first 5 minutes)
.\.venv\Scripts\python quran_trim.py tv_program.mp4 --start 00:00 --end 05:00

# Download best audio from any supported URL (YouTube, SoundCloud, Archive.org, direct MP3/M4A)
.\.venv\Scripts\python quran_trim.py "https://archive.org/details/example_recitation"
```

Outputs are saved next to the input as `<filename>_clean.<ext>` and `timestamps.json` (configurable via `-o` and `--timestamps`).

## Running Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
