# Quran Recitation Silence & Gap Remover (`quran-trim`)

A fast, modular Python CLI and bilingual (Arabic / English) Streamlit web app to automatically remove long silences, breaths, and prayer transitions from Quran recitation audio and video files — specifically engineered for vintage live recordings to preserve analog tape warmth, room character, and full dynamic range.

## Features & Architecture

- **Dual-Rate Zero-Loss Pipeline**: Decodes a temporary 16kHz mono stream (with an 80Hz high-pass filter for analysis only) for hybrid Silero VAD + vocal-band energy detection, then maps those exact timestamps back to slice and concatenate the **original high-sample-rate audio/video stream** (`soundfile` for `.wav -> .wav`, streaming FFmpeg for `.mp3`, `.m4a`, `.mp4`, `.mkv`, `.webm`, `.opus`, `.flac`).
- **Auto-Profiler**: Automatically distinguishes clean studio recordings (filtering close-mic inhalation breaths via high-frequency spectral balance and `>= 220ms` duration gating) from vintage/reverberant live recordings (keeping the SNR-aware energy gate active while rejecting short unconfirmed low-energy cough/thump noise blips so soft verse endings and Madd stay intact).
- **Untouched Dynamics**: Never applies denoising, EQ, loudness normalization, or dynamic range compression to the output stream.
- **Smooth Transitions & Breathing Room**: Applies asymmetric padding (`--pad-start 200`, `--pad-end 300` to protect Ghunnah and soft verse endings), a `20ms` equal-power crossfade at every splice point, and a clean `--gap-ms` (`450ms` default) room-tone breathing gap between stitched segments.
- **Modes & Clipping**:
  - `natural` *(default)*: Auto-profiled hybrid Silero VAD (`--threshold 0.20`) + energy gate — removes long silences while keeping soft verse endings, melodic Maqam sustain, and live audience reactions. Prints a hint if periodic prayer transitions are detected.
  - `prayer`: Detects periodic Ruku/Sujud windows (`>= 12s`) and Muballigh echo clusters (`14s–65s`), stripping opening and transition Takbirs while strictly preserving Al-Fatihah and Short Surahs.
  - `strict` *(advanced)*: Filters segments against an adaptive RMS cutoff anchored to the duration-weighted `p25` speech floor (`--strict-margin-db 12.0` default) to drop distant crowd shouts.
  - `--start` / `--end`: Optional `MM:SS` or `HH:MM:SS` timestamps to trim only a specific section (e.g., recitation before commentary).

## Setup

Requires `ffmpeg` and `ffprobe` on `PATH` (`packages.txt` provides `ffmpeg` automatically on Debian/Ubuntu/Streamlit Cloud).

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` uses PyTorch's CPU-only wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`) so cloud builds stay lightweight without multi-gigabyte CUDA dependencies.

## Web App Usage (Streamlit)

Launch the bilingual (Arabic-first / English) web interface locally:

```powershell
.\.venv\Scripts\streamlit run app.py
```

In the web app you can:
1. Upload local audio/video files (`mp3`, `wav`, `m4a`, `mp4`, `mkv`, `webm`, `opus`, `flac`) or paste a YouTube URL (with an automatic fallback prompt to upload locally if YouTube blocks datacenter IPs).
2. Choose **تلاوة طبيعية | Natural** or **صلاة جهرية | Prayer Mode**, optional `Start` / `End` (`MM:SS`) bounds, and export format (`m4a` or `mp3`).
3. Watch live progress, preview the cleaned audio inline, inspect the Original Duration / Final Duration / Time Saved % / Detected Profile cards, and download the cleaned audio and `timestamps.json`.

### Streamlit Community Cloud Deployment

1. Push this repository to GitHub (the root includes `app.py`, `requirements.txt`, `packages.txt`, and `.streamlit/config.toml`).
2. Create a new app on [share.streamlit.io](https://share.streamlit.io) pointing to `app.py`.
3. Streamlit Cloud automatically installs `ffmpeg` from `packages.txt` and CPU-only PyTorch dependencies from `requirements.txt`.

## CLI Usage

```powershell
# Default natural mode (auto-profiled)
.\.venv\Scripts\python quran_trim.py recitation.wav

# Prayer mode to strip Ruku/Sujud Takbirs and Muballigh echoes
.\.venv\Scripts\python quran_trim.py taraweeh.webm --mode prayer

# Extract and clean a specific range (e.g., first 5 minutes)
.\.venv\Scripts\python quran_trim.py tv_program.mp4 --start 00:00 --end 05:00

# Advanced strict mode for vintage live recordings with distant crowd shouts
.\.venv\Scripts\python quran_trim.py live_tape.mp3 --mode strict

# Custom VAD, padding, and breathing gap settings on video
.\.venv\Scripts\python quran_trim.py surah.mp4 --threshold 0.20 --pad-start 200 --pad-end 300 --min-silence 700 --gap-ms 450

# Download best audio from YouTube and clean it
.\.venv\Scripts\python quran_trim.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode natural
```

Outputs are saved next to the input as `<filename>_clean.<ext>` and `timestamps.json` (configurable via `-o` and `--timestamps`).

## Running Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
