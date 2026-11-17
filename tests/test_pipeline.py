"""End-to-end verification suite for the Quran recitation silence remover."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quran_cleaner import cli
from quran_cleaner.media import extract_vad_audio, is_youtube_url, parse_timestamp, probe_media
from quran_cleaner.vad import VADConfig, detect_speech_segments

CLI_SCRIPT = PROJECT_ROOT / "quran_trim.py"


def _synthesize_sapi_wav(text: str, out_wav: Path, rate: int = -2) -> None:
    """Synthesize spoken audio using Windows SAPI SpeechSynthesizer."""
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{str(out_wav).replace(chr(39), chr(39)*2)}'); "
        f"$s.Speak('{text.replace(chr(39), chr(39)*2)}'); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        capture_output=True,
        text=True,
    )


def _resample_to_48k_mono(wav_path: Path, target_sr: int = 48000) -> np.ndarray:
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(wav_path),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def _trim_silence(sig: np.ndarray, thresh: float = 0.005) -> np.ndarray:
    idx = np.flatnonzero(np.abs(sig) > thresh)
    if idx.size == 0:
        return sig
    return sig[idx[0] : idx[-1] + 1]


def _normalize_rms(sig: np.ndarray, target_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
    target_rms = 10.0 ** (target_db / 20.0)
    return sig * (target_rms / rms)


def _build_vintage_live_fixture(work_dir: Path) -> tuple[Path, dict[str, tuple[int, int]]]:
    """Create a 48kHz 24-bit stereo WAV simulating a vintage live recitation recording.

    Layout:
      - 1.0s analog tape floor
      - Verse 1 (-14 dBFS close-mic + soft 250ms tail)
      - 1.8s silence gap
      - Distant crowd reaction (-31 dBFS, ~17 dB quieter than reciter)
      - 1.8s silence gap
      - Verse 2a (-14 dBFS) + 0.35s internal pause (<700ms min_silence) + Verse 2b (-14 dBFS)
      - 1.5s trailing silence
    """
    sr = 48000
    p1_raw = work_dir / "sapi_p1.wav"
    crowd_raw = work_dir / "sapi_crowd.wav"
    p2a_raw = work_dir / "sapi_p2a.wav"
    p2b_raw = work_dir / "sapi_p2b.wav"

    _synthesize_sapi_wav("Bismillahi rahmani rahim alhamdu lillahi rabbil alamin", p1_raw, rate=-2)
    _synthesize_sapi_wav("Allah ya salam", crowd_raw, rate=0)
    _synthesize_sapi_wav("Ar rahmani rahim", p2a_raw, rate=-2)
    _synthesize_sapi_wav("Maliki yawmid din", p2b_raw, rate=-2)

    p1 = _normalize_rms(_trim_silence(_resample_to_48k_mono(p1_raw, sr)), -16.0)
    # Add a soft trailing Ghunnah-like decay at the end of Verse 1
    tail_len = int(0.25 * sr)
    if p1.size > tail_len:
        p1[-tail_len:] *= np.linspace(1.0, 0.12, tail_len)

    crowd = _normalize_rms(_trim_silence(_resample_to_48k_mono(crowd_raw, sr)), -33.0)
    p2a = _normalize_rms(_trim_silence(_resample_to_48k_mono(p2a_raw, sr)), -16.0)
    p2b = _normalize_rms(_trim_silence(_resample_to_48k_mono(p2b_raw, sr)), -16.0)

    parts: list[np.ndarray] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0

    def _add(name: str | None, arr: np.ndarray) -> None:
        nonlocal cursor
        if name:
            spans[name] = (cursor, cursor + arr.size)
        parts.append(arr)
        cursor += arr.size

    _add(None, np.zeros(int(1.0 * sr), dtype=np.float64))
    _add("verse1", p1)
    _add(None, np.zeros(int(1.8 * sr), dtype=np.float64))
    _add("crowd", crowd)
    _add(None, np.zeros(int(1.8 * sr), dtype=np.float64))
    _add("verse2a", p2a)
    _add(None, np.zeros(int(0.35 * sr), dtype=np.float64))
    _add("verse2b", p2b)
    _add(None, np.zeros(int(1.5 * sr), dtype=np.float64))

    mono = np.concatenate(parts)

    # Mix subtle continuous analog tape hiss (-50 dBFS pink-ish lowpass noise)
    rng = np.random.default_rng(42)
    white = rng.standard_normal(mono.size)
    tape_noise = np.convolve(white, np.ones(8) / 8.0, mode="same")
    tape_noise = _normalize_rms(tape_noise, -50.0)
    mono = np.clip(mono + tape_noise, -0.98, 0.98)

    # Make genuine stereo (tiny L/R room phase micro-difference)
    stereo = np.stack([mono, mono * 0.995], axis=1)
    fixture_wav = work_dir / "recitation_live.wav"
    sf.write(str(fixture_wav), stereo, sr, subtype="PCM_24")
    return fixture_wav, spans


class EndToEndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="quran_e2e_")
        cls.work_dir = Path(cls._tmp.name)
        cls.fixture_wav, cls.spans = _build_vintage_live_fixture(cls.work_dir)
        cls.orig_audio, cls.orig_sr = sf.read(str(cls.fixture_wav), dtype="float64")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI_SCRIPT), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def test_cli_help_and_error_handling(self) -> None:
        res_help = self._run_cli("--help")
        self.assertEqual(res_help.returncode, 0, res_help.stderr)
        for flag in (
            "--mode",
            "prayer",
            "--threshold",
            "--pad-start",
            "--pad-end",
            "--min-silence",
            "--gap-ms",
            "--start",
            "--end",
        ):
            self.assertIn(flag, res_help.stdout)

        self.assertEqual(parse_timestamp("05:00"), 300.0)
        self.assertEqual(parse_timestamp("01:02:03"), 3723.0)
        with self.assertRaises(ValueError):
            parse_timestamp("bad:time")
        with self.assertRaises(ValueError):
            parse_timestamp("01:75")

        res_range = self._run_cli(str(self.fixture_wav), "--start", "00:10", "--end", "00:05")
        self.assertNotEqual(res_range.returncode, 0)
        self.assertIn("--start", res_range.stderr.lower())

        res_missing = self._run_cli(str(self.work_dir / "does_not_exist.wav"))
        self.assertNotEqual(res_missing.returncode, 0)
        self.assertIn("not found", res_missing.stderr.lower())

        bad_ext = self.work_dir / "notes.txt"
        bad_ext.write_text("hello", encoding="utf-8")
        res_ext = self._run_cli(str(bad_ext))
        self.assertNotEqual(res_ext.returncode, 0)
        self.assertIn("unsupported file format", res_ext.stderr.lower())

        res_thresh = self._run_cli(str(self.fixture_wav), "--threshold", "1.5")
        self.assertNotEqual(res_thresh.returncode, 0)
        self.assertIn("threshold", res_thresh.stderr.lower())

    def test_wav_natural_mode_zero_loss_and_crossfade(self) -> None:
        out_dir = self.work_dir / "wav_natural"
        out_dir.mkdir(exist_ok=True)
        src = out_dir / "sample.wav"
        src.write_bytes(self.fixture_wav.read_bytes())

        res = self._run_cli(str(src))
        self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)

        clean_wav = out_dir / "sample_clean.wav"
        ts_json = out_dir / "timestamps.json"
        self.assertTrue(clean_wav.is_file())
        self.assertTrue(ts_json.is_file())

        # Check terminal summary labels
        for label in (
            "Original Duration",
            "Final Duration",
            "Time Saved (%)",
            "Total Segments Kept",
        ):
            self.assertIn(label, res.stdout)

        # Verify zero-quality-loss audio properties preserved
        info = sf.info(str(clean_wav))
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.subtype, "PCM_24")

        data = json.loads(ts_json.read_text(encoding="utf-8"))
        self.assertEqual(data["mode"], "natural")
        self.assertEqual(data["settings"]["gap_ms"], 450)
        # Should keep 3 segments: verse1, crowd shout, and verse2 (2a+350ms+2b bridged)
        self.assertEqual(data["summary"]["segments_kept"], 3)
        expected_dur = sum(s["duration_s"] for s in data["segments"]) + 2 * 0.450
        self.assertAlmostEqual(data["summary"]["final_duration_s"], expected_dur, places=2)
        self.assertGreater(data["summary"]["time_saved_pct"], 20.0)

        # Verify untouched core samples in verse1 match original PCM_24 values without denoising/compression
        clean_audio, _ = sf.read(str(clean_wav), dtype="float64")
        seg0 = data["segments"][0]
        s0_orig = int(round(seg0["start_s"] * 48000))
        core_offset = int(0.35 * 48000)
        core_len = int(0.8 * 48000)
        diff = np.max(
            np.abs(
                clean_audio[core_offset : core_offset + core_len]
                - self.orig_audio[s0_orig + core_offset : s0_orig + core_offset + core_len]
            )
        )
        self.assertLess(diff, 2e-6)

        # Verify 450ms room-tone breathing gap between segment 0 and segment 1 is smooth & low-level
        gap_start = int(round(seg0["duration_s"] * 48000))
        gap_mid = clean_audio[gap_start + int(0.05 * 48000) : gap_start + int(0.40 * 48000), 0]
        gap_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(gap_mid * gap_mid) + 1e-12)))
        self.assertLess(gap_rms_db, -44.0)

        # Verify smooth splice transitions across both boundaries of the inserted gap
        window = clean_audio[gap_start - 2000 : gap_start + int(0.45 * 48000) + 2000, 0]
        max_step = float(np.max(np.abs(np.diff(window))))
        self.assertLess(max_step, 0.05)

    def test_wav_strict_mode_filters_crowd(self) -> None:
        out_dir = self.work_dir / "wav_strict"
        out_dir.mkdir(exist_ok=True)
        src = out_dir / "live.wav"
        src.write_bytes(self.fixture_wav.read_bytes())

        res = self._run_cli(str(src), "--mode", "strict")
        self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)

        clean_wav = out_dir / "live_clean.wav"
        ts_json = out_dir / "timestamps.json"
        self.assertTrue(clean_wav.is_file())
        self.assertTrue(ts_json.is_file())

        data = json.loads(ts_json.read_text(encoding="utf-8"))
        self.assertEqual(data["mode"], "strict")
        self.assertIn("calibration", data)
        # Crowd shout (~-31 dBFS) filtered out; only close-mic verse1 & verse2 kept
        self.assertEqual(data["summary"]["segments_kept"], 2)

        crowd_start_s = self.spans["crowd"][0] / 48000.0
        crowd_end_s = self.spans["crowd"][1] / 48000.0
        for seg in data["segments"]:
            # Ensure crowd center isn't inside any kept segment
            crowd_mid = 0.5 * (crowd_start_s + crowd_end_s)
            self.assertFalse(seg["start_s"] <= crowd_mid <= seg["end_s"])

    def test_mp3_and_m4a_preserve_rate_and_metadata(self) -> None:
        mp3_dir = self.work_dir / "mp3_test"
        mp3_dir.mkdir(exist_ok=True)
        cover_jpg = mp3_dir / "cover.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:d=1",
                "-frames:v",
                "1",
                str(cover_jpg),
            ],
            check=True,
        )
        src_mp3 = mp3_dir / "vintage.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(self.fixture_wav),
                "-i",
                str(cover_jpg),
                "-map",
                "0:a",
                "-map",
                "1:v",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "256k",
                "-c:v",
                "mjpeg",
                "-disposition:v:0",
                "attached_pic",
                "-metadata",
                "title=Surah Al-Fatiha Live",
                str(src_mp3),
            ],
            check=True,
        )

        res_mp3 = self._run_cli(str(src_mp3), "--mode", "strict")
        self.assertEqual(res_mp3.returncode, 0, res_mp3.stderr + "\n" + res_mp3.stdout)
        clean_mp3 = mp3_dir / "vintage_clean.mp3"
        self.assertTrue(clean_mp3.is_file())
        info_mp3 = probe_media(clean_mp3)
        self.assertEqual(info_mp3.sample_rate, 48000)
        self.assertEqual(info_mp3.channels, 2)
        self.assertTrue(info_mp3.has_attached_pic)
        self.assertFalse(info_mp3.has_video)

        tags_out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=title",
                "-of",
                "default=nw=1",
                str(clean_mp3),
            ],
            text=True,
        )
        self.assertIn("Surah Al-Fatiha Live", tags_out)

        m4a_dir = self.work_dir / "m4a_test"
        m4a_dir.mkdir(exist_ok=True)
        src_m4a = m4a_dir / "vintage.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(self.fixture_wav),
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                str(src_m4a),
            ],
            check=True,
        )
        res_m4a = self._run_cli(str(src_m4a))
        self.assertEqual(res_m4a.returncode, 0, res_m4a.stderr + "\n" + res_m4a.stdout)
        clean_m4a = m4a_dir / "vintage_clean.m4a"
        self.assertTrue(clean_m4a.is_file())
        info_m4a = probe_media(clean_m4a)
        self.assertEqual(info_m4a.sample_rate, 48000)
        self.assertEqual(info_m4a.channels, 2)
        self.assertLess(info_m4a.duration, self.orig_audio.shape[0] / 48000.0 * 0.85)

    def test_mp4_and_mkv_video_av_sync(self) -> None:
        total_s = self.orig_audio.shape[0] / 48000.0
        for ext, fps in ((".mp4", "30"), (".mkv", "25")):
            vdir = self.work_dir / f"vid_{ext[1:]}"
            vdir.mkdir(exist_ok=True)
            src_vid = vdir / f"recitation{ext}"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=duration={total_s:.3f}:size=320x180:rate={fps}",
                    "-i",
                    str(self.fixture_wav),
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(src_vid),
                ],
                check=True,
            )

            res = self._run_cli(str(src_vid), "--mode", "strict")
            self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)

            clean_vid = vdir / f"recitation_clean{ext}"
            self.assertTrue(clean_vid.is_file())
            self.assertTrue((vdir / "timestamps.json").is_file())

            streams_json = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-print_format",
                        "json",
                        "-show_streams",
                        str(clean_vid),
                    ],
                    text=True,
                )
            )
            v_stream = next(s for s in streams_json["streams"] if s["codec_type"] == "video")
            a_stream = next(s for s in streams_json["streams"] if s["codec_type"] == "audio")
            self.assertEqual(int(a_stream["sample_rate"]), 48000)
            self.assertEqual(int(a_stream["channels"]), 2)

            v_dur = float(v_stream.get("duration") or probe_media(clean_vid).duration)
            a_dur = float(a_stream.get("duration") or probe_media(clean_vid).duration)
            self.assertLess(abs(v_dur - a_dur), 0.04, f"{ext} A/V drift: v={v_dur}, a={a_dur}")
            self.assertLess(v_dur, total_s * 0.85)

    def test_youtube_url_detection_and_cli_flow(self) -> None:
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_youtube_url("https://youtu.be/abc123XYZ"))
        self.assertTrue(is_youtube_url("https://music.youtube.com/watch?v=abc123"))
        self.assertFalse(is_youtube_url("https://example.com/audio.mp3"))
        self.assertFalse(is_youtube_url(str(self.fixture_wav)))

        yt_dir = self.work_dir / "yt_mock"
        yt_dir.mkdir(exist_ok=True)
        fake_downloaded = yt_dir / "Sheikh Recitation [test123].m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(self.fixture_wav),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(fake_downloaded),
            ],
            check=True,
        )

        with patch("quran_cleaner.cli.download_youtube_audio", return_value=fake_downloaded) as mock_dl:
            rc = cli.main(
                [
                    "https://www.youtube.com/watch?v=test123",
                    "--mode",
                    "natural",
                    "--output-dir",
                    str(yt_dir),
                ]
            )
        self.assertEqual(rc, 0)
        mock_dl.assert_called_once()
        self.assertTrue((yt_dir / "Sheikh Recitation [test123]_clean.m4a").is_file())
        self.assertTrue((yt_dir / "timestamps.json").is_file())

    def test_studio_auto_profiler_and_breath_rejection(self) -> None:
        sr = 16000
        rng = np.random.default_rng(7)
        sapi_path = self.work_dir / "studio_sapi.wav"
        _synthesize_sapi_wav("Bismillahi ar rahmani ar rahim", sapi_path, rate=-1)
        speech = _normalize_rms(_trim_silence(_resample_to_48k_mono(sapi_path, sr)), -14.0)

        # Synthetic high-frequency inhalation breath (300ms, >2.8kHz noise burst at -34 dBFS)
        raw_noise = rng.standard_normal(int(0.30 * sr))
        fft = np.fft.rfft(raw_noise)
        freqs = np.fft.rfftfreq(raw_noise.size, 1.0 / sr)
        fft[freqs < 3000.0] = 0.0
        breath = _normalize_rms(np.fft.irfft(fft, n=raw_noise.size), -34.0)

        # Clean studio noise floor (< -70 dBFS)
        silence_1 = np.zeros(int(1.2 * sr), dtype=np.float64)
        silence_2 = np.zeros(int(1.2 * sr), dtype=np.float64)
        silence_3 = np.zeros(int(1.2 * sr), dtype=np.float64)
        studio_arr = np.concatenate([silence_1, speech, silence_2, breath, silence_3, speech, silence_1])
        studio_arr += _normalize_rms(rng.standard_normal(studio_arr.size), -72.0)

        studio_wav = self.work_dir / "studio_breath.wav"
        sf.write(str(studio_wav), studio_arr, sr, subtype="PCM_16")

        vad_audio = extract_vad_audio(studio_wav)
        res = detect_speech_segments(vad_audio, VADConfig(mode="natural"), exact_duration=studio_arr.size / sr)
        self.assertEqual(res.profile.kind, "studio")
        self.assertLess(res.profile.noise_floor_p5_db, -55.0)
        # Only the 2 speech blocks kept; the 300ms high-frequency breath in the middle is rejected
        self.assertEqual(len(res.segments), 2)

    def test_prayer_mode_strips_takbirs_and_preserves_short_surah(self) -> None:
        # Construct realistic prayer timeline in 16kHz VAD space:
        # [0..24s] Rak'ah 1 Al-Fatihah + 1.5s Ameen + [25.5..39s] Short Surah
        # [39..84s] 45s Ruku/Sujud with 3 isolated 1.5s Takbirs (gaps 8s, 10s, 12s)
        # [84..110s] Rak'ah 2 Al-Fatihah
        sr = 16000
        rng = np.random.default_rng(11)
        dur = 112.0
        arr = _normalize_rms(rng.standard_normal(int(dur * sr)), -40.0)

        def _fill_speech(t0: float, t1: float) -> None:
            s0, s1 = int(t0 * sr), int(t1 * sr)
            t = np.arange(s1 - s0) / float(sr)
            # Formant-like voiced harmonic stack so energy & duration represent real vocalization
            sig = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 540 * t)
            arr[s0:s1] += _normalize_rms(sig, -12.0)

        _fill_speech(1.0, 23.0)   # Al-Fatihah
        _fill_speech(24.5, 38.0)  # Short Surah right after Al-Fatihah
        _fill_speech(44.0, 45.6)  # Ruku Takbir
        _fill_speech(56.0, 57.8)  # Sami'allahu + Muballigh
        _fill_speech(70.0, 71.5)  # Sujud Takbir
        _fill_speech(84.0, 109.0) # Rak'ah 2

        nat = detect_speech_segments(arr.astype(np.float32), VADConfig(mode="natural"), exact_duration=dur)
        self.assertGreaterEqual(nat.prayer_windows_count, 1)
        self.assertGreaterEqual(len(nat.segments), 5)

        pry = detect_speech_segments(arr.astype(np.float32), VADConfig(mode="prayer"), exact_duration=dur)
        self.assertEqual(pry.prayer_stripped_count, 3)
        # Rak'ah 1 (Al-Fatihah + Short Surah) + Rak'ah 2 preserved; all 3 Takbirs stripped
        self.assertEqual(len(pry.segments), 3)

    def test_clip_range_start_end(self) -> None:
        out_dir = self.work_dir / "clip_test"
        out_dir.mkdir(exist_ok=True)
        src = out_dir / "full.wav"
        src.write_bytes(self.fixture_wav.read_bytes())

        res = self._run_cli(str(src), "--start", "00:08", "--end", "00:14")
        self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)
        data = json.loads((out_dir / "timestamps.json").read_text(encoding="utf-8"))
        self.assertEqual(data["settings"]["clip_start_s"], 8.0)
        self.assertEqual(data["settings"]["clip_end_s"], 14.0)
        self.assertAlmostEqual(data["summary"]["original_duration_s"], 6.0, places=2)
        for seg in data["segments"]:
            self.assertGreaterEqual(seg["start_s"], 8.0)
            self.assertLessEqual(seg["end_s"], 14.0)

    def test_vintage_unconfirmed_noise_blips_filtered(self) -> None:
        sr = 16000
        rng = np.random.default_rng(19)
        sapi_path = self.work_dir / "vintage_sapi.wav"
        _synthesize_sapi_wav("Bismillahi ar rahmani ar rahim", sapi_path, rate=-2)
        speech = _normalize_rms(_trim_silence(_resample_to_48k_mono(sapi_path, sr)), -14.0)

        # Short 0.6s low-energy unpitched cough/rustle burst (-31 dBFS) missed by Silero VAD
        # but above the vintage energy gate onset (-34 dBFS), so _filter_unconfirmed_noise_blips drops it.
        raw_bump = rng.standard_normal(int(0.6 * sr))
        bump_fft = np.fft.rfft(raw_bump)
        freqs = np.fft.rfftfreq(raw_bump.size, 1.0 / sr)
        bump_fft[(freqs < 2500.0) | (freqs > 6000.0)] = 0.0
        cough_like = _normalize_rms(np.fft.irfft(bump_fft, n=raw_bump.size), -31.0)

        gap = np.zeros(int(1.6 * sr), dtype=np.float64)
        arr = np.concatenate([gap, speech, gap, cough_like, gap, speech, gap])
        arr += _normalize_rms(rng.standard_normal(arr.size), -49.0)

        res = detect_speech_segments(arr.astype(np.float32), VADConfig(mode="natural"))
        self.assertEqual(res.profile.kind, "vintage/live")
        # Both speech blocks are kept while the short low-energy cough/noise blip is dropped
        self.assertEqual(len(res.segments), 2)

    def test_prayer_mode_opening_takbir_and_muballigh_cluster(self) -> None:
        sr = 16000
        rng = np.random.default_rng(23)
        dur = 106.0
        arr = _normalize_rms(rng.standard_normal(int(dur * sr)), -40.0)

        def _fill_speech(t0: float, t1: float) -> None:
            s0, s1 = int(t0 * sr), int(t1 * sr)
            t = np.arange(s1 - s0) / float(sr)
            sig = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 540 * t)
            arr[s0:s1] += _normalize_rms(sig, -12.0)

        _fill_speech(0.5, 2.2)    # Opening Takbir (< 5.0s at < 3.0s start, 4.0s gap before Rak'ah 1)
        _fill_speech(6.2, 35.0)   # Rak'ah 1 (>= 25s)
        # Dense Muballigh cluster in 35s span (35..70s): 26s speech (silence_ratio < 0.35), all <= 12.0s
        _fill_speech(36.0, 46.5)  # 10.5s Imam + Muballigh Takbir
        _fill_speech(47.5, 56.5)  # 9.0s Imam + Muballigh Sami'allahu
        _fill_speech(57.5, 64.0)  # 6.5s Sujud Takbir
        _fill_speech(70.0, 102.0) # Rak'ah 2 (>= 25s)

        nat = detect_speech_segments(arr.astype(np.float32), VADConfig(mode="natural"), exact_duration=dur)
        self.assertGreaterEqual(nat.prayer_windows_count, 1)
        self.assertEqual(nat.prayer_stripped_count, 0)
        self.assertEqual(len(nat.segments), 6)

        pry = detect_speech_segments(arr.astype(np.float32), VADConfig(mode="prayer"), exact_duration=dur)
        self.assertEqual(pry.prayer_stripped_count, 4)
        self.assertEqual(len(pry.segments), 2)

    def test_web_app_cleaning_job_and_youtube_fallback(self) -> None:
        import app as web_app

        # 1. Uploaded WAV exported directly as playable M4A + live progress checks
        progress_events: list[tuple[int, str]] = []
        res_m4a = web_app.run_cleaning_job(
            mode="natural",
            start_text="",
            end_text="",
            out_format="m4a",
            uploaded_name="vintage.wav",
            uploaded_bytes=self.fixture_wav.read_bytes(),
            progress_cb=lambda pct, msg: progress_events.append((pct, msg)),
        )
        self.assertEqual(res_m4a["output_filename"], "vintage_clean.m4a")
        self.assertGreater(len(res_m4a["audio_bytes"]), 1000)
        self.assertGreaterEqual(len(progress_events), 4)
        self.assertEqual(progress_events[-1][0], 100)
        self.assertEqual(res_m4a["segments_kept"], 3)
        self.assertEqual(res_m4a["profile_kind"], "vintage/live")

        m4a_probe_path = self.work_dir / "web_out.m4a"
        m4a_probe_path.write_bytes(res_m4a["audio_bytes"])
        m4a_info = probe_media(m4a_probe_path)
        self.assertEqual(m4a_info.audio_codec, "aac")
        self.assertEqual(m4a_info.sample_rate, 48000)

        # 2. Uploaded MP4 video exported as audio-only MP3 cleanly
        mp4_path = self.work_dir / "web_src.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=160x120:r=25:d=15",
                "-i",
                str(self.fixture_wav),
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(mp4_path),
            ],
            check=True,
        )
        res_mp3 = web_app.run_cleaning_job(
            mode="natural",
            start_text="00:00",
            end_text="00:13",
            out_format="mp3",
            uploaded_name="video_recitation.mp4",
            uploaded_bytes=mp4_path.read_bytes(),
        )
        self.assertEqual(res_mp3["output_filename"], "video_recitation_clean.mp3")
        mp3_probe_path = self.work_dir / "web_out.mp3"
        mp3_probe_path.write_bytes(res_mp3["audio_bytes"])
        mp3_info = probe_media(mp3_probe_path)
        self.assertFalse(mp3_info.has_video)
        self.assertEqual(mp3_info.audio_codec, "mp3")

        # 3. Graceful datacenter IP block message on YouTube DownloadError (403 / bot check)
        with patch(
            "app.download_youtube_audio",
            side_effect=RuntimeError("ERROR: [youtube] HTTP Error 403: Forbidden - Sign in to confirm you're not a bot"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                web_app.run_cleaning_job(
                    mode="natural",
                    start_text="",
                    end_text="",
                    out_format="m4a",
                    youtube_url="https://www.youtube.com/watch?v=ws_ofyVk_w0",
                )
        msg = str(ctx.exception)
        self.assertIn("Datacenter IP", msg)
        self.assertIn("Upload Audio/Video File", msg)

    def test_web_app_ui_smoke(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15)
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 2)
        self.assertEqual(len(at.radio), 2)
        self.assertEqual(len(at.text_input), 3)
        self.assertEqual(len(at.button), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
