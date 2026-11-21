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
from quran_cleaner.media import (
    extract_vad_audio,
    is_external_url,
    is_youtube_url,
    parse_timestamp,
    probe_media,
)
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
      - 2.6s silence gap
      - Distant crowd reaction (-31 dBFS, ~17 dB quieter than reciter)
      - 2.6s silence gap
      - Verse 2a (-14 dBFS) + 0.35s internal pause (<900ms min_silence) + Verse 2b (-14 dBFS)
      - 2.0s trailing silence
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
    _add(None, np.zeros(int(2.6 * sr), dtype=np.float64))
    _add("crowd", crowd)
    _add(None, np.zeros(int(2.6 * sr), dtype=np.float64))
    _add("verse2a", p2a)
    _add(None, np.zeros(int(0.35 * sr), dtype=np.float64))
    _add("verse2b", p2b)
    _add(None, np.zeros(int(2.0 * sr), dtype=np.float64))

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
            "--threshold",
            "--pad-start",
            "--pad-end",
            "--min-silence",
            "--gap-ms",
            "--start",
            "--end",
        ):
            self.assertIn(flag, res_help.stdout)
        self.assertNotIn("--mode", res_help.stdout.lower())
        self.assertNotIn("--strict-margin", res_help.stdout.lower())
        self.assertNotIn("strict", res_help.stdout.lower())
        self.assertNotIn("prayer", res_help.stdout.lower())

        res_strict = self._run_cli(str(self.fixture_wav), "--mode", "strict")
        self.assertNotEqual(res_strict.returncode, 0)
        res_margin = self._run_cli(str(self.fixture_wav), "--strict-margin-db", "10")
        self.assertNotEqual(res_margin.returncode, 0)

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
        self.assertEqual(data["mode"], "auto")
        self.assertNotIn("calibration", data)
        self.assertNotIn("strict_margin_db", data["settings"])
        self.assertFalse(data["summary"]["low_snr_guarded"])
        self.assertEqual(data["settings"]["pad_start_ms"], 650)
        self.assertEqual(data["settings"]["pad_end_ms"], 1100)
        self.assertEqual(data["settings"]["min_silence_ms"], 900)
        self.assertEqual(data["settings"]["crossfade_ms"], 400)
        self.assertEqual(data["settings"]["gap_ms"], 0)
        self.assertTrue(data["settings"]["direct_reverb_overlap"])
        self.assertFalse(data["settings"]["room_tone_bridge"])
        self.assertEqual(data["settings"]["edge_fade_in_ms"], 150)
        self.assertEqual(data["settings"]["edge_fade_out_ms"], 500)
        # Isolated -33 dBFS murmur inside 2.6s pauses is stripped; verse1 & verse2 (bridged) kept
        self.assertEqual(data["summary"]["segments_kept"], 2)
        self.assertEqual(data["summary"]["stray_shouts_stripped"], 1)
        expected_dur = sum(s["duration_s"] for s in data["segments"]) - 0.400
        self.assertAlmostEqual(data["summary"]["final_duration_s"], expected_dur, places=2)
        self.assertGreater(data["summary"]["time_saved_pct"], 25.0)

        # Verify untouched core samples in verse1 match original PCM_24 values without denoising/compression
        clean_audio, _ = sf.read(str(clean_wav), dtype="float64")
        seg0 = data["segments"][0]
        s0_orig = int(round(seg0["start_s"] * 48000))
        core_offset = int(0.75 * 48000)
        core_len = int(0.8 * 48000)
        diff = np.max(
            np.abs(
                clean_audio[core_offset : core_offset + core_len]
                - self.orig_audio[s0_orig + core_offset : s0_orig + core_offset + core_len]
            )
        )
        self.assertLess(diff, 2e-6)

        # Verify smooth splice transitions across direct 400ms reverb-tail overlap
        overlap_end = int(round(seg0["duration_s"] * 48000))
        overlap_start = max(0, overlap_end - int(0.40 * 48000))
        window = clean_audio[max(0, overlap_start - 1000) : overlap_end + 1000, 0]
        max_step = float(np.max(np.abs(np.diff(window))))
        self.assertLess(max_step, 0.05)

        # Verify gentle 150ms start fade-in & 500ms end fade-out keep outer edges near digital zero
        start_edge_peak = float(np.max(np.abs(clean_audio[: int(0.005 * 48000)])))
        end_edge_peak = float(np.max(np.abs(clean_audio[-int(0.005 * 48000) :])))
        self.assertLess(start_edge_peak, 5e-3)
        self.assertLess(end_edge_peak, 5e-3)

    def test_low_snr_safety_gate_and_hardening_guards(self) -> None:
        """Verify SNR < 12 dB safety gate, < 1.2s post-padding cut guard, and ultra-safe murmur bounds."""
        import app as web_app

        sr = 16000
        rng = np.random.default_rng(2026)

        def _harm_rms(dur_s: float, target_db: float, f0: float = 170.0) -> np.ndarray:
            n = int(round(dur_s * sr))
            t = np.arange(n, dtype=np.float64) / float(sr)
            mod = 0.70 + 0.30 * np.sin(2.0 * np.pi * 3.0 * t)
            sig = (
                0.55 * np.sin(2.0 * np.pi * f0 * t)
                + 0.30 * np.sin(2.0 * np.pi * (2.0 * f0) * t)
                + 0.15 * np.sin(2.0 * np.pi * (3.0 * f0) * t)
            ) * mod
            return _normalize_rms(sig, target_db)

        # 1. Low-SNR Safety Gate (SNR < 12.0 dB): pause silence removal & return [0, duration] untouched
        noisy = rng.normal(0.0, 10.0 ** (-20.0 / 20.0), int(12.0 * sr))
        noisy[int(1.0 * sr) : int(3.5 * sr)] += _harm_rms(2.5, -10.5)
        noisy[int(8.0 * sr) : int(10.5 * sr)] += _harm_rms(2.5, -10.5)
        vad_guarded = detect_speech_segments(noisy.astype(np.float32), VADConfig(), exact_duration=12.0)
        self.assertLess(vad_guarded.profile.snr_db, 12.0)
        self.assertTrue(vad_guarded.low_snr_guarded)
        self.assertEqual(len(vad_guarded.segments), 1)
        self.assertEqual(vad_guarded.segments[0].start, 0.0)
        self.assertEqual(vad_guarded.segments[0].end, 12.0)

        low_dir = self.work_dir / "low_snr_gate"
        low_dir.mkdir(exist_ok=True)
        low_wav = low_dir / "noisy_recitation.wav"
        sf.write(str(low_wav), np.column_stack([noisy, noisy]), sr, subtype="PCM_16")
        res_cli = self._run_cli(str(low_wav))
        self.assertEqual(res_cli.returncode, 0, res_cli.stderr + "\n" + res_cli.stdout)
        self.assertIn("low-snr safety gate", res_cli.stdout.lower())
        self.assertIn("حماية الحروف", res_cli.stdout)

        low_ts = json.loads((low_dir / "timestamps.json").read_text(encoding="utf-8"))
        self.assertTrue(low_ts["summary"]["low_snr_guarded"])
        self.assertEqual(low_ts["summary"]["segments_kept"], 1)
        self.assertEqual(low_ts["summary"]["time_saved_pct"], 0.0)
        orig_raw, _ = sf.read(str(low_wav), dtype="int16")
        clean_raw, _ = sf.read(str(low_dir / "noisy_recitation_clean.wav"), dtype="int16")
        self.assertTrue(np.array_equal(orig_raw, clean_raw))

        res_web_low = web_app.run_cleaning_job(
            out_format="m4a",
            quality="192",
            uploaded_name="noisy.wav",
            uploaded_bytes=low_wav.read_bytes(),
        )
        self.assertTrue(res_web_low["low_snr_guarded"])
        self.assertEqual(res_web_low["segments_kept"], 1)

        # 2. Minimum effective cut guard (< 1.2s post-padding merges vintage/live; studio unchanged)
        # 2.6s raw gap - (1.1s pad_end + 0.65s pad_start) = 0.85s < 1.20s -> merged to 1 segment
        short_cut = rng.normal(0.0, 10.0 ** (-50.0 / 20.0), int(14.0 * sr))
        short_cut[int(1.0 * sr) : int(5.0 * sr)] += _harm_rms(4.0, -10.0)
        short_cut[int(7.6 * sr) : int(12.6 * sr)] += _harm_rms(5.0, -10.0)
        res_merged = detect_speech_segments(short_cut.astype(np.float32), VADConfig(), exact_duration=14.0)
        self.assertEqual(res_merged.profile.kind, "vintage/live")
        self.assertEqual(len(res_merged.segments), 1)

        # 3.4s raw gap - 1.75s = 1.65s >= 1.20s -> split into 2 segments
        long_cut = rng.normal(0.0, 10.0 ** (-50.0 / 20.0), int(15.0 * sr))
        long_cut[int(1.0 * sr) : int(5.0 * sr)] += _harm_rms(4.0, -10.0)
        long_cut[int(8.4 * sr) : int(13.4 * sr)] += _harm_rms(5.0, -10.0)
        res_split = detect_speech_segments(long_cut.astype(np.float32), VADConfig(), exact_duration=15.0)
        self.assertEqual(len(res_split.segments), 2)

        # Studio keeps cutting even when post-padding gap is < 1.2s (1.4s - 0.5s = 0.9s)
        studio_cut = rng.normal(0.0, 10.0 ** (-70.0 / 20.0), int(12.0 * sr))
        studio_cut[int(1.0 * sr) : int(4.5 * sr)] += _harm_rms(3.5, -10.0)
        studio_cut[int(5.9 * sr) : int(10.0 * sr)] += _harm_rms(4.1, -10.0)
        res_studio = detect_speech_segments(studio_cut.astype(np.float32), VADConfig(), exact_duration=12.0)
        self.assertEqual(res_studio.profile.kind, "studio")
        self.assertEqual(len(res_studio.segments), 2)

        # 3. Ultra-safe murmur bounds: margin = max(10, min(14, snr * 0.40)) & cluster_span <= 3.5s
        from quran_cleaner.vad import _filter_crowd_murmur_clusters

        murmur_test = rng.normal(0.0, 10.0 ** (-52.0 / 20.0), int(43.0 * sr))
        murmur_test[int(1.0 * sr) : int(7.0 * sr)] += _harm_rms(6.0, -8.0)     # reciter median ~ -8 dBFS
        murmur_test[int(10.5 * sr) : int(12.8 * sr)] += _harm_rms(2.3, -17.0)  # 9 dB below (< 10..14 dB margin) -> KEPT
        murmur_test[int(16.2 * sr) : int(22.2 * sr)] += _harm_rms(6.0, -8.0)
        # 5.2s cluster (> 3.5s limit: two 2.0s bursts 1.2s apart) at -25 dBFS -> KEPT
        murmur_test[int(25.6 * sr) : int(27.6 * sr)] += _harm_rms(2.0, -25.0)
        murmur_test[int(28.8 * sr) : int(30.8 * sr)] += _harm_rms(2.0, -25.0)
        # Isolated 2.2s burst (<= 3.5s) at -26 dBFS (18 dB below reciter) -> STRIPPED
        murmur_test[int(34.2 * sr) : int(36.4 * sr)] += _harm_rms(2.2, -26.0)
        murmur_test[int(39.4 * sr) : int(42.0 * sr)] += _harm_rms(2.6, -8.0)

        res_murmur = detect_speech_segments(murmur_test.astype(np.float32), VADConfig(), exact_duration=43.0)
        self.assertEqual(res_murmur.stray_shouts_stripped, 1)
        self.assertTrue(any(s.start <= 11.5 <= s.end for s in res_murmur.segments))
        self.assertTrue(any(s.start <= 27.0 <= s.end for s in res_murmur.segments))
        self.assertFalse(any(s.start <= 35.3 <= s.end for s in res_murmur.segments))

        # Verify floor of 10.0 dB margin even at moderate SNR (15 dB -> 15*0.40=6 -> clamped to 10.0 dB)
        # and verify multi-burst cluster <= 3.5s is stripped while 3.8s > 3.5s is preserved
        segs_mod = [
            (0.0, 8.0, -10.0, np.full(10, -10.0)),
            (11.0, 12.2, -19.0, np.full(10, -19.0)),   # 9 dB below (< 10 dB floor) -> KEPT
            (15.5, 23.5, -10.0, np.full(10, -10.0)),
            (26.5, 27.5, -22.0, np.full(10, -22.0)),   # 2 bursts, 1.1s gap, 3.1s span <= 3.5s -> STRIPPED (2)
            (28.6, 29.6, -22.0, np.full(10, -22.0)),
            (33.0, 41.0, -10.0, np.full(10, -10.0)),
            (44.0, 45.4, -22.0, np.full(10, -22.0)),   # 2 bursts, 1.0s gap, 3.8s span > 3.5s -> KEPT
            (46.4, 47.8, -22.0, np.full(10, -22.0)),
            (51.0, 59.0, -10.0, np.full(10, -10.0)),
        ]
        kept_mod, stripped_cnt = _filter_crowd_murmur_clusters(segs_mod, snr_db=15.0, clip_start_s=0.0, clip_end_s=60.0)
        self.assertEqual(stripped_cnt, 2)
        self.assertEqual(len(kept_mod), 7)

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

        res_mp3 = self._run_cli(str(src_mp3))
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

            res = self._run_cli(str(src_vid))
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

        self.assertTrue(is_external_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_external_url("https://soundcloud.com/qari/surah-rahman"))
        self.assertTrue(is_external_url("https://archive.org/download/quran/001.mp3"))
        self.assertTrue(is_external_url("http://cdn.example.org/recitation.m4a"))
        self.assertFalse(is_external_url(str(self.fixture_wav)))
        self.assertFalse(is_external_url("ftp://example.com/file.mp3"))

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

        with patch("quran_cleaner.cli.download_external_audio", return_value=fake_downloaded) as mock_dl:
            rc = cli.main(
                [
                    "https://archive.org/download/quran/test123.m4a",
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
        res = detect_speech_segments(vad_audio, VADConfig(), exact_duration=studio_arr.size / sr)
        self.assertEqual(res.profile.kind, "studio")
        self.assertLess(res.profile.noise_floor_p5_db, -55.0)
        # Only the 2 speech blocks kept; the 300ms high-frequency breath in the middle is rejected
        self.assertEqual(len(res.segments), 2)

    def test_clip_range_start_end(self) -> None:
        out_dir = self.work_dir / "clip_test"
        out_dir.mkdir(exist_ok=True)
        src = out_dir / "full.wav"
        src.write_bytes(self.fixture_wav.read_bytes())

        res = self._run_cli(str(src), "--start", "00:09", "--end", "00:16")
        self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)
        data = json.loads((out_dir / "timestamps.json").read_text(encoding="utf-8"))
        self.assertEqual(data["settings"]["clip_start_s"], 9.0)
        self.assertEqual(data["settings"]["clip_end_s"], 16.0)
        self.assertAlmostEqual(data["summary"]["original_duration_s"], 7.0, places=2)
        for seg in data["segments"]:
            self.assertGreaterEqual(seg["start_s"], 9.0)
            self.assertLessEqual(seg["end_s"], 16.0)

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

        gap = np.zeros(int(2.4 * sr), dtype=np.float64)
        arr = np.concatenate([gap, speech, gap, cough_like, gap, speech, gap])
        arr += _normalize_rms(rng.standard_normal(arr.size), -49.0)

        res = detect_speech_segments(arr.astype(np.float32), VADConfig())
        self.assertEqual(res.profile.kind, "vintage/live")
        # Both speech blocks are kept while the short low-energy cough/noise blip is dropped
        self.assertEqual(len(res.segments), 2)

    def test_web_app_cleaning_job_and_youtube_fallback(self) -> None:
        import app as web_app

        # 1. Uploaded WAV exported directly as playable M4A + live progress checks
        progress_events: list[tuple[int, str]] = []
        res_m4a = web_app.run_cleaning_job(
            start_text="",
            end_text="",
            out_format="m4a",
            uploaded_name="vintage.wav",
            uploaded_bytes=self.fixture_wav.read_bytes(),
            progress_cb=lambda pct, msg: progress_events.append((pct, msg)),
        )
        self.assertEqual(res_m4a["output_filename"], "vintage [أنصتوا].m4a")
        self.assertEqual(res_m4a["mode"], "auto")
        self.assertFalse(res_m4a["low_snr_guarded"])
        self.assertGreater(len(res_m4a["audio_bytes"]), 1000)
        self.assertGreaterEqual(len(progress_events), 4)
        self.assertEqual(progress_events[-1][0], 100)
        self.assertEqual(res_m4a["segments_kept"], 2)
        self.assertEqual(res_m4a["profile_kind"], "vintage/live")
        self.assertTrue(res_m4a["direct_overlap"])

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
            start_text="00:00",
            end_text="00:13",
            out_format="mp3",
            uploaded_name="video_recitation.mp4",
            uploaded_bytes=mp4_path.read_bytes(),
        )
        self.assertEqual(res_mp3["output_filename"], "video_recitation [أنصتوا].mp3")
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
                    start_text="",
                    end_text="",
                    out_format="m4a",
                    youtube_url="https://www.youtube.com/watch?v=ws_ofyVk_w0",
                )
        msg = str(ctx.exception)
        self.assertIn("Datacenter IP", msg)
        self.assertIn("Upload Audio/Video File", msg)

    def test_adaptive_room_tone_and_high_ambient_stray_shouts(self) -> None:
        """Verify high-ambient direct overlap (1250/0/400), stray shout removal, and studio digital silence."""
        sr = 48000
        rng = np.random.default_rng(123)

        def _add_vocal(buf: np.ndarray, start_s: float, dur_s: float, amp: float) -> None:
            s_i = int(round(start_s * sr))
            n = int(round(dur_s * sr))
            t = np.arange(n, dtype=np.float64) / sr
            mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 3.2 * t)
            wave = (
                0.55 * np.sin(2.0 * np.pi * 165.0 * t)
                + 0.30 * np.sin(2.0 * np.pi * 330.0 * t)
                + 0.15 * np.sin(2.0 * np.pi * 495.0 * t)
            ) * mod
            wave /= float(np.sqrt(np.mean(wave * wave) + 1e-12))
            fade = min(int(0.03 * sr), n // 4)
            wave[:fade] *= np.linspace(0.0, 1.0, fade)
            wave[-fade:] *= np.linspace(1.0, 0.0, fade)
            buf[s_i : s_i + n] += wave * amp

        # 1. High-ambient vintage live WAV (-26 dBFS floor, ~16 dB SNR >= 12 dB gate & < 18 dB):
        #    Verse 1 (0.5..4.5) + isolated stray shout (7.5..8.7 inside >=3s gaps)
        #    + Verse 2 (11.7..15.7) + immediate crowd reaction (16.5..17.6, gap_before=0.8s < 1.5s)
        total_live_s = 19.5
        live_floor_amp = 10.0 ** (-26.0 / 20.0)
        raw_noise = rng.normal(0.0, 1.0, int(total_live_s * sr) + 4)
        live = 0.25 * raw_noise[:-4] + 0.50 * raw_noise[1:-3] + 0.25 * raw_noise[2:-2]
        live *= live_floor_amp / float(np.sqrt(np.mean(live * live) + 1e-12))

        rec_amp = 10.0 ** (-9.0 / 20.0)
        shout_amp = 10.0 ** (-23.0 / 20.0)
        _add_vocal(live, 0.5, 4.0, rec_amp)
        _add_vocal(live, 7.5, 1.2, shout_amp)
        _add_vocal(live, 11.7, 4.0, rec_amp)
        _add_vocal(live, 16.5, 1.1, shout_amp)

        live_wav = self.work_dir / "high_ambient_live.wav"
        sf.write(str(live_wav), np.column_stack([live, live]), sr, subtype="PCM_16")
        res = self._run_cli(str(live_wav))
        self.assertEqual(res.returncode, 0, res.stderr + "\n" + res.stdout)

        live_ts = json.loads((self.work_dir / "timestamps.json").read_text(encoding="utf-8"))
        self.assertEqual(live_ts["profile"]["detected"], "vintage/live")
        self.assertGreater(live_ts["profile"]["noise_floor_p5_db"], -30.0)
        self.assertEqual(live_ts["settings"]["pad_start_ms"], 650)
        self.assertEqual(live_ts["settings"]["pad_end_ms"], 1100)
        self.assertEqual(live_ts["settings"]["min_silence_ms"], 1250)
        self.assertEqual(live_ts["settings"]["gap_ms"], 0)
        self.assertEqual(live_ts["settings"]["crossfade_ms"], 400)
        self.assertTrue(live_ts["settings"]["direct_reverb_overlap"])
        self.assertFalse(live_ts["settings"]["room_tone_bridge"])
        self.assertEqual(live_ts["summary"]["stray_shouts_stripped"], 1)

        kept_segs = live_ts["segments"]
        # Stray shout around 8.1s stripped; Verse 1 (~2.5s), Verse 2 + immediate reaction (~17.0s) kept
        self.assertFalse(any(s["start_s"] <= 8.1 <= s["end_s"] for s in kept_segs))
        self.assertTrue(any(s["start_s"] <= 2.5 <= s["end_s"] for s in kept_segs))
        self.assertTrue(any(s["start_s"] <= 17.0 <= s["end_s"] for s in kept_segs))

        # Check direct 400ms overlap splice maintains continuous hall floor ambience (no zero hole)
        clean_live, _ = sf.read(str(self.work_dir / "high_ambient_live_clean.wav"), dtype="float64", always_2d=True)
        splice_end = int(round(kept_segs[0]["duration_s"] * sr))
        splice_start = max(0, splice_end - int(0.40 * sr))
        overlap_mid = clean_live[splice_start : splice_end, 0]
        overlap_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(overlap_mid * overlap_mid) + 1e-12)))
        self.assertGreater(overlap_rms_db, -36.0)
        self.assertLess(overlap_rms_db, -18.0)

        # 2. Studio recording (-64 dBFS floor): pure digital silence (700/450/20, pad 200/300)
        studio_dur_s = 11.0
        studio_amp = 10.0 ** (-64.0 / 20.0)
        studio = rng.normal(0.0, studio_amp, int(studio_dur_s * sr))
        _add_vocal(studio, 0.5, 3.0, rec_amp)
        _add_vocal(studio, 6.5, 3.0, rec_amp)

        studio_wav = self.work_dir / "clean_studio.wav"
        studio_out = self.work_dir / "clean_studio_out.wav"
        studio_ts_p = self.work_dir / "studio_ts.json"
        sf.write(str(studio_wav), np.column_stack([studio, studio]), sr, subtype="PCM_16")
        res_s = self._run_cli(str(studio_wav), "-o", str(studio_out), "--timestamps", str(studio_ts_p))
        self.assertEqual(res_s.returncode, 0, res_s.stderr + "\n" + res_s.stdout)

        studio_ts = json.loads(studio_ts_p.read_text(encoding="utf-8"))
        self.assertEqual(studio_ts["profile"]["detected"], "studio")
        self.assertEqual(studio_ts["settings"]["pad_start_ms"], 200)
        self.assertEqual(studio_ts["settings"]["pad_end_ms"], 300)
        self.assertEqual(studio_ts["settings"]["min_silence_ms"], 700)
        self.assertEqual(studio_ts["settings"]["gap_ms"], 450)
        self.assertEqual(studio_ts["settings"]["crossfade_ms"], 20)
        self.assertFalse(studio_ts["settings"]["direct_reverb_overlap"])
        self.assertFalse(studio_ts["settings"]["room_tone_bridge"])

        clean_studio, _ = sf.read(str(studio_out), dtype="float64", always_2d=True)
        sg_start = int(round(studio_ts["segments"][0]["duration_s"] * sr))
        sg_mid = clean_studio[sg_start + int(0.05 * sr) : sg_start + int(0.40 * sr), 0]
        sg_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(sg_mid * sg_mid) + 1e-12)))
        self.assertLess(sg_rms_db, -90.0)

    def test_multi_range_and_quality_presets(self) -> None:
        """Verify multi-range seamless stitching and 128/256/Lossless WAV quality options."""
        import app as web_app

        self.assertTrue(web_app.is_lossless_source("recitation.wav"))
        self.assertTrue(web_app.is_lossless_source("recitation.FLAC"))
        self.assertFalse(web_app.is_lossless_source("recitation.mp3"))
        self.assertIn("lossless_wav", web_app.quality_options_for_source("recitation.flac"))
        self.assertNotIn("lossless_wav", web_app.quality_options_for_source("recitation.m4a"))

        flac_path = self.work_dir / "source_lossless.flac"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(self.fixture_wav), "-c:a", "flac", str(flac_path)],
            check=True,
        )

        # Multi-range Lossless WAV output from FLAC source (spans two verses across 2.6s gaps)
        res_wav = web_app.run_cleaning_job(
            ranges=[("00:00", "00:05"), ("00:10", "00:16")],
            out_format="m4a",
            quality="lossless_wav",
            uploaded_name="source_lossless.flac",
            uploaded_bytes=flac_path.read_bytes(),
        )
        self.assertEqual(res_wav["output_filename"], "source_lossless [أنصتوا].wav")
        self.assertEqual(res_wav["mime_type"], "audio/wav")
        self.assertEqual(res_wav["ranges_count"], 2)
        wav_probe_p = self.work_dir / "web_multi.wav"
        wav_probe_p.write_bytes(res_wav["audio_bytes"])
        wav_info = probe_media(wav_probe_p)
        self.assertTrue(wav_info.audio_codec.startswith("pcm_"))
        self.assertEqual(wav_info.sample_rate, 48000)
        ts_multi = json.loads(res_wav["timestamps_json"])
        self.assertEqual(len(ts_multi["ranges"]), 2)
        self.assertGreaterEqual(res_wav["segments_kept"], 2)

        # Multi-range 128 kbps M4A vs 256 kbps MP3
        res_128 = web_app.run_cleaning_job(
            ranges=[("00:00", "00:05"), ("00:10", "00:16")],
            out_format="m4a",
            quality="128",
            uploaded_name="source_lossless.flac",
            uploaded_bytes=flac_path.read_bytes(),
        )
        m4a_p = self.work_dir / "web_multi_128.m4a"
        m4a_p.write_bytes(res_128["audio_bytes"])
        m4a_info = probe_media(m4a_p)
        self.assertEqual(m4a_info.audio_codec, "aac")
        self.assertIsNotNone(m4a_info.audio_bit_rate)
        self.assertLess(m4a_info.audio_bit_rate, 180_000)

        res_256 = web_app.run_cleaning_job(
            ranges=[("00:00", "00:05"), ("00:10", "00:16")],
            out_format="mp3",
            quality="256",
            uploaded_name="source_lossless.flac",
            uploaded_bytes=flac_path.read_bytes(),
        )
        mp3_p = self.work_dir / "web_multi_256.mp3"
        mp3_p.write_bytes(res_256["audio_bytes"])
        mp3_info = probe_media(mp3_p)
        self.assertEqual(mp3_info.audio_codec, "mp3")
        self.assertIsNotNone(mp3_info.audio_bit_rate)
        self.assertGreaterEqual(mp3_info.audio_bit_rate, 240_000)

    def test_surgical_acoustic_fixes_and_external_url_quality_guards(self) -> None:
        """Verify 12..22 dB vintage 2.5dB sustain, studio p90-26dB quiet passage, URL/WAV guards, and removed mode args."""
        import app as web_app

        sr = 16000
        rng = np.random.default_rng(77)

        def _harm(dur_s: float, amp: float) -> np.ndarray:
            n = int(round(dur_s * sr))
            t = np.arange(n, dtype=np.float64) / float(sr)
            sig = (
                0.55 * np.sin(2.0 * np.pi * 175.0 * t)
                + 0.30 * np.sin(2.0 * np.pi * 350.0 * t)
                + 0.15 * np.sin(2.0 * np.pi * 525.0 * t)
            )
            return sig * (amp / float(np.sqrt(np.mean(sig * sig) + 1e-12)))

        # 1. Moderate-low SNR vintage sustain (spread ~15 dB, >= 12 dB gate & < 22 dB):
        #    steady -10 dBFS head + 2.0s soft tail (~-16.0 dBFS)
        low_snr = rng.normal(0.0, 10.0 ** (-25.0 / 20.0), int(11.0 * sr))
        low_snr[int(1.0 * sr) : int(5.0 * sr)] += _harm(4.0, 10.0 ** (-10.0 / 20.0))
        low_snr[int(5.0 * sr) : int(7.0 * sr)] += _harm(2.0, 10.0 ** (-16.0 / 20.0))
        low_snr[int(7.5 * sr) : int(10.0 * sr)] += _harm(2.5, 10.0 ** (-10.0 / 20.0))
        low_res = detect_speech_segments(low_snr.astype(np.float32), VADConfig(), exact_duration=11.0)
        self.assertEqual(low_res.profile.kind, "vintage/live")
        self.assertFalse(low_res.low_snr_guarded)
        self.assertGreaterEqual(low_res.profile.snr_db, 12.0)
        self.assertLess(low_res.profile.snr_db, 22.0)
        # Soft -16.0 dBFS tail [5.0..7.0s] must remain preserved inside first segment
        self.assertTrue(any(s.start <= 1.2 and s.end >= 6.8 for s in low_res.segments))

        # 2. Studio soft passage (-33 dBFS vs -10 dBFS p90 -> p90-23, safely above p90-26)
        studio = rng.normal(0.0, 10.0 ** (-72.0 / 20.0), int(12.0 * sr))
        studio[int(1.0 * sr) : int(4.5 * sr)] += _harm(3.5, 10.0 ** (-10.0 / 20.0))
        studio[int(6.2 * sr) : int(8.0 * sr)] += _harm(1.8, 10.0 ** (-33.0 / 20.0))
        studio[int(9.0 * sr) : int(11.0 * sr)] += _harm(2.0, 10.0 ** (-10.0 / 20.0))
        studio_res = detect_speech_segments(studio.astype(np.float32), VADConfig(), exact_duration=12.0)
        self.assertEqual(studio_res.profile.kind, "studio")
        self.assertTrue(any(s.start <= 6.5 and s.end >= 7.7 for s in studio_res.segments))

        # 3. Removed mode flags/args (prayer/strict) are rejected cleanly in CLI & web runner
        res_bad_mode = self._run_cli(str(self.fixture_wav), "--mode", "prayer")
        self.assertNotEqual(res_bad_mode.returncode, 0)
        with self.assertRaises(TypeError):
            web_app.run_cleaning_job(  # type: ignore[call-arg]
                mode="strict",
                out_format="m4a",
                quality="192",
                uploaded_name="sample.wav",
                uploaded_bytes=self.fixture_wav.read_bytes(),
            )

        # 4. Web app Lossless WAV blocked for compressed uploads and external URLs, plus SoundCloud mock
        with self.assertRaises(ValueError):
            web_app.run_cleaning_job(
                out_format="m4a",
                quality="lossless_wav",
                uploaded_name="track.mp3",
                uploaded_bytes=b"dummy",
            )
        with self.assertRaises(ValueError):
            web_app.run_cleaning_job(
                out_format="m4a",
                quality="lossless_wav",
                external_url="https://soundcloud.com/qari/recitation",
            )

        m4a_dl = self.work_dir / "sc_mock.m4a"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(self.fixture_wav), "-c:a", "aac", "-b:a", "192k", str(m4a_dl)],
            check=True,
        )
        with patch("app.download_youtube_audio", return_value=m4a_dl) as mock_dl:
            res_sc = web_app.run_cleaning_job(
                out_format="m4a",
                quality="192",
                external_url="https://soundcloud.com/qari/surah-qaf",
            )
        mock_dl.assert_called_once()
        self.assertEqual(res_sc["output_filename"], "sc_mock [أنصتوا].m4a")
        self.assertEqual(res_sc["quality"], "192")

    def test_universal_vintage_overlap_cluster_murmur_and_dynamic_pad_end(self) -> None:
        """Verify <=-50dBFS vintage/live gets 400ms overlap, 650/1100/1300ms dynamic pad_end, and <=3.5s cluster murmur rules."""
        sr = 16000
        rng = np.random.default_rng(99)

        def _add_harm(buf: np.ndarray, t0: float, dur: float, amp: float) -> None:
            s0 = int(round(t0 * sr))
            n = int(round(dur * sr))
            t = np.arange(n, dtype=np.float64) / sr
            mod = 0.70 + 0.30 * np.sin(2.0 * np.pi * 3.0 * t)
            sig = (
                0.55 * np.sin(2.0 * np.pi * 170.0 * t)
                + 0.30 * np.sin(2.0 * np.pi * 340.0 * t)
                + 0.15 * np.sin(2.0 * np.pi * 510.0 * t)
            ) * mod
            buf[s0 : s0 + n] += sig * (amp / float(np.sqrt(np.mean(sig * sig) + 1e-12)))

        # Timeline (~60.0s total) with -52 dBFS vintage floor (<= -50 dBFS like Hud/Yunus/An'am):
        #  1.0..9.0s: Loud Verse 1 ending (-6 dBFS finale >= reciter_p75 -> gets 1300ms pad_end!)
        # 12.0..13.2s + 13.9..15.1s: 2-burst crowd murmur cluster (span 3.1s <= 3.5s, 0.7s < 1.8s link,
        #                          gaps >= 3.0s before and >= 5.0s after, 16 dB below reciter -> STRIPPED as 2 bursts)
        # 20.2..28.2s: Moderate Verse 2 ending (-11 dBFS < reciter_p75 -> gets 1100ms pad_end!)
        # 31.6..46.4s: Sustained soft recitation (> 3.5s: three 4.2s phrases with 1.1s gaps = 14.8s at -24.5 dBFS -> KEPT!)
        # 49.8..57.8s: Loud Verse 3 (-8 dBFS)
        arr = rng.normal(0.0, 10.0 ** (-52.0 / 20.0), int(60.0 * sr))
        _add_harm(arr, 1.0, 7.0, 10.0 ** (-8.5 / 20.0))
        _add_harm(arr, 8.0, 1.0, 10.0 ** (-6.0 / 20.0))
        _add_harm(arr, 12.0, 1.2, 10.0 ** (-24.5 / 20.0))
        _add_harm(arr, 13.9, 1.2, 10.0 ** (-24.5 / 20.0))
        _add_harm(arr, 20.2, 8.0, 10.0 ** (-11.0 / 20.0))
        _add_harm(arr, 31.6, 4.2, 10.0 ** (-24.5 / 20.0))
        _add_harm(arr, 36.9, 4.2, 10.0 ** (-24.5 / 20.0))
        _add_harm(arr, 42.2, 4.2, 10.0 ** (-24.5 / 20.0))
        _add_harm(arr, 49.8, 8.0, 10.0 ** (-8.0 / 20.0))

        res = detect_speech_segments(arr.astype(np.float32), VADConfig(), exact_duration=60.0)
        self.assertEqual(res.profile.kind, "vintage/live")
        self.assertLessEqual(res.profile.noise_floor_p5_db, -50.0)
        self.assertEqual(res.resolved_pad_start_ms, 650)
        self.assertEqual(res.resolved_pad_end_ms, 1100)
        self.assertEqual(res.resolved_min_silence_ms, 900)
        self.assertEqual(res.resolved_crossfade_ms, 400)
        self.assertEqual(res.resolved_gap_ms, 0)
        self.assertTrue(res.direct_overlap)
        self.assertEqual(res.stray_shouts_stripped, 1)

        # Murmur cluster [12.0..15.1] stripped; soft >3.5s recitation [31.6..46.4] preserved!
        self.assertFalse(any(s.start <= 13.5 <= s.end for s in res.segments))
        self.assertTrue(any(s.start <= 38.0 <= s.end for s in res.segments))

        # Loud ending (Verse 1 ending at ~9.0s) gets ~1300ms pad_end (~10.30s), moderate Verse 2 (~28.2s) gets ~1100ms (~29.30s)
        seg_v1 = next(s for s in res.segments if s.start <= 1.2 <= s.end)
        seg_v2 = next(s for s in res.segments if s.start <= 21.0 <= s.end)
        self.assertGreaterEqual(seg_v1.end, 10.20)
        self.assertLess(seg_v2.end, 29.40)

    def test_web_app_ui_smoke(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15)
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 2)
        self.assertEqual(len(at.radio), 1)
        self.assertEqual(len(at.selectbox), 1)
        self.assertEqual(at.selectbox[0].value, "192")
        self.assertNotIn("lossless_wav", at.selectbox[0].options)
        self.assertEqual(len(at.text_input), 3)
        self.assertEqual(len(at.button), 3)
        self.assertIsNotNone(at.button(key="run_external_btn"))

        # Click "+ Add Range" and confirm dynamic second range row + remove button appear cleanly
        at.button(key="add_range_btn").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.text_input), 5)
        self.assertIsNotNone(at.button(key="del_range_1"))
        at.button(key="del_range_1").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.text_input), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
