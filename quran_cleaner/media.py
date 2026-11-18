"""Media discovery, external URL downloading (yt-dlp), FFprobe inspection, and 16kHz VAD extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

SUPPORTED_LOCAL_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".webm", ".opus", ".flac"}
VAD_SAMPLE_RATE = 16000

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


@dataclass(frozen=True)
class MediaInfo:
    """Metadata for the source audio/video file probed via FFprobe."""

    path: Path
    duration: float
    has_video: bool
    has_attached_pic: bool
    audio_codec: str
    sample_rate: int
    channels: int
    channel_layout: Optional[str]
    audio_bit_rate: Optional[int]
    sample_fmt: Optional[str]
    bits_per_raw_sample: Optional[int]
    video_codec: Optional[str] = None
    video_fps: Optional[Fraction] = None
    video_bit_rate: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


def require_ffmpeg() -> tuple[str, str]:
    """Locate ffmpeg and ffprobe executables or raise a clear error."""
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if not ffmpeg_bin or not ffprobe_bin:
        raise RuntimeError(
            "FFmpeg and FFprobe are required and must be available on PATH."
        )
    return ffmpeg_bin, ffprobe_bin


def is_external_url(source: str) -> bool:
    """Return True if the input string is a valid http:// or https:// external media URL."""
    stripped = source.strip()
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    parsed = urlparse(stripped)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc.strip())


def is_youtube_url(source: str) -> bool:
    """Return True if the input string looks like a YouTube URL."""
    if not is_external_url(source):
        return False
    parsed = urlparse(source.strip())
    host = parsed.netloc.lower().split(":")[0]
    return host in _YOUTUBE_HOSTS


def validate_local_media(path: Path) -> Path:
    """Verify that a local input path exists and has a supported extension."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_LOCAL_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_EXTENSIONS))
        raise ValueError(
            f"Unsupported file format '{ext or '<none>'}'. Supported formats: {supported}"
        )
    return resolved


def download_external_audio(url: str, output_dir: Path) -> Path:
    """Download the best native audio stream from any supported URL via yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required to download external URLs. Install it via requirements.txt."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError(f"yt-dlp returned no metadata for URL: {url}")

        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            downloaded = Path(requested[0]["filepath"])
        else:
            downloaded = Path(ydl.prepare_filename(info))

    if not downloaded.is_file():
        raise RuntimeError(f"Downloaded audio file could not be found: {downloaded}")
    return downloaded.resolve()


download_youtube_audio = download_external_audio


def _parse_int(value: object) -> Optional[int]:
    if value is None or value == "" or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fps(rate_str: Optional[str]) -> Optional[Fraction]:
    if not rate_str or rate_str in {"0/0", "N/A"}:
        return None
    try:
        frac = Fraction(rate_str)
        return frac if frac > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_media(path: Path) -> MediaInfo:
    """Inspect media streams and audio properties using FFprobe."""
    _, ffprobe_bin = require_ffmpeg()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )
    if audio_stream is None:
        raise ValueError(f"No audio stream found in input file: {path}")

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    attached_pic = any(
        (s.get("disposition") or {}).get("attached_pic") == 1 for s in video_streams
    )
    real_video_stream = next(
        (
            s
            for s in video_streams
            if (s.get("disposition") or {}).get("attached_pic") != 1
        ),
        None,
    )

    duration_raw = fmt.get("duration") or audio_stream.get("duration")
    if duration_raw is None:
        raise RuntimeError(f"Could not determine duration for: {path}")
    duration = float(duration_raw)

    sample_rate = _parse_int(audio_stream.get("sample_rate"))
    channels = _parse_int(audio_stream.get("channels"))
    if not sample_rate or not channels:
        raise RuntimeError(f"Invalid audio stream parameters in: {path}")

    audio_bit_rate = _parse_int(audio_stream.get("bit_rate")) or _parse_int(
        fmt.get("bit_rate")
    )

    video_fps = None
    if real_video_stream is not None:
        video_fps = _parse_fps(real_video_stream.get("avg_frame_rate")) or _parse_fps(
            real_video_stream.get("r_frame_rate")
        )

    return MediaInfo(
        path=path,
        duration=duration,
        has_video=real_video_stream is not None,
        has_attached_pic=attached_pic,
        audio_codec=str(audio_stream.get("codec_name") or ""),
        sample_rate=sample_rate,
        channels=channels,
        channel_layout=audio_stream.get("channel_layout"),
        audio_bit_rate=audio_bit_rate,
        sample_fmt=audio_stream.get("sample_fmt"),
        bits_per_raw_sample=_parse_int(audio_stream.get("bits_per_raw_sample")),
        video_codec=(
            str(real_video_stream.get("codec_name") or "")
            if real_video_stream is not None
            else None
        ),
        video_fps=video_fps,
        video_bit_rate=(
            _parse_int(real_video_stream.get("bit_rate"))
            if real_video_stream is not None
            else None
        ),
        width=(
            _parse_int(real_video_stream.get("width"))
            if real_video_stream is not None
            else None
        ),
        height=(
            _parse_int(real_video_stream.get("height"))
            if real_video_stream is not None
            else None
        ),
    )


def parse_timestamp(value: str) -> float:
    """Parse MM:SS, HH:MM:SS, or numeric seconds into non-negative float seconds."""
    raw = value.strip()
    if not raw:
        raise ValueError("Timestamp cannot be empty.")

    if ":" not in raw:
        try:
            secs = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp '{value}'. Use MM:SS or HH:MM:SS.") from exc
        if secs < 0:
            raise ValueError(f"Timestamp must be non-negative: '{value}'.")
        return secs

    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid timestamp '{value}'. Use MM:SS or HH:MM:SS.")

    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp '{value}'. Use MM:SS or HH:MM:SS.") from exc

    if len(nums) == 2:
        mins, secs = nums
        if mins < 0 or secs < 0 or secs >= 60.0 or not mins.is_integer():
            raise ValueError(f"Invalid MM:SS timestamp '{value}'.")
        return mins * 60.0 + secs

    hours, mins, secs = nums
    if (
        hours < 0
        or mins < 0
        or secs < 0
        or mins >= 60.0
        or secs >= 60.0
        or not hours.is_integer()
        or not mins.is_integer()
    ):
        raise ValueError(f"Invalid HH:MM:SS timestamp '{value}'.")
    return hours * 3600.0 + mins * 60.0 + secs


def extract_vad_audio(
    path: Path,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> np.ndarray:
    """Decode a temporary 16kHz mono float32 stream strictly for VAD analysis.

    Applies a gentle 80Hz high-pass filter on this temporary analysis stream only
    to strip vintage tape rumble and AC mains hum without touching the original
    high-sample-rate output audio/video stream.
    """
    ffmpeg_bin, _ = require_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(path),
    ]
    if start_s is not None and start_s > 0.0:
        cmd.extend(["-ss", f"{start_s:.4f}"])
    if end_s is not None:
        cmd.extend(["-to", f"{end_s:.4f}"])
    cmd.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(VAD_SAMPLE_RATE),
            "-af",
            "highpass=f=80",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg 16kHz extraction failed: {err}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"Extracted 16kHz VAD stream is empty for: {path}")
    return audio
