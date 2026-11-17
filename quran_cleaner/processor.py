"""Zero-quality-loss dual-rate media slicer, 20ms crossfade engine, breathing gaps, and export."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .media import MediaInfo, require_ffmpeg
from .vad import Segment, VADConfig, VADResult

_DECODE_CHUNK_FRAMES = 32768
_MAX_EDGE_FADE_MS = 5.0
_AUDIO_ONLY_EXTENSIONS = {".mp3", ".m4a", ".wav", ".opus", ".ogg", ".flac"}

_Y4M_BYTES_PER_PIXEL: dict[str, float] = {
    "C420": 1.5,
    "C420jpeg": 1.5,
    "C420paldv": 1.5,
    "C420mpeg2": 1.5,
    "C422": 2.0,
    "C444": 3.0,
    "Cmono": 1.0,
    "C444alpha": 4.0,
    "C420p10": 3.0,
    "C420p12": 3.0,
    "C420p16": 3.0,
    "C422p10": 4.0,
    "C422p12": 4.0,
    "C422p16": 4.0,
    "C444p10": 6.0,
    "C444p12": 6.0,
    "C444p16": 6.0,
}


@dataclass(frozen=True)
class ProcessResult:
    """Summary metrics and output artifact paths from processing."""

    input_path: Path
    output_path: Path
    timestamps_path: Path
    original_duration: float
    final_duration: float
    time_saved_pct: float
    segments_kept: int
    segments: list[Segment]


def _format_timecode(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _equal_power_curves(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        empty = np.empty((0, 1), dtype=np.float64)
        return empty, empty
    theta = np.linspace(0.0, np.pi / 2.0, length, endpoint=True, dtype=np.float64)[:, None]
    return np.cos(theta), np.sin(theta)


class CrossfadeSegmentStream:
    """Streaming equal-power crossfade blender for sequential audio segments.

    Splices segments cleanly using an equal-power (cos/sin) 20ms crossfade
    without loading the full high-sample-rate audio file into RAM.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        crossfade_ms: int,
        write_fn: Callable[[np.ndarray], None],
        fade_start_edge: bool = False,
        fade_end_edge: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.fade_samples = max(0, int(round((crossfade_ms / 1000.0) * sample_rate)))
        self._write_fn = write_fn
        self._fade_out, self._fade_in = _equal_power_curves(self.fade_samples)

        edge_target = int(round((_MAX_EDGE_FADE_MS / 1000.0) * sample_rate))
        self._edge_samples = min(self.fade_samples, edge_target) if self.fade_samples > 0 else 0
        self._start_fade_remaining = self._edge_samples if fade_start_edge else 0
        self._start_fade_total = self._edge_samples
        self._fade_end_edge = fade_end_edge

        self._prev_tail: Optional[np.ndarray] = None
        self._head_chunks: list[np.ndarray] = []
        self._head_len = 0
        self._tail_buf = np.empty((0, channels), dtype=np.float64)
        self.frames_written = 0

    def _emit(self, block: np.ndarray) -> None:
        if block.shape[0] == 0:
            return
        out = block
        if self._start_fade_remaining > 0:
            n = min(self._start_fade_remaining, out.shape[0])
            start_idx = self._start_fade_total - self._start_fade_remaining
            ramp = np.linspace(
                start_idx / float(self._start_fade_total),
                (start_idx + n) / float(self._start_fade_total),
                n,
                endpoint=True,
                dtype=np.float64,
            )[:, None]
            out = out.copy()
            out[:n] *= ramp
            self._start_fade_remaining -= n

        self._write_fn(out)
        self.frames_written += out.shape[0]

    def _blend(self, prev_tail: np.ndarray, curr_head: np.ndarray) -> np.ndarray:
        n = prev_tail.shape[0]
        if n == self.fade_samples:
            f_out, f_in = self._fade_out, self._fade_in
        else:
            f_out, f_in = _equal_power_curves(n)
        return prev_tail * f_out + curr_head * f_in

    def feed(self, chunk: np.ndarray, is_last_segment: bool) -> None:
        if chunk.shape[0] == 0:
            return
        if chunk.ndim == 1:
            chunk = chunk[:, None]

        if self._prev_tail is not None:
            self._head_chunks.append(chunk)
            self._head_len += chunk.shape[0]
            needed = self._prev_tail.shape[0]
            if self._head_len < needed:
                return

            full_head = np.concatenate(self._head_chunks, axis=0)
            self._head_chunks.clear()
            self._head_len = 0

            self._emit(self._blend(self._prev_tail, full_head[:needed]))
            self._prev_tail = None
            chunk = full_head[needed:]
            if chunk.shape[0] == 0:
                return

        if self.fade_samples == 0:
            self._emit(chunk)
            return

        if is_last_segment:
            if not self._fade_end_edge:
                self._emit(chunk)
                return
            keep = self._edge_samples
        else:
            keep = self.fade_samples

        combined = (
            np.concatenate([self._tail_buf, chunk], axis=0)
            if self._tail_buf.shape[0] > 0
            else chunk
        )
        if combined.shape[0] > keep:
            emit_len = combined.shape[0] - keep
            self._emit(combined[:emit_len])
            self._tail_buf = combined[emit_len:]
        else:
            self._tail_buf = combined

    def end_segment(self, is_last_segment: bool) -> None:
        if self._prev_tail is not None:
            full_head = (
                np.concatenate(self._head_chunks, axis=0)
                if self._head_chunks
                else np.empty((0, self.channels), dtype=np.float64)
            )
            self._head_chunks.clear()
            self._head_len = 0
            n = min(self._prev_tail.shape[0], full_head.shape[0])
            if n > 0:
                self._emit(self._blend(self._prev_tail[:n], full_head[:n]))
            if self._prev_tail.shape[0] > n:
                self._emit(self._prev_tail[n:])
            self._prev_tail = None
            if full_head.shape[0] > n:
                self._tail_buf = full_head[n:]

        if not is_last_segment:
            self._prev_tail = self._tail_buf
            self._tail_buf = np.empty((0, self.channels), dtype=np.float64)
            return

        if self._tail_buf.shape[0] > 0:
            tail = self._tail_buf
            if self._fade_end_edge and tail.shape[0] > 1:
                n = tail.shape[0]
                ramp = np.linspace(1.0, 0.0, n, endpoint=True, dtype=np.float64)[:, None]
                tail = tail * ramp
            self._emit(tail)
            self._tail_buf = np.empty((0, self.channels), dtype=np.float64)


def _feed_gap_segment(
    blender: CrossfadeSegmentStream,
    gap_buffer: Optional[np.ndarray],
) -> None:
    """Feed the clean room-tone/silence breathing gap between adjacent segments."""
    if gap_buffer is None or gap_buffer.shape[0] == 0:
        return
    blender.feed(gap_buffer, is_last_segment=False)
    blender.end_segment(is_last_segment=False)


def _build_room_tone_buffer(
    input_path: Path,
    info: MediaInfo,
    vad_result: VADResult,
    gap_samples: int,
    fade_samples: int,
) -> Optional[np.ndarray]:
    """Build a native-rate low-level room-tone (or silence) gap buffer of length gap+fade."""
    if gap_samples <= 0:
        return None

    needed = gap_samples + fade_samples
    sr = info.sample_rate
    channels = info.channels
    gap_window = vad_result.quietest_gap_window

    if gap_window is None or gap_window[2] > -30.0:
        return np.zeros((needed, channels), dtype=np.float64)

    g_start, g_end, g_rms_db = gap_window
    dur = max(0.20, g_end - g_start)

    try:
        if input_path.suffix.lower() == ".wav":
            with sf.SoundFile(str(input_path), mode="r") as sf_in:
                s_idx = max(0, min(sf_in.frames - 1, int(round(g_start * sr))))
                n_read = min(sf_in.frames - s_idx, max(1, int(round(dur * sr))))
                sf_in.seek(s_idx)
                grain = sf_in.read(n_read, dtype="float64", always_2d=True)
        else:
            ffmpeg_bin, _ = require_ffmpeg()
            raw = subprocess.check_output(
                [
                    ffmpeg_bin,
                    "-v",
                    "error",
                    "-ss",
                    f"{g_start:.4f}",
                    "-i",
                    str(input_path),
                    "-t",
                    f"{dur:.4f}",
                    "-vn",
                    "-c:a",
                    "pcm_f32le",
                    "-ar",
                    str(sr),
                    "-ac",
                    str(channels),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                stdin=subprocess.DEVNULL,
            )
            n_frames = len(raw) // (channels * 4)
            grain = (
                np.frombuffer(raw, dtype="<f4", count=n_frames * channels)
                .reshape(n_frames, channels)
                .astype(np.float64)
            )
    except Exception:
        return np.zeros((needed, channels), dtype=np.float64)

    if grain.shape[0] < max(64, int(0.10 * sr)):
        return np.zeros((needed, channels), dtype=np.float64)

    # Strip DC offset and keep the gap genuinely low-level (<= -46 dBFS).
    grain = grain - np.mean(grain, axis=0, keepdims=True)
    grain_rms = float(np.sqrt(np.mean(grain * grain) + 1e-12))
    grain_db = float(20.0 * np.log10(grain_rms))
    if grain_db > -46.0:
        grain *= 10.0 ** ((-46.0 - grain_db) / 20.0)

    if grain.shape[0] >= needed:
        return grain[:needed].copy()

    # Crossfade-loop mirrored grain cleanly when the extracted quiet window is shorter than needed.
    xf = min(grain.shape[0] // 4, max(32, int(0.02 * sr)))
    mirrored = np.concatenate([grain, grain[::-1]], axis=0)
    out = np.empty((needed, channels), dtype=np.float64)
    pos = 0
    step = max(1, mirrored.shape[0] - xf)
    while pos < needed:
        take = min(mirrored.shape[0], needed - pos)
        if pos == 0:
            out[:take] = mirrored[:take]
        else:
            blend_n = min(xf, take)
            f_out, f_in = _equal_power_curves(blend_n)
            out[pos : pos + blend_n] = out[pos : pos + blend_n] * f_out + mirrored[:blend_n] * f_in
            if take > blend_n:
                out[pos + blend_n : pos + take] = mirrored[blend_n:take]
        pos += step if pos + take < needed else take
    return out


def _build_sample_ranges(
    segments: list[Segment],
    sample_rate: int,
    total_frames: Optional[int],
    fade_samples: int,
) -> list[tuple[int, int]]:
    """Convert time segments to sample intervals with tail extension for exact crossfade duration."""
    ranges: list[tuple[int, int]] = []
    last_idx = len(segments) - 1
    for idx, seg in enumerate(segments):
        s_frame = max(0, int(round(seg.start * sample_rate)))
        e_frame = max(s_frame + 1, int(round(seg.end * sample_rate)))
        if idx < last_idx and fade_samples > 0:
            e_frame += fade_samples
            next_start = max(0, int(round(segments[idx + 1].start * sample_rate)))
            e_frame = min(e_frame, next_start)
        if total_frames is not None:
            e_frame = min(e_frame, total_frames)
        if e_frame > s_frame:
            ranges.append((s_frame, e_frame))
    return ranges


def _process_wav_soundfile(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    info: MediaInfo,
    vad_result: VADResult,
    crossfade_ms: int,
    gap_ms: int,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> float:
    """Slice and crossfade a WAV file directly using soundfile without re-encoding."""
    with sf.SoundFile(str(input_path), mode="r") as sf_in:
        sr = sf_in.samplerate
        channels = sf_in.channels
        subtype = sf_in.subtype
        fmt = sf_in.format
        if not sf.check_format(fmt, subtype):
            subtype = "PCM_24" if sf.check_format(fmt, "PCM_24") else "PCM_16"
        total_frames = sf_in.frames

        fade_samples = max(0, int(round((crossfade_ms / 1000.0) * sr)))
        gap_samples = max(0, int(round((gap_ms / 1000.0) * sr))) if len(segments) > 1 else 0
        ranges = _build_sample_ranges(segments, sr, total_frames, fade_samples)
        if not ranges:
            raise RuntimeError("No valid audio sample ranges remained after slicing.")

        gap_buffer = _build_room_tone_buffer(
            input_path=input_path,
            info=info,
            vad_result=vad_result,
            gap_samples=gap_samples,
            fade_samples=fade_samples,
        )

        fade_start = ranges[0][0] > 0
        fade_end = ranges[-1][1] < total_frames

        with sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=sr,
            channels=channels,
            subtype=subtype,
            format=fmt,
        ) as sf_out:
            def _write(arr: np.ndarray) -> None:
                sf_out.write(arr if channels > 1 else arr[:, 0])

            blender = CrossfadeSegmentStream(
                sample_rate=sr,
                channels=channels,
                crossfade_ms=crossfade_ms,
                write_fn=_write,
                fade_start_edge=fade_start,
                fade_end_edge=fade_end,
            )

            last_idx = len(ranges) - 1
            for idx, (s_frame, e_frame) in enumerate(ranges):
                sf_in.seek(s_frame)
                remaining = e_frame - s_frame
                is_last = idx == last_idx
                while remaining > 0:
                    to_read = min(_DECODE_CHUNK_FRAMES, remaining)
                    data = sf_in.read(to_read, dtype="float64", always_2d=True)
                    if data.shape[0] == 0:
                        break
                    blender.feed(data, is_last_segment=is_last)
                    remaining -= data.shape[0]
                blender.end_segment(is_last_segment=is_last)
                if not is_last:
                    _feed_gap_segment(blender, gap_buffer)
                if progress_cb is not None:
                    progress_cb(min(1.0, (idx + 1) / float(len(ranges))))

            return blender.frames_written / float(sr)


def _audio_codec_args(ext: str, info: MediaInfo) -> list[str]:
    """Select transparent FFmpeg encoder settings preserving original quality."""
    ext = ext.lower()
    orig_br = info.audio_bit_rate

    if ext == ".mp3":
        if orig_br and orig_br > 0:
            target = min(320_000, max(orig_br, 192_000))
            return ["-c:a", "libmp3lame", "-b:a", str(target)]
        return ["-c:a", "libmp3lame", "-q:a", "0"]

    if ext in {".m4a", ".mp4"}:
        target = min(320_000, max(orig_br or 256_000, 192_000))
        return ["-c:a", "aac", "-b:a", str(target)]

    if ext == ".mkv":
        codec = info.audio_codec.lower()
        if codec == "flac":
            return ["-c:a", "flac"]
        if codec.startswith("pcm_"):
            return ["-c:a", codec]
        if codec == "opus":
            target = min(256_000, max(orig_br or 160_000, 160_000))
            return ["-c:a", "libopus", "-b:a", str(target)]
        if codec == "mp3":
            target = min(320_000, max(orig_br or 256_000, 192_000))
            return ["-c:a", "libmp3lame", "-b:a", str(target)]
        target = min(320_000, max(orig_br or 256_000, 192_000))
        return ["-c:a", "aac", "-b:a", str(target)]

    if ext in {".webm", ".opus"}:
        target = min(256_000, max(orig_br or 160_000, 160_000))
        return ["-c:a", "libopus", "-b:a", str(target)]

    if ext == ".ogg":
        return ["-c:a", "libvorbis", "-q:a", "6"]

    if ext == ".flac":
        return ["-c:a", "flac"]

    target = min(320_000, max(orig_br or 256_000, 192_000))
    return ["-c:a", "aac", "-b:a", str(target)]


def _read_stderr_file(err_file: tempfile.TemporaryFile) -> str:
    err_file.seek(0)
    return err_file.read().decode("utf-8", errors="replace").strip()


def _stream_audio_slice_encode(
    input_path: Path,
    segments: list[Segment],
    info: MediaInfo,
    vad_result: VADResult,
    crossfade_ms: int,
    gap_samples: int,
    encoder_cmd_tail: list[str],
    output_path: Path,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> float:
    """Decode original audio in one pass, slice + crossfade + gap in numpy, pipe to encoder."""
    ffmpeg_bin, _ = require_ffmpeg()
    sr = info.sample_rate
    channels = info.channels
    fade_samples = max(0, int(round((crossfade_ms / 1000.0) * sr)))
    est_total_frames = int(round(info.duration * sr))
    ranges = _build_sample_ranges(segments, sr, est_total_frames, fade_samples)
    if not ranges:
        raise RuntimeError("No valid audio ranges remained after slicing.")

    gap_buffer = (
        _build_room_tone_buffer(
            input_path=input_path,
            info=info,
            vad_result=vad_result,
            gap_samples=gap_samples,
            fade_samples=fade_samples,
        )
        if len(ranges) > 1 and gap_samples > 0
        else None
    )

    fade_start = ranges[0][0] > 0
    fade_end = ranges[-1][1] < max(0, est_total_frames - int(0.05 * sr))

    dec_cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(input_path),
        "-to",
        f"{(ranges[-1][1] / float(sr)) + 0.25:.4f}",
        "-vn",
        "-c:a",
        "pcm_f32le",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-f",
        "f32le",
        "pipe:1",
    ]
    enc_cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        *encoder_cmd_tail,
        str(output_path),
    ]

    with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
        dec = subprocess.Popen(
            dec_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=dec_err,
        )
        enc = subprocess.Popen(
            enc_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=enc_err,
        )
        assert dec.stdout is not None and enc.stdin is not None

        def _write_pcm(arr: np.ndarray) -> None:
            enc.stdin.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

        blender = CrossfadeSegmentStream(
            sample_rate=sr,
            channels=channels,
            crossfade_ms=crossfade_ms,
            write_fn=_write_pcm,
            fade_start_edge=fade_start,
            fade_end_edge=fade_end,
        )

        frame_bytes = channels * 4
        chunk_bytes = _DECODE_CHUNK_FRAMES * frame_bytes
        pos = 0
        seg_idx = 0
        last_idx = len(ranges) - 1
        active_seg_idx: Optional[int] = None
        total_target_frames = max(1, ranges[-1][1])
        last_pct_step = -1

        try:
            while True:
                raw = dec.stdout.read(chunk_bytes)
                if not raw:
                    break
                n_frames = len(raw) // frame_bytes
                if n_frames == 0:
                    continue
                arr = np.frombuffer(raw, dtype="<f4", count=n_frames * channels).reshape(
                    n_frames, channels
                )
                chunk_start = pos
                chunk_end = pos + n_frames

                while seg_idx < len(ranges):
                    s_f, e_f = ranges[seg_idx]
                    if e_f <= chunk_start:
                        if active_seg_idx == seg_idx:
                            is_last = seg_idx == last_idx
                            blender.end_segment(is_last_segment=is_last)
                            if not is_last:
                                _feed_gap_segment(blender, gap_buffer)
                            active_seg_idx = None
                        seg_idx += 1
                        continue
                    if s_f >= chunk_end:
                        break

                    active_seg_idx = seg_idx
                    local_s = max(0, s_f - chunk_start)
                    local_e = min(n_frames, e_f - chunk_start)
                    is_last = seg_idx == last_idx
                    blender.feed(
                        arr[local_s:local_e].astype(np.float64, copy=False),
                        is_last_segment=is_last,
                    )
                    if e_f <= chunk_end:
                        blender.end_segment(is_last_segment=is_last)
                        if not is_last:
                            _feed_gap_segment(blender, gap_buffer)
                        active_seg_idx = None
                        seg_idx += 1
                    else:
                        break

                pos = chunk_end
                if progress_cb is not None:
                    frac = min(1.0, pos / float(total_target_frames))
                    pct_step = int(frac * 40)
                    if pct_step > last_pct_step:
                        last_pct_step = pct_step
                        progress_cb(frac)

            if active_seg_idx is not None:
                is_last = active_seg_idx == last_idx
                blender.end_segment(is_last_segment=is_last)
                if not is_last:
                    _feed_gap_segment(blender, gap_buffer)

            dec.stdout.close()
            enc.stdin.close()
        except Exception:
            dec.kill()
            enc.kill()
            dec.wait()
            enc.wait()
            raise

        dec_rc = dec.wait()
        enc_rc = enc.wait()
        if dec_rc != 0:
            raise RuntimeError(f"FFmpeg decode failed: {_read_stderr_file(dec_err)}")
        if enc_rc != 0:
            raise RuntimeError(f"FFmpeg encode failed: {_read_stderr_file(enc_err)}")

        return blender.frames_written / float(sr)


def _process_compressed_audio(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    info: MediaInfo,
    vad_result: VADResult,
    crossfade_ms: int,
    gap_ms: int,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> float:
    """Process compressed audio (.mp3, .m4a, .webm, etc.) preserving sample rate and tags."""
    ext = output_path.suffix.lower()
    codec_args = _audio_codec_args(ext, info)

    extra_args: list[str] = [
        "-i",
        str(input_path),
        "-map",
        "0:a",
        "-map_metadata",
        "1",
    ]
    if info.has_attached_pic and ext in {".mp3", ".m4a"}:
        extra_args.extend(
            ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        )
    if ext == ".m4a":
        extra_args.extend(["-movflags", "+faststart"])

    gap_samples = max(0, int(round((gap_ms / 1000.0) * info.sample_rate)))
    encoder_tail = [*extra_args, *codec_args]
    return _stream_audio_slice_encode(
        input_path=input_path,
        segments=segments,
        info=info,
        vad_result=vad_result,
        crossfade_ms=crossfade_ms,
        gap_samples=gap_samples,
        encoder_cmd_tail=encoder_tail,
        output_path=output_path,
        progress_cb=progress_cb,
    )


def _parse_y4m_frame_bytes(header: bytes) -> int:
    tokens = header.decode("ascii", errors="strict").strip().split()
    width = next(int(t[1:]) for t in tokens if t.startswith("W"))
    height = next(int(t[1:]) for t in tokens if t.startswith("H"))
    c_tok = next((t for t in tokens if t.startswith("C")), "C420")
    mult = _Y4M_BYTES_PER_PIXEL.get(c_tok)
    if mult is None:
        for key, val in _Y4M_BYTES_PER_PIXEL.items():
            if c_tok.startswith(key):
                mult = val
                break
        if mult is None:
            raise RuntimeError(f"Unsupported Y4M pixel format token: {c_tok}")
    return int(round(width * height * mult))


def _snap_video_segments(
    segments: list[Segment],
    fps: Fraction,
    crossfade_ms: int,
) -> list[tuple[int, int, Segment]]:
    """Snap kept time intervals onto exact video frame boundaries and merge overlaps."""
    raw_ranges: list[tuple[int, int, Segment]] = []
    for seg in segments:
        f_start = max(0, int(round(seg.start * fps)))
        f_end = max(f_start + 1, int(round(seg.end * fps)))
        raw_ranges.append((f_start, f_end, seg))

    min_frame_gap = int(round((crossfade_ms / 1000.0) * float(fps)))
    merged: list[tuple[int, int, Segment]] = []
    for f_start, f_end, seg in raw_ranges:
        if not merged:
            merged.append((f_start, f_end, seg))
            continue
        prev_s, prev_e, prev_seg = merged[-1]
        if f_start - prev_e <= min_frame_gap:
            new_end = max(prev_e, f_end)
            merged[-1] = (
                prev_s,
                new_end,
                Segment(
                    start=round(float(Fraction(prev_s) / fps), 4),
                    end=round(float(Fraction(new_end) / fps), 4),
                    rms_db=prev_seg.rms_db,
                    core_db=prev_seg.core_db,
                ),
            )
        else:
            merged.append(
                (
                    f_start,
                    f_end,
                    Segment(
                        start=round(float(Fraction(f_start) / fps), 4),
                        end=round(float(Fraction(f_end) / fps), 4),
                        rms_db=seg.rms_db,
                        core_db=seg.core_db,
                    ),
                )
            )
    return merged


def _process_video(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    info: MediaInfo,
    vad_result: VADResult,
    crossfade_ms: int,
    gap_ms: int,
) -> tuple[float, list[Segment]]:
    """Slice video via frame-exact yuv4mpegpipe stream, hold gap frames, mux crossfaded audio."""
    ffmpeg_bin, _ = require_ffmpeg()
    fps = info.video_fps or Fraction(30, 1)
    snapped = _snap_video_segments(segments, fps, crossfade_ms)
    gap_frames = max(0, int(round((gap_ms / 1000.0) * float(fps)))) if len(snapped) > 1 else 0
    actual_gap_s = float(Fraction(gap_frames) / fps)
    gap_samples = int(round(actual_gap_s * info.sample_rate))
    ext = output_path.suffix.lower()

    with tempfile.TemporaryDirectory(prefix="quran_trim_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        temp_video = tmp_dir / f"video_only{ext}"

        dec_cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            str(input_path),
            "-an",
            "-vf",
            f"fps=fps={fps.numerator}/{fps.denominator}",
            "-f",
            "yuv4mpegpipe",
            "pipe:1",
        ]
        enc_video_cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "yuv4mpegpipe",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(temp_video),
        ]

        actual_segments: list[Segment] = []
        with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
            dec = subprocess.Popen(
                dec_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=dec_err,
            )
            enc = subprocess.Popen(
                enc_video_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=enc_err,
            )
            assert dec.stdout is not None and enc.stdin is not None

            try:
                header = dec.stdout.readline()
                if not header.startswith(b"YUV4MPEG2"):
                    raise RuntimeError(f"Unexpected Y4M stream header: {header!r}")
                frame_bytes = _parse_y4m_frame_bytes(header)
                enc.stdin.write(header)

                seg_idx = 0
                frame_idx = 0
                seg_frames_written = 0
                last_tag = b""
                last_buf = b""

                def _emit_hold_gap() -> None:
                    if gap_frames > 0 and last_tag and last_buf:
                        for _ in range(gap_frames):
                            enc.stdin.write(last_tag)
                            enc.stdin.write(last_buf)

                while True:
                    tag = dec.stdout.readline()
                    if not tag:
                        break
                    buf = dec.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break

                    while seg_idx < len(snapped) and frame_idx >= snapped[seg_idx][1]:
                        if seg_frames_written > 0:
                            s_f = snapped[seg_idx][0]
                            e_f = s_f + seg_frames_written
                            orig_seg = snapped[seg_idx][2]
                            actual_segments.append(
                                Segment(
                                    start=float(Fraction(s_f) / fps),
                                    end=float(Fraction(e_f) / fps),
                                    rms_db=orig_seg.rms_db,
                                    core_db=orig_seg.core_db,
                                )
                            )
                            if seg_idx + 1 < len(snapped):
                                _emit_hold_gap()
                        seg_idx += 1
                        seg_frames_written = 0

                    if seg_idx < len(snapped):
                        s_f, e_f, _ = snapped[seg_idx]
                        if s_f <= frame_idx < e_f:
                            enc.stdin.write(tag)
                            enc.stdin.write(buf)
                            last_tag = tag
                            last_buf = buf
                            seg_frames_written += 1

                    frame_idx += 1

                if seg_idx < len(snapped) and seg_frames_written > 0:
                    s_f = snapped[seg_idx][0]
                    e_f = s_f + seg_frames_written
                    orig_seg = snapped[seg_idx][2]
                    actual_segments.append(
                        Segment(
                            start=float(Fraction(s_f) / fps),
                            end=float(Fraction(e_f) / fps),
                            rms_db=orig_seg.rms_db,
                            core_db=orig_seg.core_db,
                        )
                    )

                dec.stdout.close()
                enc.stdin.close()
            except Exception:
                dec.kill()
                enc.kill()
                dec.wait()
                enc.wait()
                raise

            dec_rc = dec.wait()
            enc_rc = enc.wait()
            if dec_rc != 0:
                raise RuntimeError(f"FFmpeg video decode failed: {_read_stderr_file(dec_err)}")
            if enc_rc != 0:
                raise RuntimeError(f"FFmpeg video encode failed: {_read_stderr_file(enc_err)}")

        if not actual_segments:
            raise RuntimeError("No video frames remained after slicing.")

        audio_codec_args = _audio_codec_args(ext, info)
        mux_tail = [
            "-i",
            str(temp_video),
            "-i",
            str(input_path),
            "-map",
            "1:v",
            "-map",
            "0:a",
            "-map_metadata",
            "2",
            "-c:v",
            "copy",
            *audio_codec_args,
        ]
        if ext == ".mp4":
            mux_tail.extend(["-movflags", "+faststart"])

        final_duration = _stream_audio_slice_encode(
            input_path=input_path,
            segments=actual_segments,
            info=info,
            vad_result=vad_result,
            crossfade_ms=crossfade_ms,
            gap_samples=gap_samples,
            encoder_cmd_tail=mux_tail,
            output_path=output_path,
        )

        rounded_segments = [
            Segment(
                start=round(s.start, 4),
                end=round(s.end, 4),
                rms_db=s.rms_db,
                core_db=s.core_db,
            )
            for s in actual_segments
        ]
        return final_duration, rounded_segments


def write_timestamps_json(
    timestamps_path: Path,
    input_path: Path,
    output_path: Path,
    config: VADConfig,
    vad_result: VADResult,
    final_segments: list[Segment],
    final_duration: float,
    source_duration: float,
    clip_start_s: Optional[float] = None,
    clip_end_s: Optional[float] = None,
) -> None:
    """Write a structured JSON log of all kept segments, profile info, and stats."""
    original = vad_result.total_duration
    saved_s = max(0.0, original - final_duration)
    saved_pct = (saved_s / original * 100.0) if original > 0 else 0.0

    payload: dict[str, object] = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "mode": config.mode,
        "profile": {
            "detected": vad_result.profile.kind,
            "noise_floor_p5_db": vad_result.profile.noise_floor_p5_db,
            "noise_floor_p10_db": vad_result.profile.noise_floor_p10_db,
            "p90_db": vad_result.profile.p90_db,
            "snr_db": vad_result.profile.snr_db,
        },
        "settings": {
            "threshold": config.threshold,
            "pad_start_ms": config.pad_start_ms,
            "pad_end_ms": config.pad_end_ms,
            "min_silence_ms": config.min_silence_ms,
            "crossfade_ms": config.crossfade_ms,
            "gap_ms": config.gap_ms,
            "strict_margin_db": config.strict_margin_db if config.mode == "strict" else None,
            "clip_start_s": round(clip_start_s, 4) if clip_start_s is not None else None,
            "clip_end_s": round(clip_end_s, 4) if clip_end_s is not None else None,
        },
        "summary": {
            "source_duration_s": round(source_duration, 4),
            "original_duration_s": round(original, 4),
            "final_duration_s": round(final_duration, 4),
            "time_saved_s": round(saved_s, 4),
            "time_saved_pct": round(saved_pct, 2),
            "segments_kept": len(final_segments),
            "prayer_windows_detected": vad_result.prayer_windows_count,
            "prayer_segments_stripped": vad_result.prayer_stripped_count,
        },
        "segments": [
            {
                "index": i + 1,
                "start_s": round(seg.start, 4),
                "end_s": round(seg.end, 4),
                "duration_s": round(seg.duration, 4),
                "start_tc": _format_timecode(seg.start),
                "end_tc": _format_timecode(seg.end),
                "rms_db": seg.rms_db,
                "core_db": seg.core_db,
            }
            for i, seg in enumerate(final_segments)
        ],
    }
    if config.mode == "strict":
        payload["calibration"] = {
            "speech_p25_db": vad_result.speech_p25_db,
            "reciter_median_db": vad_result.reciter_median_db,
            "strict_cutoff_db": vad_result.strict_cutoff_db,
            "raw_speech_islands": vad_result.raw_segments_count,
            "kept_speech_islands": vad_result.filtered_segments_count,
        }

    timestamps_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_clean_media(
    input_path: Path,
    output_path: Path,
    timestamps_path: Path,
    info: MediaInfo,
    config: VADConfig,
    vad_result: VADResult,
    clip_start_s: Optional[float] = None,
    clip_end_s: Optional[float] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> ProcessResult:
    """Slice and concatenate the original stream at native quality with 20ms crossfades."""
    if not vad_result.segments:
        raise RuntimeError(
            "No speech segments were detected above the configured thresholds. "
            "Try lowering --threshold or switching to --mode natural."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    final_segments = vad_result.segments
    out_ext = output_path.suffix.lower()

    try:
        if info.has_video and out_ext not in _AUDIO_ONLY_EXTENSIONS:
            final_duration, final_segments = _process_video(
                input_path=input_path,
                output_path=tmp_output,
                segments=vad_result.segments,
                info=info,
                vad_result=vad_result,
                crossfade_ms=config.crossfade_ms,
                gap_ms=config.gap_ms,
            )
            if progress_cb is not None:
                progress_cb(1.0)
        elif input_path.suffix.lower() == ".wav" and out_ext == ".wav":
            final_duration = _process_wav_soundfile(
                input_path=input_path,
                output_path=tmp_output,
                segments=vad_result.segments,
                info=info,
                vad_result=vad_result,
                crossfade_ms=config.crossfade_ms,
                gap_ms=config.gap_ms,
                progress_cb=progress_cb,
            )
        else:
            final_duration = _process_compressed_audio(
                input_path=input_path,
                output_path=tmp_output,
                segments=vad_result.segments,
                info=info,
                vad_result=vad_result,
                crossfade_ms=config.crossfade_ms,
                gap_ms=config.gap_ms,
                progress_cb=progress_cb,
            )

        os.replace(tmp_output, output_path)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink(missing_ok=True)
        raise

    write_timestamps_json(
        timestamps_path=timestamps_path,
        input_path=input_path,
        output_path=output_path,
        config=config,
        vad_result=vad_result,
        final_segments=final_segments,
        final_duration=final_duration,
        source_duration=info.duration,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )

    original = vad_result.total_duration
    saved_pct = max(0.0, (original - final_duration) / original * 100.0) if original > 0 else 0.0

    return ProcessResult(
        input_path=input_path,
        output_path=output_path,
        timestamps_path=timestamps_path,
        original_duration=original,
        final_duration=final_duration,
        time_saved_pct=saved_pct,
        segments_kept=len(final_segments),
        segments=final_segments,
    )
