"""Command-line interface and formatted terminal summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .media import (
    download_external_audio,
    extract_vad_audio,
    is_external_url,
    parse_timestamp,
    probe_media,
    require_ffmpeg,
    validate_local_media,
)
from .processor import ProcessResult, render_clean_media
from .vad import VADConfig, VADResult, detect_speech_segments


def _format_duration(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs = rem_ms / 1000.0
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f} ({seconds:.2f}s)"
    return f"{minutes:02d}:{secs:05.2f} ({seconds:.2f}s)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quran-trim",
        description=(
            "Zero-config Quran recitation silence, breath, and murmur gap remover "
            "using an auto-profiled dual-rate Silero VAD + vocal-band energy pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help=(
            "Path to local audio/video file (.mp3, .wav, .m4a, .mp4, .mkv, .webm) "
            "or any external URL supported by yt-dlp (YouTube, SoundCloud, Archive.org, direct audio links)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["natural", "strict"],
        default="natural",
        help=(
            "natural (default): Auto-profiled studio/vintage pipeline preserving soft "
            "endings and hall ambience while filtering isolated crowd murmurs; "
            "strict (advanced): adaptive p25 RMS filter to drop distant crowd shouts."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="TIME",
        help="Optional clip start timestamp (MM:SS or HH:MM:SS).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="TIME",
        help="Optional clip end timestamp (MM:SS or HH:MM:SS).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Silero VAD speech probability threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--pad-start",
        type=int,
        default=None,
        metavar="MS",
        help="Padding kept before each speech onset in ms (auto: 650ms vintage/live, 200ms studio).",
    )
    parser.add_argument(
        "--pad-end",
        type=int,
        default=None,
        metavar="MS",
        help="Padding kept after each speech ending in ms (auto: 1100-1300ms vintage/live, 300ms studio).",
    )
    parser.add_argument(
        "--min-silence",
        type=int,
        default=None,
        metavar="MS",
        help="Minimum silence duration in ms before splitting (auto-calibrated if omitted).",
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=None,
        metavar="MS",
        help="Equal-power audio crossfade duration at splice boundaries in ms (auto if omitted).",
    )
    parser.add_argument(
        "--gap-ms",
        type=int,
        default=None,
        metavar="MS",
        help="Room-tone/silence breathing gap inserted between segments in ms (auto if omitted).",
    )
    parser.add_argument(
        "--strict-margin-db",
        type=float,
        default=12.0,
        metavar="DB",
        help="[Advanced] Strict mode dB margin below the lower-percentile (p25) speech floor.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output file path (defaults to <filename>_clean.<ext>).",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="Optional explicit path for timestamps.json (defaults next to the input).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save YouTube downloads (defaults to current directory).",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[Optional[float], Optional[float]]:
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be strictly between 0.0 and 1.0.")
    if args.pad_start is not None and args.pad_start < 0:
        raise ValueError("--pad-start must be >= 0 ms.")
    if args.pad_end is not None and args.pad_end < 0:
        raise ValueError("--pad-end must be >= 0 ms.")
    if args.min_silence is not None and args.min_silence < 0:
        raise ValueError("--min-silence must be >= 0 ms.")
    if args.crossfade is not None and args.crossfade < 0:
        raise ValueError("--crossfade must be >= 0 ms.")
    if args.gap_ms is not None and args.gap_ms < 0:
        raise ValueError("--gap-ms must be >= 0 ms.")
    if args.strict_margin_db < 0:
        raise ValueError("--strict-margin-db must be >= 0 dB.")

    start_s = parse_timestamp(args.start) if args.start is not None else None
    end_s = parse_timestamp(args.end) if args.end is not None else None
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError("--start must be strictly earlier than --end.")
    return start_s, end_s


def _print_summary(
    result: ProcessResult,
    mode: str,
    gap_ms: int,
    vad_result: VADResult,
    clip_start_s: Optional[float],
    clip_end_s: Optional[float],
) -> None:
    prof = vad_result.profile
    print()
    print("Quran Recitation Cleanup Summary")
    print("-" * 48)
    print(f"Mode                : {mode}")
    print(
        f"Detected Profile    : {prof.kind} "
        f"(floor p5: {prof.noise_floor_p5_db:.1f} dBFS, SNR: {prof.snr_db:.1f} dB)"
    )
    if clip_start_s is not None or clip_end_s is not None:
        s_lbl = _format_duration(clip_start_s or 0.0).split(" (")[0]
        e_lbl = _format_duration(clip_end_s or vad_result.total_duration).split(" (")[0]
        print(f"Clip Range          : {s_lbl} -> {e_lbl}")
    if vad_result.direct_overlap:
        gap_kind = "direct reverb overlap"
    elif vad_result.use_room_tone:
        gap_kind = "room-tone bridge"
    else:
        gap_kind = "digital silence"
    print(
        f"Adaptive Tuning     : min-silence {vad_result.resolved_min_silence_ms} ms, "
        f"pad {vad_result.resolved_pad_start_ms}/{vad_result.resolved_pad_end_ms} ms, "
        f"gap {gap_ms} ms ({gap_kind}), xfade {vad_result.resolved_crossfade_ms} ms"
    )
    if vad_result.stray_shouts_stripped > 0:
        print(f"Crowd Filtering     : {vad_result.stray_shouts_stripped} isolated background murmur/shout segments stripped")
    if (
        mode == "strict"
        and vad_result.reciter_median_db is not None
        and vad_result.strict_cutoff_db is not None
    ):
        if vad_result.speech_p25_db is not None:
            print(f"Speech Floor (p25)  : {vad_result.speech_p25_db:.2f} dBFS")
        print(f"Reciter Median RMS  : {vad_result.reciter_median_db:.2f} dBFS")
        print(f"Strict Cutoff       : {vad_result.strict_cutoff_db:.2f} dBFS")
    print(f"Original Duration   : {_format_duration(result.original_duration)}")
    print(f"Final Duration      : {_format_duration(result.final_duration)}")
    print(f"Time Saved (%)      : {result.time_saved_pct:.2f}%")
    print(f"Total Segments Kept : {result.segments_kept}")
    print("-" * 48)
    print(f"Cleaned File        : {result.output_path}")
    print(f"Timestamps Log      : {result.timestamps_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        start_s, end_s = _validate_args(args)
        require_ffmpeg()

        if is_external_url(args.input):
            dl_dir = (args.output_dir or Path.cwd()).expanduser().resolve()
            print(f"Downloading best audio from URL into: {dl_dir}")
            input_path = download_external_audio(args.input, dl_dir)
            print(f"Downloaded: {input_path.name}")
        else:
            input_path = validate_local_media(Path(args.input))

        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
        )
        timestamps_path = (
            args.timestamps.expanduser().resolve()
            if args.timestamps is not None
            else output_path.parent / "timestamps.json"
        )

        info = probe_media(input_path)
        clip_start = start_s if start_s is not None else 0.0
        clip_end = min(end_s, info.duration) if end_s is not None else info.duration
        if clip_start >= info.duration:
            raise ValueError(
                f"--start ({clip_start:.2f}s) exceeds media duration ({info.duration:.2f}s)."
            )
        if clip_start >= clip_end:
            raise ValueError("--start must be strictly earlier than --end.")

        clip_duration = clip_end - clip_start
        range_note = (
            f", range: {clip_start:.1f}s..{clip_end:.1f}s"
            if (start_s is not None or end_s is not None)
            else ""
        )
        print(
            f"Analyzing 16kHz VAD stream "
            f"(source: {info.sample_rate} Hz, {info.channels}ch, {info.audio_codec}{range_note})..."
        )
        audio_16k = extract_vad_audio(
            input_path,
            start_s=start_s,
            end_s=clip_end if end_s is not None else None,
        )

        config = VADConfig(
            mode=args.mode,
            threshold=args.threshold,
            pad_start_ms=args.pad_start,
            pad_end_ms=args.pad_end,
            min_silence_ms=args.min_silence,
            crossfade_ms=args.crossfade,
            gap_ms=args.gap_ms,
            strict_margin_db=args.strict_margin_db,
        )
        vad_result = detect_speech_segments(
            audio_16k=audio_16k,
            config=config,
            exact_duration=clip_duration,
            clip_start_s=clip_start,
        )

        print(
            f"Slicing original stream ({len(vad_result.segments)} segments, "
            f"{vad_result.resolved_crossfade_ms}ms crossfade, {vad_result.resolved_gap_ms}ms gap)..."
        )
        result = render_clean_media(
            input_path=input_path,
            output_path=output_path,
            timestamps_path=timestamps_path,
            info=info,
            config=config,
            vad_result=vad_result,
            clip_start_s=start_s,
            clip_end_s=clip_end if end_s is not None else None,
        )
        _print_summary(
            result=result,
            mode=args.mode,
            gap_ms=vad_result.resolved_gap_ms,
            vad_result=vad_result,
            clip_start_s=start_s,
            clip_end_s=clip_end if end_s is not None else None,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
