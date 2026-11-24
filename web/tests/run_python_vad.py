"""Subprocess helper that runs the existing Python VAD engine on a raw f32le 16kHz buffer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quran_cleaner.vad import VADConfig, detect_speech_segments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Python quran_cleaner VAD on raw float32 audio.")
    parser.add_argument("raw_f32_path", type=Path, help="Path to little-endian float32 16kHz mono PCM.")
    parser.add_argument("--duration", type=float, default=None, help="Optional exact clip duration in seconds.")
    parser.add_argument("--clip-start", type=float, default=0.0, help="Clip start offset in seconds.")
    parser.add_argument("--config-json", type=str, default=None, help="Optional JSON VADConfig overrides.")
    args = parser.parse_args()

    audio_16k = np.fromfile(args.raw_f32_path, dtype="<f4")
    cfg_kwargs = json.loads(args.config_json) if args.config_json else {}
    cfg = VADConfig(**cfg_kwargs)

    res = detect_speech_segments(
        audio_16k=audio_16k,
        config=cfg,
        exact_duration=args.duration,
        clip_start_s=args.clip_start,
    )

    payload = {
        "total_duration": res.total_duration,
        "profile": {
            "kind": res.profile.kind,
            "noise_floor_p5_db": res.profile.noise_floor_p5_db,
            "noise_floor_p10_db": res.profile.noise_floor_p10_db,
            "p90_db": res.profile.p90_db,
            "snr_db": res.profile.snr_db,
        },
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "duration": round(seg.duration, 4),
                "rms_db": seg.rms_db,
                "core_db": seg.core_db,
            }
            for seg in res.segments
        ],
        "raw_segments_count": res.raw_segments_count,
        "filtered_segments_count": res.filtered_segments_count,
        "quietest_gap_window": list(res.quietest_gap_window) if res.quietest_gap_window is not None else None,
        "resolved_pad_start_ms": res.resolved_pad_start_ms,
        "resolved_pad_end_ms": res.resolved_pad_end_ms,
        "resolved_min_silence_ms": res.resolved_min_silence_ms,
        "resolved_crossfade_ms": res.resolved_crossfade_ms,
        "resolved_gap_ms": res.resolved_gap_ms,
        "use_room_tone": res.use_room_tone,
        "direct_overlap": res.direct_overlap,
        "low_snr_guarded": res.low_snr_guarded,
        "stray_shouts_stripped": res.stray_shouts_stripped,
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
