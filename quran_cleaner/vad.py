"""Hybrid Silero VAD + auto-profiler, low-SNR safety gate, studio breath filter, and crowd murmur cluster filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np

from .media import VAD_SAMPLE_RATE

ProfileType = Literal["studio", "vintage/live"]

_FRAME_SAMPLES = 512  # 32ms windows at 16kHz (matches Silero VAD windowing)
_MIN_SPEECH_MS = 200
_STUDIO_MIN_SPEECH_MS = 220
_VAD_CHUNK_SECONDS = 60
_VAD_OVERLAP_SECONDS = 2
_LOW_SNR_GUARD_DB = 12.0
_MIN_VINTAGE_EFFECTIVE_CUT_S = 1.2


@dataclass(frozen=True)
class VADConfig:
    """Configuration for hybrid Silero VAD, auto-profiling, padding, and gaps."""

    threshold: float = 0.20
    pad_start_ms: Optional[int] = None
    pad_end_ms: Optional[int] = None
    min_silence_ms: Optional[int] = None
    crossfade_ms: Optional[int] = None
    gap_ms: Optional[int] = None


@dataclass(frozen=True)
class Segment:
    """A retained audio/video time interval in seconds."""

    start: float
    end: float
    rms_db: Optional[float] = None
    core_db: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioProfile:
    """Acoustic environment profile measured from the temporary 16kHz analysis stream."""

    kind: ProfileType
    noise_floor_p5_db: float
    noise_floor_p10_db: float
    p90_db: float
    snr_db: float


@dataclass(frozen=True)
class VADResult:
    """Outputs and calibration diagnostics from the VAD pipeline."""

    segments: list[Segment]
    total_duration: float
    profile: AudioProfile
    raw_segments_count: int
    filtered_segments_count: int
    quietest_gap_window: Optional[tuple[float, float, float]] = None
    resolved_pad_start_ms: int = 200
    resolved_pad_end_ms: int = 300
    resolved_min_silence_ms: int = 700
    resolved_crossfade_ms: int = 20
    resolved_gap_ms: int = 450
    use_room_tone: bool = False
    direct_overlap: bool = False
    low_snr_guarded: bool = False
    stray_shouts_stripped: int = 0


def _segment_frame_dbs(
    analysis_audio: np.ndarray, start_idx: int, end_idx: int
) -> tuple[np.ndarray, float]:
    """Return 32ms frame RMS dBFS values and overall segment linear RMS dBFS."""
    seg = analysis_audio[start_idx:end_idx]
    if seg.size == 0:
        return np.array([-120.0], dtype=np.float64), -120.0

    overall_rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    overall_db = float(20.0 * np.log10(overall_rms))

    n_full = seg.size // _FRAME_SAMPLES
    if n_full == 0:
        return np.array([overall_db], dtype=np.float64), overall_db

    frames = seg[: n_full * _FRAME_SAMPLES].reshape(n_full, _FRAME_SAMPLES)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    frame_dbs = 20.0 * np.log10(frame_rms)

    rem = seg[n_full * _FRAME_SAMPLES :]
    if rem.size >= _FRAME_SAMPLES // 4:
        rem_db = float(20.0 * np.log10(np.sqrt(np.mean(rem * rem) + 1e-12)))
        frame_dbs = np.append(frame_dbs, rem_db)

    return frame_dbs, overall_db


def _hf_breath_ratio(seg: np.ndarray) -> float:
    """Return ratio of spectral energy >= 2.8 kHz relative to 80-7800 Hz speech band."""
    if seg.size < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / VAD_SAMPLE_RATE)
    total = float(np.sum(spec[(freqs >= 80.0) & (freqs <= 7800.0)])) + 1e-12
    hf = float(np.sum(spec[(freqs >= 2800.0) & (freqs <= 7800.0)]))
    return hf / total


def _compute_frame_dbs(analysis_audio: np.ndarray) -> np.ndarray:
    """Compute 32ms non-overlapping frame RMS levels in dBFS."""
    n_frames = analysis_audio.size // _FRAME_SAMPLES
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    frames = analysis_audio[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
    return 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1) + 1e-12))


def _profile_audio(
    analysis_audio: np.ndarray,
    frame_dbs: Optional[np.ndarray] = None,
) -> AudioProfile:
    """Classify recording into 'studio' (low noise floor < -55 dBFS) vs 'vintage/live'."""
    if frame_dbs is None:
        frame_dbs = _compute_frame_dbs(analysis_audio)
    if frame_dbs.size == 0:
        return AudioProfile("vintage/live", -45.0, -45.0, -15.0, 30.0)

    non_zero = frame_dbs[frame_dbs > -85.0]
    if non_zero.size == 0:
        return AudioProfile("studio", -80.0, -80.0, -80.0, 0.0)

    nf_p5 = float(np.percentile(non_zero, 5))
    nf_p10 = float(np.percentile(non_zero, 10))
    p90 = float(np.percentile(non_zero, 90))
    snr = max(0.0, p90 - nf_p10)
    kind: ProfileType = "studio" if nf_p5 < -55.0 else "vintage/live"

    return AudioProfile(
        kind=kind,
        noise_floor_p5_db=round(nf_p5, 2),
        noise_floor_p10_db=round(nf_p10, 2),
        p90_db=round(p90, 2),
        snr_db=round(snr, 2),
    )


def _merge_intervals(
    intervals: list[tuple[int, int]], bridge_samples: int = 0
) -> list[tuple[int, int]]:
    """Sort and merge overlapping or closely spaced sample intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for s_idx, e_idx in ordered[1:]:
        prev_s, prev_e = merged[-1]
        if s_idx <= prev_e + bridge_samples:
            merged[-1] = (prev_s, max(prev_e, e_idx))
        else:
            merged.append((s_idx, e_idx))
    return merged


def _apply_padding_and_merge(
    segments: list[Segment],
    pad_start_s: float,
    pad_end_s: float | Sequence[float],
    min_bound: float,
    max_bound: float,
    min_splice_gap_s: float,
    min_effective_cut_s: float = 0.0,
) -> list[Segment]:
    """Apply asymmetric start/end padding and merge sub-crossfade or sub-threshold gaps."""
    if not segments:
        return []

    padded: list[Segment] = []
    for idx, seg in enumerate(segments):
        p_end = pad_end_s[idx] if isinstance(pad_end_s, (list, tuple)) else float(pad_end_s)
        padded.append(
            Segment(
                start=max(min_bound, seg.start - pad_start_s),
                end=min(max_bound, seg.end + p_end),
            )
        )

    merged: list[Segment] = [padded[0]]
    for seg in padded[1:]:
        prev = merged[-1]
        gap_between = seg.start - prev.end
        if gap_between <= min_splice_gap_s or (
            min_effective_cut_s > 0.0 and gap_between < min_effective_cut_s
        ):
            merged[-1] = Segment(start=prev.start, end=max(prev.end, seg.end))
        else:
            merged.append(seg)
    return merged


def _run_silero_vad_chunked(
    audio_16k: np.ndarray,
    threshold: float,
    island_bridge_ms: int,
    min_speech_ms: int = _MIN_SPEECH_MS,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> list[tuple[int, int]]:
    """Run Silero VAD (ONNX) in 60s overlapping chunks so RNN states never drift."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=True)
    peak = float(np.max(np.abs(audio_16k)))
    norm_audio = audio_16k / peak if peak > 1.0 else audio_16k

    chunk_len = _VAD_CHUNK_SECONDS * VAD_SAMPLE_RATE
    overlap_len = _VAD_OVERLAP_SECONDS * VAD_SAMPLE_RATE
    step_len = max(VAD_SAMPLE_RATE, chunk_len - overlap_len)

    raw_intervals: list[tuple[int, int]] = []
    total_len = norm_audio.size

    pos = 0
    while pos < total_len:
        end_pos = min(total_len, pos + chunk_len)
        chunk_tensor = torch.from_numpy(norm_audio[pos:end_pos])
        model.reset_states()
        ts_list = get_speech_timestamps(
            chunk_tensor,
            model,
            threshold=threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=island_bridge_ms,
            speech_pad_ms=0,
            return_seconds=False,
        )
        for ts in ts_list:
            raw_intervals.append((pos + int(ts["start"]), pos + int(ts["end"])))
        if progress_cb is not None and total_len > 0:
            progress_cb(min(1.0, end_pos / float(total_len)))
        if end_pos >= total_len:
            break
        pos += step_len

    bridge_samples = int(round((island_bridge_ms / 1000.0) * VAD_SAMPLE_RATE))
    return _merge_intervals(raw_intervals, bridge_samples=bridge_samples)


def _detect_energy_islands(
    analysis_audio: np.ndarray,
    profile: AudioProfile,
    island_bridge_ms: int,
    frame_dbs: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Detect melodic/sustained vocal islands via SNR-aware hysteresis RMS gate."""
    if frame_dbs is None:
        frame_dbs = _compute_frame_dbs(analysis_audio)
    n_frames = frame_dbs.size
    if n_frames == 0:
        return []

    spread = profile.snr_db
    is_studio = profile.kind == "studio"

    if is_studio:
        # On clean studio recordings, run strictly as a high-level melodic-sustain
        # safety net above breaths so Maqam holds & quiet passages stay intact.
        if spread < 15.0:
            return []
        onset_db = max(profile.noise_floor_p5_db + 16.0, profile.p90_db - 26.0)
        sustain_db = max(profile.noise_floor_p5_db + 12.0, onset_db - 4.0)
        min_island_dur = 0.40
    else:
        if spread < 6.0:
            return []
        if spread >= 22.0:
            onset_db = max(profile.noise_floor_p10_db + 15.0, profile.p90_db - 25.0)
            sustain_db = onset_db - 4.0
        else:
            # Moderate/low-SNR hall ambiance: stay >= 2.5 dB below onset so soft endings hold.
            onset_db = max(profile.noise_floor_p10_db + 0.52 * spread, profile.p90_db - 25.0)
            sustain_db = onset_db - max(2.5, min(3.5, 0.18 * spread))
        min_island_dur = 0.35

    raw_islands: list[tuple[int, int]] = []
    in_island = False
    start_f = 0

    for idx, val_db in enumerate(frame_dbs):
        if not in_island and val_db >= onset_db:
            start_f = idx
            in_island = True
        elif in_island and val_db < sustain_db:
            dur_s = (idx - start_f) * _FRAME_SAMPLES / float(VAD_SAMPLE_RATE)
            peak_db = float(np.max(frame_dbs[start_f:idx]))
            if dur_s >= min_island_dur or (not is_studio and peak_db >= onset_db + 4.0):
                raw_islands.append((start_f * _FRAME_SAMPLES, idx * _FRAME_SAMPLES))
            in_island = False

    if in_island:
        dur_s = (n_frames - start_f) * _FRAME_SAMPLES / float(VAD_SAMPLE_RATE)
        peak_db = float(np.max(frame_dbs[start_f:]))
        if dur_s >= min_island_dur or (not is_studio and peak_db >= onset_db + 4.0):
            raw_islands.append((start_f * _FRAME_SAMPLES, analysis_audio.size))

    bridge_samples = int(round((island_bridge_ms / 1000.0) * VAD_SAMPLE_RATE))
    bridged = _merge_intervals(raw_islands, bridge_samples=bridge_samples)
    min_dur_ms = _STUDIO_MIN_SPEECH_MS if is_studio else _MIN_SPEECH_MS
    min_samples = int(round((min_dur_ms / 1000.0) * VAD_SAMPLE_RATE))
    return [(s, e) for s, e in bridged if (e - s) >= min_samples]


def _filter_studio_breaths(
    analysis_audio: np.ndarray,
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Reject short close-mic inhalation breaths on clean studio recordings."""
    if not intervals:
        return []

    min_samples = int(round((_STUDIO_MIN_SPEECH_MS / 1000.0) * VAD_SAMPLE_RATE))
    stats: list[tuple[int, int, float, float]] = []
    for s_idx, e_idx in intervals:
        if e_idx - s_idx < min_samples:
            continue
        _, rms_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        hf_ratio = _hf_breath_ratio(analysis_audio[s_idx:e_idx])
        stats.append((s_idx, e_idx, rms_db, hf_ratio))

    if not stats:
        return []

    ref_db = float(np.percentile([x[2] for x in stats], 75))
    kept: list[tuple[int, int]] = []
    for s_idx, e_idx, rms_db, hf_ratio in stats:
        dur_s = (e_idx - s_idx) / float(VAD_SAMPLE_RATE)
        is_breath = dur_s < 0.90 and hf_ratio >= 0.22 and rms_db < (ref_db - 10.0)
        if not is_breath:
            kept.append((s_idx, e_idx))
    return kept


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    """Compute duration-weighted step quantile so crowd outliers cannot smear into p25."""
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    total = float(cum_weights[-1])
    if total <= 0.0:
        return float(np.percentile(values, pct))
    idx = int(np.searchsorted(cum_weights / total, pct / 100.0, side="left"))
    idx = min(max(idx, 0), sorted_vals.size - 1)
    return float(sorted_vals[idx])


def _filter_unconfirmed_noise_blips(
    analysis_audio: np.ndarray,
    intervals: list[tuple[int, int]],
    silero_intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Drop short low-energy energy-only cough/thump blips unconfirmed by Silero VAD."""
    if len(intervals) < 3 or not silero_intervals:
        return intervals

    stats: list[tuple[int, int, float, float, bool]] = []
    sil_idx = 0
    n_sil = len(silero_intervals)

    for s_idx, e_idx in intervals:
        frame_dbs, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        peak_db = float(np.max(frame_dbs))
        while sil_idx < n_sil and silero_intervals[sil_idx][1] <= s_idx:
            sil_idx += 1
        j = sil_idx
        confirmed = False
        while j < n_sil and silero_intervals[j][0] < e_idx:
            if silero_intervals[j][1] > s_idx:
                confirmed = True
                break
            j += 1
        stats.append((s_idx, e_idx, overall_db, peak_db, confirmed))

    durs = np.array(
        [max(1, e - s) / float(VAD_SAMPLE_RATE) for s, e, _, _, _ in stats],
        dtype=np.float64,
    )
    ov_vals = np.array([x[2] for x in stats], dtype=np.float64)
    med_db = _weighted_percentile(ov_vals, durs, 50.0)

    kept: list[tuple[int, int]] = []
    for s_idx, e_idx, overall_db, peak_db, confirmed in stats:
        dur_s = (e_idx - s_idx) / float(VAD_SAMPLE_RATE)
        is_noise_blip = (
            not confirmed
            and dur_s < 1.80
            and overall_db < (med_db - 11.5)
            and peak_db < (med_db - 6.0)
        )
        if not is_noise_blip:
            kept.append((s_idx, e_idx))
    return kept


def _resolve_profile_params(
    profile: AudioProfile,
    config: VADConfig,
) -> tuple[int, int, int, int, int, bool, bool]:
    """Derive adaptive min_silence, gap, crossfade, padding, overlap, and crowd flags."""
    is_studio = profile.kind == "studio"
    is_high_ambient = (not is_studio) and (
        profile.noise_floor_p5_db > -30.0 or profile.snr_db < 18.0
    )

    if is_studio:
        auto_min_sil, auto_gap, auto_xfade = 700, 450, 20
        auto_pad_start, auto_pad_end = 200, 300
    else:
        auto_min_sil = 1250 if is_high_ambient else 900
        auto_gap, auto_xfade = 0, 400
        auto_pad_start, auto_pad_end = 650, 1100

    eff_min_sil = config.min_silence_ms if config.min_silence_ms is not None else auto_min_sil
    eff_gap = config.gap_ms if config.gap_ms is not None else auto_gap
    eff_xfade = config.crossfade_ms if config.crossfade_ms is not None else auto_xfade
    eff_pad_start = config.pad_start_ms if config.pad_start_ms is not None else auto_pad_start
    eff_pad_end = config.pad_end_ms if config.pad_end_ms is not None else auto_pad_end
    direct_overlap = (not is_studio) and (eff_gap == 0)
    use_room_tone = (not is_studio) and (eff_gap > 0) and (profile.noise_floor_p5_db > -50.0)
    return (
        eff_min_sil,
        eff_gap,
        eff_xfade,
        eff_pad_start,
        eff_pad_end,
        direct_overlap,
        use_room_tone,
    )


def _filter_crowd_murmur_clusters(
    segs_info: list[tuple[float, float, float, np.ndarray]],
    snr_db: float,
    clip_start_s: float,
    clip_end_s: float,
) -> tuple[list[tuple[float, float, float, np.ndarray]], int]:
    """Strip short isolated crowd bursts/clusters (<=3.5s) >=10 dB below reciter core."""
    if len(segs_info) < 3:
        return segs_info, 0

    core_vals = np.array([x[2] for x in segs_info], dtype=np.float64)
    durs = np.array([max(0.05, x[1] - x[0]) for x in segs_info], dtype=np.float64)
    reciter_core_med = _weighted_percentile(core_vals, durs, 50.0)
    margin_db = max(10.0, min(14.0, snr_db * 0.40))
    cutoff_db = reciter_core_med - margin_db

    n = len(segs_info)
    strip_indices: set[int] = set()
    idx = 0
    while idx < n:
        if segs_info[idx][2] > cutoff_db:
            idx += 1
            continue

        j = idx
        while (
            j + 1 < n
            and segs_info[j + 1][2] <= cutoff_db
            and (segs_info[j + 1][0] - segs_info[j][1]) < 1.8
        ):
            j += 1

        cluster_span = segs_info[j][1] - segs_info[idx][0]
        gap_before = (
            (segs_info[idx][0] - clip_start_s)
            if idx == 0
            else (segs_info[idx][0] - segs_info[idx - 1][1])
        )
        gap_after = (
            (clip_end_s - segs_info[j][1])
            if j == n - 1
            else (segs_info[j + 1][0] - segs_info[j][1])
        )

        if cluster_span <= 3.5 and gap_before >= 1.5 and gap_after >= 1.8:
            combined_fdbs = np.concatenate([segs_info[k][3] for k in range(idx, j + 1)])
            cluster_core_db = float(np.percentile(combined_fdbs, 75))
            if cluster_core_db <= cutoff_db:
                strip_indices.update(range(idx, j + 1))

        idx = j + 1

    if len(strip_indices) >= n:
        return segs_info, 0

    kept = [item for k, item in enumerate(segs_info) if k not in strip_indices]
    return kept, len(strip_indices)


def _compute_segment_pad_ends_s(
    analysis_audio: np.ndarray,
    segments: list[Segment],
    profile: AudioProfile,
    config: VADConfig,
    base_pad_end_ms: int,
    clip_start_s: float,
) -> list[float]:
    """Extend pad_end to 1300ms on loud vintage/live endings (core_db >= reciter p75)."""
    base_s = base_pad_end_ms / 1000.0
    if config.pad_end_ms is not None or profile.kind == "studio" or not segments:
        return [base_s] * len(segments)

    tail_window_samples = int(round(0.80 * VAD_SAMPLE_RATE))
    seg_cores: list[float] = []
    seg_durs: list[float] = []
    tail_cores: list[float] = []

    for seg in segments:
        s_idx = max(0, min(analysis_audio.size, int(round((seg.start - clip_start_s) * VAD_SAMPLE_RATE))))
        e_idx = max(s_idx, min(analysis_audio.size, int(round((seg.end - clip_start_s) * VAD_SAMPLE_RATE))))
        seg_fdbs, _ = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        seg_cores.append(float(np.percentile(seg_fdbs, 75)))
        seg_durs.append(max(0.05, seg.end - seg.start))

        tail_s = max(s_idx, e_idx - tail_window_samples)
        tail_fdbs, _ = _segment_frame_dbs(analysis_audio, tail_s, e_idx)
        tail_cores.append(float(np.percentile(tail_fdbs, 75)))

    reciter_p75 = _weighted_percentile(
        np.array(seg_cores, dtype=np.float64),
        np.array(seg_durs, dtype=np.float64),
        75.0,
    )
    loud_s = max(base_s, 1.300)
    return [loud_s if tc >= reciter_p75 else base_s for tc in tail_cores]


def _find_quietest_gap_window(
    analysis_audio: np.ndarray,
    kept_segments: list[Segment],
    clip_start_s: float,
    clip_end_s: float,
    target_duration_s: float = 0.6,
    profile: Optional[AudioProfile] = None,
    frame_dbs: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float, float]]:
    """Sample a clean ambient loop window from the recording's quietest floor frames (p5-p15)."""
    clip_dur = clip_end_s - clip_start_s
    if not kept_segments or clip_dur <= 0.3:
        return None

    gaps: list[tuple[float, float]] = []
    cursor = clip_start_s
    for seg in kept_segments:
        if seg.start - cursor >= 0.25:
            gaps.append((cursor, seg.start))
        cursor = max(cursor, seg.end)
    if clip_end_s - cursor >= 0.25:
        gaps.append((cursor, clip_end_s))

    floor_lo: Optional[float] = None
    floor_hi: Optional[float] = None
    if profile is not None:
        if frame_dbs is None:
            frame_dbs = _compute_frame_dbs(analysis_audio)
        non_zero = frame_dbs[frame_dbs > -85.0]
        p15 = float(np.percentile(non_zero, 15)) if non_zero.size > 0 else profile.noise_floor_p10_db
        floor_lo = profile.noise_floor_p5_db - 3.0
        floor_hi = max(profile.noise_floor_p10_db + 3.0, p15 + 2.0)

    win_samples = max(int(0.25 * VAD_SAMPLE_RATE), int(round(target_duration_s * VAD_SAMPLE_RATE)))
    best_any: Optional[tuple[float, float, float]] = None
    best_floor: Optional[tuple[float, float, float]] = None

    def _consider(cand: tuple[float, float, float]) -> None:
        nonlocal best_any, best_floor
        if best_any is None or cand[2] < best_any[2]:
            best_any = cand
        if floor_lo is not None and floor_hi is not None and floor_lo <= cand[2] <= floor_hi:
            if best_floor is None or cand[2] < best_floor[2]:
                best_floor = cand

    for g_start, g_end in gaps:
        inner_s = g_start + 0.10 if (g_end - g_start) > 0.5 else g_start
        inner_e = g_end - 0.10 if (g_end - g_start) > 0.5 else g_end
        s_idx = max(0, int(round((inner_s - clip_start_s) * VAD_SAMPLE_RATE)))
        e_idx = min(analysis_audio.size, int(round((inner_e - clip_start_s) * VAD_SAMPLE_RATE)))
        if e_idx - s_idx < int(0.2 * VAD_SAMPLE_RATE):
            continue

        span = analysis_audio[s_idx:e_idx]
        if span.size <= win_samples:
            rms_db = float(20.0 * np.log10(np.sqrt(np.mean(span * span) + 1e-12)))
            _consider(
                (
                    clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                    clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
                    rms_db,
                )
            )
            continue

        step = max(1600, win_samples // 4)
        for off in range(0, span.size - win_samples + 1, step):
            w = span[off : off + win_samples]
            rms_db = float(20.0 * np.log10(np.sqrt(np.mean(w * w) + 1e-12)))
            w_start = clip_start_s + (s_idx + off) / float(VAD_SAMPLE_RATE)
            w_end = clip_start_s + (s_idx + off + win_samples) / float(VAD_SAMPLE_RATE)
            _consider((w_start, w_end, rms_db))

    # Fallback: if no skipped gap lay in the p5-p15 floor band, scan contiguous frame_dbs.
    if best_floor is None and floor_lo is not None and floor_hi is not None:
        if frame_dbs is None:
            frame_dbs = _compute_frame_dbs(analysis_audio)
        win_frames = max(8, int(round(win_samples / float(_FRAME_SAMPLES))))
        if frame_dbs.size >= win_frames:
            step_f = max(4, win_frames // 2)
            for f0 in range(0, frame_dbs.size - win_frames + 1, step_f):
                slice_dbs = frame_dbs[f0 : f0 + win_frames]
                mean_db = float(np.mean(slice_dbs))
                if floor_lo <= mean_db <= floor_hi and float(np.max(slice_dbs)) <= floor_hi + 3.0:
                    w_start = clip_start_s + (f0 * _FRAME_SAMPLES) / float(VAD_SAMPLE_RATE)
                    w_end = clip_start_s + ((f0 + win_frames) * _FRAME_SAMPLES) / float(VAD_SAMPLE_RATE)
                    if best_floor is None or mean_db < best_floor[2]:
                        best_floor = (w_start, w_end, mean_db)

    return best_floor or best_any


def detect_speech_segments(
    audio_16k: np.ndarray,
    config: Optional[VADConfig] = None,
    exact_duration: Optional[float] = None,
    clip_start_s: float = 0.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> VADResult:
    """Detect vocal segments via auto-profiled hybrid Silero VAD + energy gate."""
    cfg = config if config is not None else VADConfig()
    vad_duration = audio_16k.size / float(VAD_SAMPLE_RATE)
    clip_dur = exact_duration if exact_duration is not None else vad_duration
    clip_end_s = clip_start_s + clip_dur

    analysis_audio = audio_16k.astype(np.float64, copy=False) - float(np.mean(audio_16k))
    frame_dbs = _compute_frame_dbs(analysis_audio)
    profile = _profile_audio(analysis_audio, frame_dbs=frame_dbs)

    if profile.snr_db < _LOW_SNR_GUARD_DB:
        if progress_cb is not None:
            progress_cb(1.0)
        seg_fdbs, overall_db = _segment_frame_dbs(analysis_audio, 0, analysis_audio.size)
        core_db = float(np.percentile(seg_fdbs, 75))
        full_seg = Segment(
            start=round(clip_start_s, 4),
            end=round(clip_end_s, 4),
            rms_db=round(overall_db, 2),
            core_db=round(core_db, 2),
        )
        return VADResult(
            segments=[full_seg],
            total_duration=clip_dur,
            profile=profile,
            raw_segments_count=1,
            filtered_segments_count=1,
            resolved_pad_start_ms=0,
            resolved_pad_end_ms=0,
            resolved_min_silence_ms=0,
            resolved_crossfade_ms=0,
            resolved_gap_ms=0,
            use_room_tone=False,
            direct_overlap=False,
            low_snr_guarded=True,
            stray_shouts_stripped=0,
        )

    (
        eff_min_sil,
        eff_gap,
        eff_xfade,
        eff_pad_start,
        eff_pad_end,
        direct_overlap,
        use_room_tone,
    ) = _resolve_profile_params(profile, cfg)
    island_bridge_ms = eff_min_sil
    min_speech_ms = _STUDIO_MIN_SPEECH_MS if profile.kind == "studio" else _MIN_SPEECH_MS

    silero_intervals = _run_silero_vad_chunked(
        audio_16k=audio_16k,
        threshold=cfg.threshold,
        island_bridge_ms=island_bridge_ms,
        min_speech_ms=min_speech_ms,
        progress_cb=(lambda p: progress_cb(p * 0.90)) if progress_cb is not None else None,
    )

    # Dense studio recitations with short inter-verse breaths (<700ms) push frame p5
    # into vocal tails; inspect low-tail p0.2 when Silero covers >=90% of the audio.
    if profile.kind != "studio" and audio_16k.size > 0:
        silero_cov = sum(e - s for s, e in silero_intervals) / float(audio_16k.size)
        non_zero = frame_dbs[frame_dbs > -85.0]
        if silero_cov >= 0.90 and non_zero.size > 0 and float(np.percentile(non_zero, 0.2)) <= -50.0:
            profile = AudioProfile(
                kind="studio",
                noise_floor_p5_db=profile.noise_floor_p5_db,
                noise_floor_p10_db=profile.noise_floor_p10_db,
                p90_db=profile.p90_db,
                snr_db=profile.snr_db,
            )
            (
                eff_min_sil,
                eff_gap,
                eff_xfade,
                eff_pad_start,
                eff_pad_end,
                direct_overlap,
                use_room_tone,
            ) = _resolve_profile_params(profile, cfg)
            min_speech_ms = _STUDIO_MIN_SPEECH_MS
            island_bridge_ms = eff_min_sil
            if cfg.min_silence_ms is None:
                silero_intervals = _run_silero_vad_chunked(
                    audio_16k=audio_16k,
                    threshold=cfg.threshold,
                    island_bridge_ms=island_bridge_ms,
                    min_speech_ms=min_speech_ms,
                )

    energy_intervals = _detect_energy_islands(
        analysis_audio=analysis_audio,
        profile=profile,
        island_bridge_ms=island_bridge_ms,
        frame_dbs=frame_dbs,
    )

    bridge_samples = int(round((island_bridge_ms / 1000.0) * VAD_SAMPLE_RATE))
    raw_samples = _merge_intervals(
        [*silero_intervals, *energy_intervals],
        bridge_samples=bridge_samples,
    )

    if profile.kind == "studio":
        raw_samples = _filter_studio_breaths(analysis_audio, raw_samples)
    else:
        raw_samples = _filter_unconfirmed_noise_blips(
            analysis_audio, raw_samples, silero_intervals
        )

    if not raw_samples:
        return VADResult(
            segments=[],
            total_duration=clip_dur,
            profile=profile,
            raw_segments_count=0,
            filtered_segments_count=0,
            resolved_pad_start_ms=eff_pad_start,
            resolved_pad_end_ms=eff_pad_end,
            resolved_min_silence_ms=eff_min_sil,
            resolved_crossfade_ms=eff_xfade,
            resolved_gap_ms=eff_gap,
            use_room_tone=use_room_tone,
            direct_overlap=direct_overlap,
        )

    raw_count = len(raw_samples)
    stray_stripped_count = 0

    segs_raw: list[tuple[float, float, float, np.ndarray]] = []
    for s_idx, e_idx in raw_samples:
        seg_fdbs, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        core_db = float(np.percentile(seg_fdbs, 75))
        segs_raw.append(
            (
                clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
                core_db,
                seg_fdbs,
            )
        )

    if profile.kind == "vintage/live":
        segs_raw, stray_stripped_count = _filter_crowd_murmur_clusters(
            segs_raw,
            snr_db=profile.snr_db,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
        )

    bridged = [Segment(start=s, end=e) for s, e, _, _ in segs_raw]
    filtered_count = len(bridged)
    if progress_cb is not None:
        progress_cb(1.0)

    pad_ends_s = _compute_segment_pad_ends_s(
        analysis_audio=analysis_audio,
        segments=bridged,
        profile=profile,
        config=cfg,
        base_pad_end_ms=eff_pad_end,
        clip_start_s=clip_start_s,
    )
    min_effective_cut_s = 0.0 if profile.kind == "studio" else _MIN_VINTAGE_EFFECTIVE_CUT_S
    padded = _apply_padding_and_merge(
        bridged,
        pad_start_s=eff_pad_start / 1000.0,
        pad_end_s=pad_ends_s,
        min_bound=clip_start_s,
        max_bound=clip_end_s,
        min_splice_gap_s=eff_xfade / 1000.0,
        min_effective_cut_s=min_effective_cut_s,
    )

    final_segments: list[Segment] = []
    for seg in padded:
        s_idx = max(0, min(audio_16k.size, int(round((seg.start - clip_start_s) * VAD_SAMPLE_RATE))))
        e_idx = max(s_idx, min(audio_16k.size, int(round((seg.end - clip_start_s) * VAD_SAMPLE_RATE))))
        seg_fdbs, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        core_db = float(np.percentile(seg_fdbs, 75))
        final_segments.append(
            Segment(
                start=round(seg.start, 4),
                end=round(seg.end, 4),
                rms_db=round(overall_db, 2),
                core_db=round(core_db, 2),
            )
        )

    quiet_gap = (
        _find_quietest_gap_window(
            analysis_audio=analysis_audio,
            kept_segments=final_segments,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            target_duration_s=max(0.5, (eff_gap + eff_xfade) / 1000.0),
            profile=profile,
            frame_dbs=frame_dbs,
        )
        if use_room_tone
        else None
    )

    return VADResult(
        segments=final_segments,
        total_duration=clip_dur,
        profile=profile,
        raw_segments_count=raw_count,
        filtered_segments_count=filtered_count,
        quietest_gap_window=quiet_gap,
        resolved_pad_start_ms=eff_pad_start,
        resolved_pad_end_ms=eff_pad_end,
        resolved_min_silence_ms=eff_min_sil,
        resolved_crossfade_ms=eff_xfade,
        resolved_gap_ms=eff_gap,
        use_room_tone=use_room_tone,
        direct_overlap=direct_overlap,
        low_snr_guarded=False,
        stray_shouts_stripped=stray_stripped_count,
    )
