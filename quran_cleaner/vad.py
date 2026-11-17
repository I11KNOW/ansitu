"""Hybrid Silero VAD + auto-profiler, studio breath filter, prayer mode, and strict calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .media import VAD_SAMPLE_RATE

Mode = Literal["natural", "prayer", "strict"]
ProfileType = Literal["studio", "vintage/live"]

_FRAME_SAMPLES = 512  # 32ms windows at 16kHz (matches Silero VAD windowing)
_MIN_SPEECH_MS = 200
_STUDIO_MIN_SPEECH_MS = 220
_ENERGY_BRIDGE_MS = 250
_VAD_CHUNK_SECONDS = 60
_VAD_OVERLAP_SECONDS = 2


@dataclass(frozen=True)
class VADConfig:
    """Configuration for hybrid Silero VAD, auto-profiling, prayer mode, and gaps."""

    mode: Mode = "natural"
    threshold: float = 0.20
    pad_start_ms: int = 200
    pad_end_ms: int = 300
    min_silence_ms: int = 700
    crossfade_ms: int = 20
    gap_ms: int = 450
    strict_margin_db: float = 12.0


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
    speech_p25_db: Optional[float]
    reciter_median_db: Optional[float]
    strict_cutoff_db: Optional[float]
    raw_segments_count: int
    filtered_segments_count: int
    prayer_windows_count: int = 0
    prayer_stripped_count: int = 0
    quietest_gap_window: Optional[tuple[float, float, float]] = None


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


def _bridge_close_segments(segments: list[Segment], min_gap_s: float) -> list[Segment]:
    """Merge adjacent segments whose silence gap is strictly shorter than min_gap_s."""
    if not segments:
        return []
    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start - prev.end < min_gap_s:
            merged[-1] = Segment(start=prev.start, end=max(prev.end, seg.end))
        else:
            merged.append(seg)
    return merged


def _apply_padding_and_merge(
    segments: list[Segment],
    pad_start_s: float,
    pad_end_s: float,
    min_bound: float,
    max_bound: float,
    min_splice_gap_s: float,
) -> list[Segment]:
    """Apply asymmetric start/end padding and merge overlapping or sub-crossfade gaps."""
    if not segments:
        return []

    padded = [
        Segment(
            start=max(min_bound, seg.start - pad_start_s),
            end=min(max_bound, seg.end + pad_end_s),
        )
        for seg in segments
    ]

    merged: list[Segment] = [padded[0]]
    for seg in padded[1:]:
        prev = merged[-1]
        if seg.start - prev.end <= min_splice_gap_s:
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
        # safety net above breaths so Maqam holds are never clipped while breaths stay rejected.
        if spread < 15.0:
            return []
        onset_db = max(profile.noise_floor_p5_db + 22.0, profile.p90_db - 20.0)
        sustain_db = onset_db - 4.0
        min_island_dur = 0.40
    else:
        if spread < 6.0:
            return []
        if spread >= 22.0:
            onset_db = max(profile.noise_floor_p10_db + 15.0, profile.p90_db - 25.0)
            sustain_db = onset_db - 4.0
        else:
            # Moderate/low-SNR hall or crowd ambiance: crisp threshold avoids bridging mosque echoes.
            onset_db = max(profile.noise_floor_p10_db + 0.52 * spread, profile.p90_db - 25.0)
            sustain_db = onset_db - min(2.0, 0.15 * spread)
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


def _find_prayer_takbir_indices(
    segs_s: list[tuple[float, float, float]],
    sil_250_s: list[tuple[float, float]],
    total_dur: float,
    strip_opening: bool = False,
    clip_start_s: float = 0.0,
) -> tuple[int, set[int]]:
    """Detect periodic Ruku/Sujud transitions (and optional opening Takbir) for prayer mode."""
    if len(segs_s) < 4:
        return 0, set()

    med_db = float(np.median([x[2] for x in segs_s]))
    is_recit = [False] * len(segs_s)

    # 1. Mark sustained Al-Fatihah halves, short Surahs, and Rak'ah recitation (> 12.5s).
    for i, (s, e, _) in enumerate(segs_s):
        if (e - s) > 12.5:
            is_recit[i] = True

    # 2. Protect post-Sujud Al-Fatihah clusters right after a long >= 12.0s silence
    #    (e.g. a ~17-19s Al-Fatihah split by one or two brief < 2.0s pauses).
    i = 0
    while i < len(segs_s):
        if is_recit[i]:
            i += 1
            continue
        gap_prev = (segs_s[i][0] - clip_start_s) if i == 0 else (segs_s[i][0] - segs_s[i - 1][1])
        if gap_prev >= 12.0:
            j = i
            while j + 1 < len(segs_s) and not is_recit[j + 1] and (segs_s[j + 1][0] - segs_s[j][1]) < 2.0:
                j += 1
            span = segs_s[j][1] - segs_s[i][0]
            speech = sum(segs_s[k][1] - segs_s[k][0] for k in range(i, j + 1))
            max_seg = max(segs_s[k][1] - segs_s[k][0] for k in range(i, j + 1))
            if speech >= 11.0 and (speech / max(span, 1e-6)) >= 0.70 and max_seg >= 9.0:
                for k in range(i, j + 1):
                    is_recit[k] = True
            i = j + 1
        else:
            i += 1

    rakah_blocks: list[tuple[int, int, float]] = []
    k = 0
    while k < len(segs_s):
        if not is_recit[k]:
            k += 1
            continue
        k2 = k
        while k2 + 1 < len(segs_s) and is_recit[k2 + 1]:
            k2 += 1
        dur_block = sum(segs_s[m][1] - segs_s[m][0] for m in range(k, k2 + 1))
        rakah_blocks.append((k, k2, dur_block))
        k = k2 + 1

    total_speech = sum(e - s for s, e, _ in segs_s)
    # Prayers are dominated by sustained unbroken Rak'ah segments (>=22s),
    # whereas live concerts alternate short phrases with crowd applause.
    raw_long_speech = sum(e - s for s, e, _ in segs_s if (e - s) >= 22.0)
    long_speech = sum(b[2] for b in rakah_blocks if b[2] >= 25.0)
    if (
        len([b for b in rakah_blocks if b[2] >= 22.0]) < 2
        or (long_speech / max(total_speech, 1e-6)) < 0.55
        or (raw_long_speech / max(total_speech, 1e-6)) < 0.65
    ):
        return 0, set()

    spans_to_check: list[tuple[int, int, float, float, float, float, bool]] = []
    for idx in range(len(rakah_blocks) - 1):
        end_prev = rakah_blocks[idx][1]
        start_next = rakah_blocks[idx + 1][0]
        if start_next - end_prev > 1:
            is_last_partial = (
                idx + 1 == len(rakah_blocks) - 1
                and total_dur - segs_s[rakah_blocks[-1][1]][1] < 20.0
            )
            spans_to_check.append(
                (
                    end_prev + 1,
                    start_next - 1,
                    segs_s[end_prev][1],
                    segs_s[start_next][0],
                    rakah_blocks[idx][2],
                    rakah_blocks[idx + 1][2],
                    is_last_partial,
                )
            )
    last_end = rakah_blocks[-1][1]
    if last_end < len(segs_s) - 1:
        spans_to_check.append(
            (
                last_end + 1,
                len(segs_s) - 1,
                segs_s[last_end][1],
                total_dur,
                rakah_blocks[-1][2],
                0.0,
                True,
            )
        )

    window_count = 0
    strip_indices: set[int] = set()
    for i0, i1, w_start, w_end, prev_dur, next_dur, is_last_partial in spans_to_check:
        w_dur = w_end - w_start
        if w_dur < 12.0:
            continue
        cands = list(range(i0, i1 + 1))
        speech_sum = sum(segs_s[m][1] - segs_s[m][0] for m in cands)
        silence_ratio = (w_dur - speech_sum) / w_dur
        inner_gaps = []
        cur = w_start
        for m in cands:
            inner_gaps.append(segs_s[m][0] - cur)
            cur = segs_s[m][1]
        inner_gaps.append(w_end - cur)
        max_gap = max(inner_gaps)
        loud_utts = sum(1 for m in cands if segs_s[m][2] >= med_db - 6.0)
        max_utt_dur = max(segs_s[m][1] - segs_s[m][0] for m in cands)

        # Rule A: Muballigh Ruku/Sujud cluster between two long Rak'ah blocks (>= 25s,
        # or closing partial Rak'ah > 12.5s at end of file): 14s-65s span of >= 3 utterances all <= 12.0s.
        is_muballigh_cluster = (
            prev_dur >= 25.0
            and (next_dur >= 25.0 or (is_last_partial and next_dur > 12.5))
            and 14.0 <= w_dur <= 65.0
            and len(cands) >= 3
            and max_utt_dur <= 12.0
            and loud_utts >= 3
        )
        if is_muballigh_cluster:
            window_count += 1
            strip_indices.update(cands)
            continue

        # Rule B: Classic high-silence Ruku/Sujud window (>= 35% silence, max gap >= 3.0s).
        if silence_ratio >= 0.35 and max_gap >= 3.0 and loud_utts >= 2:
            takbirs_in_win: list[int] = []
            has_long_verse = False
            for m in cands:
                ms, me, _ = segs_s[m]
                mdur = me - ms
                inner = [min(me, se) - max(ms, ss) for ss, se in sil_250_s if se > ms and ss < me]
                max_s = max(inner) if inner else 0.0
                if mdur <= 6.2 and max_s <= 5.5:
                    takbirs_in_win.append(m)
                elif mdur < 12.0 and len(inner) >= 2 and max_s <= 5.5:
                    takbirs_in_win.append(m)
                elif mdur >= 7.5:
                    has_long_verse = True
            if not has_long_verse and len(takbirs_in_win) >= 2 and len(takbirs_in_win) * 2 >= len(cands):
                window_count += 1
                strip_indices.update(takbirs_in_win)

    # A genuine multi-minute prayer contains periodic (>= 2) Ruku/Sujud transitions;
    # avoid single coincidental matches in long live concerts.
    if (total_dur - clip_start_s) >= 300.0 and window_count < 2:
        return 0, set()

    # Opening Takbir (prayer mode only): isolated leading utterance < 5.0s near start
    # (< 3.0s) followed by a pause >= 2.5s before Al-Fatihah.
    if strip_opening and window_count >= 1 and len(segs_s) >= 2:
        s0, e0, _ = segs_s[0]
        gap0 = segs_s[1][0] - e0
        if (s0 - clip_start_s) < 3.0 and (e0 - s0) < 5.0 and gap0 >= 2.5:
            strip_indices.add(0)

    return window_count, strip_indices


def _find_quietest_gap_window(
    analysis_audio: np.ndarray,
    kept_segments: list[Segment],
    clip_start_s: float,
    clip_end_s: float,
    target_duration_s: float = 0.6,
) -> Optional[tuple[float, float, float]]:
    """Locate the cleanest, lowest-RMS skipped silence window for authentic room tone."""
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

    if not gaps:
        return None

    win_samples = max(int(0.25 * VAD_SAMPLE_RATE), int(round(target_duration_s * VAD_SAMPLE_RATE)))
    best: Optional[tuple[float, float, float]] = None

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
            cand = (
                clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
                rms_db,
            )
            if best is None or cand[2] < best[2]:
                best = cand
            continue

        step = max(1600, win_samples // 4)
        for off in range(0, span.size - win_samples + 1, step):
            w = span[off : off + win_samples]
            rms_db = float(20.0 * np.log10(np.sqrt(np.mean(w * w) + 1e-12)))
            if best is None or rms_db < best[2]:
                w_start = clip_start_s + (s_idx + off) / float(VAD_SAMPLE_RATE)
                w_end = clip_start_s + (s_idx + off + win_samples) / float(VAD_SAMPLE_RATE)
                best = (w_start, w_end, rms_db)

    return best


def detect_speech_segments(
    audio_16k: np.ndarray,
    config: VADConfig,
    exact_duration: Optional[float] = None,
    clip_start_s: float = 0.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> VADResult:
    """Detect vocal segments via auto-profiled hybrid Silero VAD + energy gate."""
    vad_duration = audio_16k.size / float(VAD_SAMPLE_RATE)
    clip_dur = exact_duration if exact_duration is not None else vad_duration
    clip_end_s = clip_start_s + clip_dur

    analysis_audio = audio_16k.astype(np.float64, copy=False) - float(np.mean(audio_16k))
    frame_dbs = _compute_frame_dbs(analysis_audio)
    profile = _profile_audio(analysis_audio, frame_dbs=frame_dbs)

    island_bridge_ms = (
        min(_ENERGY_BRIDGE_MS, config.min_silence_ms)
        if config.mode == "strict"
        else config.min_silence_ms
    )
    min_speech_ms = _STUDIO_MIN_SPEECH_MS if profile.kind == "studio" else _MIN_SPEECH_MS

    sil_scale = 0.90 if config.mode == "strict" else 0.50
    silero_intervals = _run_silero_vad_chunked(
        audio_16k=audio_16k,
        threshold=config.threshold,
        island_bridge_ms=island_bridge_ms,
        min_speech_ms=min_speech_ms,
        progress_cb=(lambda p: progress_cb(p * sil_scale)) if progress_cb is not None else None,
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
    raw_samples = _filter_unconfirmed_noise_blips(
        analysis_audio, raw_samples, silero_intervals
    )

    if not raw_samples:
        return VADResult(
            segments=[],
            total_duration=clip_dur,
            profile=profile,
            speech_p25_db=None,
            reciter_median_db=None,
            strict_cutoff_db=None,
            raw_segments_count=0,
            filtered_segments_count=0,
        )

    speech_p25_db: Optional[float] = None
    reciter_median_db: Optional[float] = None
    strict_cutoff_db: Optional[float] = None
    raw_count = len(raw_samples)
    prayer_windows_count = 0
    prayer_stripped_count = 0

    if config.mode == "strict":
        island_stats: list[tuple[int, int, float, float]] = []
        for s_idx, e_idx in raw_samples:
            frame_dbs, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
            island_energy_db = float(max(overall_db, float(np.median(frame_dbs))))
            island_stats.append((s_idx, e_idx, overall_db, island_energy_db))

        energies = np.array([x[3] for x in island_stats], dtype=np.float64)
        durations = np.array(
            [max(1, x[1] - x[0]) / float(VAD_SAMPLE_RATE) for x in island_stats],
            dtype=np.float64,
        )
        speech_p25_db = _weighted_percentile(energies, durations, 25.0)
        reciter_median_db = _weighted_percentile(energies, durations, 50.0)
        strict_cutoff_db = speech_p25_db - config.strict_margin_db

        kept_islands = [
            Segment(
                start=clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                end=clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
                rms_db=overall_db,
                core_db=island_energy_db,
            )
            for s_idx, e_idx, overall_db, island_energy_db in island_stats
            if island_energy_db >= strict_cutoff_db
        ]
        filtered_count = len(kept_islands)
        bridged = _bridge_close_segments(kept_islands, config.min_silence_ms / 1000.0)
    else:
        segs_with_rms: list[tuple[float, float, float]] = []
        for s_idx, e_idx in raw_samples:
            _, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
            segs_with_rms.append(
                (
                    clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                    clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
                    overall_db,
                )
            )

        # Compute unbridged 250ms Silero bursts for Prayer Ruku/Sujud & Auto-Detection Hint.
        sil_250_raw = _run_silero_vad_chunked(
            audio_16k=audio_16k,
            threshold=config.threshold,
            island_bridge_ms=_ENERGY_BRIDGE_MS,
            min_speech_ms=min_speech_ms,
            progress_cb=(
                (lambda p: progress_cb(0.50 + 0.45 * p))
                if progress_cb is not None
                else None
            ),
        )
        sil_250_s = [
            (
                clip_start_s + s_idx / float(VAD_SAMPLE_RATE),
                clip_start_s + e_idx / float(VAD_SAMPLE_RATE),
            )
            for s_idx, e_idx in sil_250_raw
        ]
        prayer_windows_count, strip_indices = _find_prayer_takbir_indices(
            segs_s=segs_with_rms,
            sil_250_s=sil_250_s,
            total_dur=clip_end_s,
            strip_opening=(config.mode == "prayer"),
            clip_start_s=clip_start_s,
        )

        if config.mode == "prayer" and strip_indices:
            prayer_stripped_count = len(strip_indices)
            bridged = [
                Segment(start=s, end=e)
                for idx, (s, e, _) in enumerate(segs_with_rms)
                if idx not in strip_indices
            ]
        else:
            bridged = [Segment(start=s, end=e) for s, e, _ in segs_with_rms]
        filtered_count = len(bridged)

    padded = _apply_padding_and_merge(
        bridged,
        pad_start_s=config.pad_start_ms / 1000.0,
        pad_end_s=config.pad_end_ms / 1000.0,
        min_bound=clip_start_s,
        max_bound=clip_end_s,
        min_splice_gap_s=config.crossfade_ms / 1000.0,
    )

    final_segments: list[Segment] = []
    for seg in padded:
        s_idx = max(0, min(audio_16k.size, int(round((seg.start - clip_start_s) * VAD_SAMPLE_RATE))))
        e_idx = max(s_idx, min(audio_16k.size, int(round((seg.end - clip_start_s) * VAD_SAMPLE_RATE))))
        frame_dbs, overall_db = _segment_frame_dbs(analysis_audio, s_idx, e_idx)
        core_db = float(np.percentile(frame_dbs, 75))
        final_segments.append(
            Segment(
                start=round(seg.start, 4),
                end=round(seg.end, 4),
                rms_db=round(overall_db, 2),
                core_db=round(core_db, 2),
            )
        )

    quiet_gap = _find_quietest_gap_window(
        analysis_audio=analysis_audio,
        kept_segments=final_segments,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        target_duration_s=max(0.5, (config.gap_ms + config.crossfade_ms) / 1000.0),
    )

    return VADResult(
        segments=final_segments,
        total_duration=clip_dur,
        profile=profile,
        speech_p25_db=round(speech_p25_db, 2) if speech_p25_db is not None else None,
        reciter_median_db=(
            round(reciter_median_db, 2) if reciter_median_db is not None else None
        ),
        strict_cutoff_db=(
            round(strict_cutoff_db, 2) if strict_cutoff_db is not None else None
        ),
        raw_segments_count=raw_count,
        filtered_segments_count=filtered_count,
        prayer_windows_count=prayer_windows_count,
        prayer_stripped_count=prayer_stripped_count,
        quietest_gap_window=quiet_gap,
    )
