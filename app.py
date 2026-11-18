"""Bilingual (Arabic-first / English) Streamlit web app for أنصتوا — Ansitu."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import streamlit as st

from quran_cleaner.media import (
    download_external_audio,
    extract_vad_audio,
    is_external_url,
    parse_timestamp,
    probe_media,
    require_ffmpeg,
)
from quran_cleaner.processor import render_clean_ranges
from quran_cleaner.vad import VADConfig, VADResult, detect_speech_segments

# Backward-compatible export alias used by tests or callers referencing legacy name.
download_youtube_audio = download_external_audio

_MODE_LABELS = {
    "natural": "تلاوة طبيعية | Natural Recitation",
    "prayer": "صلاة جهرية | Prayer Mode",
}

_FORMAT_LABELS = {
    "m4a": "M4A — جودة عالية (AAC High Quality)",
    "mp3": "MP3 — متوافق مع كل الأجهزة (Universal)",
}

_QUALITY_LABELS = {
    "128": "128 kbps (جودة يوتيوب - حجم خفيف)",
    "192": "192 kbps (جودة عالية - الافتراضي)",
    "256": "256 kbps (أقصى جودة أرشيفية)",
    "lossless_wav": "Lossless WAV (جودة أصلية بدون فقدان - PCM)",
}

_MIME_TYPES = {
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
}

_LOSSLESS_EXTENSIONS = {".wav", ".flac"}
_UPLOAD_TYPES = ["mp3", "wav", "m4a", "mp4", "mkv", "webm", "opus", "flac"]


def is_lossless_source(source_name: Optional[str]) -> bool:
    """Return True when the input filename uses a lossless PCM/FLAC container."""
    if not source_name:
        return False
    return Path(source_name).suffix.lower() in _LOSSLESS_EXTENSIONS


def quality_options_for_source(source_name: Optional[str]) -> list[str]:
    """Return selectable quality presets, adding Lossless WAV for lossless sources."""
    opts = ["128", "192", "256"]
    if is_lossless_source(source_name):
        opts.append("lossless_wav")
    return opts


def format_duration_clock(seconds: float) -> str:
    """Format seconds into compact MM:SS or HH:MM:SS for UI metrics."""
    total_s = max(0, int(round(seconds)))
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_profile_label(profile_kind: str) -> str:
    """Return bilingual Studio vs Vintage/Live label."""
    if profile_kind == "studio":
        return "استوديو | Studio"
    return "تراثي / مباشر | Vintage/Live"


def parse_optional_range(
    start_text: str, end_text: str
) -> tuple[Optional[float], Optional[float]]:
    """Parse optional MM:SS / HH:MM:SS inputs and validate start < end."""
    start_clean = start_text.strip()
    end_clean = end_text.strip()
    start_s = parse_timestamp(start_clean) if start_clean else None
    end_s = parse_timestamp(end_clean) if end_clean else None
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError(
            "وقت البداية يجب أن يكون قبل وقت النهاية | "
            "Start timestamp must be earlier than end timestamp."
        )
    return start_s, end_s


def format_external_url_error(exc: Exception) -> str:
    """Return a graceful bilingual message when an external host blocks datacenter IPs or fails."""
    detail = str(exc).strip()
    lower = detail.lower()
    blocked_signals = (
        "403",
        "sign in to confirm",
        "not a bot",
        "captcha",
        "too many requests",
        "429",
        "unavailable",
        "blocked",
        "forbidden",
    )
    if any(tok in lower for tok in blocked_signals):
        return (
            "تعذّر التحميل من الرابط الخارجي لأن خوادم السحابة (Datacenter IP) محجوبة مؤقتاً من قِبل المصدر (مثل يوتيوب).\n\n"
            "الحل البديل الموصى به: قم بتنزيل الملف الصوتي أو الفيديو على جهازك أولاً، ثم استخدم تبويب "
            "«رفع ملف صوتي أو فيديو | Upload Audio/Video File» بالأعلى.\n\n"
            "The external source blocked this server's datacenter IP (bot-check / HTTP 403). "
            "Please download the audio/video locally on your device and use the primary "
            "'Upload Audio/Video File' tab instead."
        )
    return (
        "تعذّر تحميل الصوت من الرابط الخارجي. تحقّق من صحة الرابط أو حمّل الملف محلياً ثم ارفعه عبر تبويب «رفع ملف».\n\n"
        f"External media download failed ({detail}). Try downloading locally and uploading via the File tab."
    )


format_youtube_error = format_external_url_error


def run_cleaning_job(
    *,
    mode: str,
    start_text: str = "",
    end_text: str = "",
    ranges: Optional[Sequence[tuple[str, str]]] = None,
    out_format: str = "m4a",
    quality: str = "192",
    uploaded_name: Optional[str] = None,
    uploaded_bytes: Optional[bytes] = None,
    external_url: Optional[str] = None,
    youtube_url: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Execute end-to-end load, single/multi-range VAD analysis, slicing, and in-memory export."""
    def _report(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0, min(100, pct)), msg)

    src_url = external_url if external_url is not None else youtube_url

    if quality not in _QUALITY_LABELS:
        raise ValueError(f"Unsupported audio quality preset: {quality}")

    fmt = out_format.lower().lstrip(".")
    if quality == "lossless_wav":
        if src_url is not None or not is_lossless_source(uploaded_name):
            raise ValueError(
                "خيار Lossless WAV متاح فقط عند رفع ملف WAV أو FLAC محلياً | "
                "Lossless WAV is only available when uploading a local .wav or .flac file."
            )
        eff_fmt = "wav"
        bitrate_kbps: Optional[int] = None
    else:
        if fmt not in {"m4a", "mp3"}:
            raise ValueError(f"Unsupported output format: {out_format}")
        eff_fmt = fmt
        bitrate_kbps = int(quality)

    raw_ranges = list(ranges) if ranges is not None else [(start_text, end_text)]
    if not raw_ranges:
        raw_ranges = [("", "")]

    parsed_ranges: list[tuple[Optional[float], Optional[float]]] = []
    for idx, (s_txt, e_txt) in enumerate(raw_ranges, start=1):
        if len(raw_ranges) > 1 and not s_txt.strip() and not e_txt.strip():
            raise ValueError(
                f"يرجى إدخال وقت البداية أو النهاية للمقطع رقم {idx} أو حذفه | "
                f"Please enter Start/End timestamps for Range #{idx} or remove the blank row."
            )
        parsed_ranges.append(parse_optional_range(s_txt, e_txt))

    require_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="quran_web_") as tmp_root:
        work_dir = Path(tmp_root)

        if src_url is not None:
            url = src_url.strip()
            if not is_external_url(url):
                raise ValueError(
                    "رابط غير صالح — يرجى إدخال رابط يبدأ بـ http:// أو https:// | "
                    "Please enter a valid http:// or https:// media URL."
                )
            _report(8, "جاري التحميل من الرابط الخارجي... | Downloading audio from URL...")
            try:
                input_path = download_youtube_audio(url, work_dir)
            except Exception as exc:
                raise RuntimeError(format_external_url_error(exc)) from exc
            source_stem = input_path.stem
        elif uploaded_bytes is not None and uploaded_name:
            _report(10, "جاري تجهيز الملف... | Preparing uploaded file...")
            safe_name = Path(uploaded_name).name or "recitation.mp3"
            input_path = work_dir / safe_name
            input_path.write_bytes(uploaded_bytes)
            source_stem = input_path.stem
        else:
            raise ValueError(
                "يرجى اختيار ملف أو إدخال رابط خارجي | Please provide a media file or external URL."
            )

        _report(22, "جاري فحص خصائص الصوت... | Probing audio properties...")
        info = probe_media(input_path)
        config = VADConfig(mode=mode)  # type: ignore[arg-type]

        range_jobs: list[tuple[Optional[float], Optional[float], VADResult]] = []
        n_ranges = len(parsed_ranges)
        for r_idx, (start_s, end_s) in enumerate(parsed_ranges):
            clip_start = start_s if start_s is not None else 0.0
            clip_end = min(end_s, info.duration) if end_s is not None else info.duration
            if clip_start >= info.duration:
                raise ValueError(
                    f"وقت البداية في المقطع #{r_idx + 1} ({clip_start:.1f}s) يتجاوز مدة الملف ({info.duration:.1f}s) | "
                    f"Range #{r_idx + 1} start exceeds file duration."
                )
            if clip_start >= clip_end:
                raise ValueError(
                    f"في المقطع #{r_idx + 1}، يجب أن يكون وقت البداية قبل النهاية | "
                    f"Range #{r_idx + 1} start must be earlier than end."
                )
            clip_duration = clip_end - clip_start

            base_pct = 28 + int(round((r_idx / float(n_ranges)) * 40))
            span_pct = max(5, int(round(40.0 / float(n_ranges))))
            range_lbl = f" ({r_idx + 1}/{n_ranges})" if n_ranges > 1 else ""

            _report(
                base_pct,
                f"جاري استخراج مسار التحليل{range_lbl}... | Extracting 16kHz VAD stream{range_lbl}...",
            )
            audio_16k = extract_vad_audio(
                input_path,
                start_s=start_s,
                end_s=clip_end if end_s is not None else None,
            )

            def _on_vad(frac: float, b_pct: int = base_pct, s_pct: int = span_pct, lbl: str = range_lbl) -> None:
                _report(
                    b_pct + int(round(frac * s_pct)),
                    f"جاري تحليل الصوت والتوقفات{lbl}... | Analyzing speech & pauses{lbl}...",
                )

            vad_result = detect_speech_segments(
                audio_16k=audio_16k,
                config=config,
                exact_duration=clip_duration,
                clip_start_s=clip_start,
                progress_cb=_on_vad,
            )
            range_jobs.append((start_s, clip_end if end_s is not None else None, vad_result))

        out_filename = f"{source_stem}_clean.{eff_fmt}"
        output_path = work_dir / out_filename
        timestamps_path = work_dir / "timestamps.json"

        def _on_render(frac: float) -> None:
            pct = 70 + int(round(frac * 27))
            _report(pct, "جاري تقطيع ودمج الصوت بجودته الأصلية... | Slicing & crossfading native audio...")

        _report(70, "جاري تقطيع ودمج الصوت بجودته الأصلية... | Slicing & crossfading native audio...")
        result = render_clean_ranges(
            input_path=input_path,
            output_path=output_path,
            timestamps_path=timestamps_path,
            info=info,
            config=config,
            range_jobs=range_jobs,
            progress_cb=_on_render,
            bitrate_kbps=bitrate_kbps,
        )

        _report(100, "اكتملت التنقية بنجاح | Completed successfully!")
        ts_text = timestamps_path.read_text(encoding="utf-8")
        ts_data = json.loads(ts_text)
        primary_vad = range_jobs[0][2]

        return {
            "audio_bytes": output_path.read_bytes(),
            "output_filename": out_filename,
            "mime_type": _MIME_TYPES[eff_fmt],
            "timestamps_json": ts_text,
            "segments": ts_data.get("segments", []),
            "mode": mode,
            "quality": quality,
            "bitrate_kbps": bitrate_kbps,
            "ranges_count": len(range_jobs),
            "original_duration": result.original_duration,
            "final_duration": result.final_duration,
            "time_saved_pct": result.time_saved_pct,
            "segments_kept": result.segments_kept,
            "profile_kind": primary_vad.profile.kind,
            "noise_floor_p5_db": primary_vad.profile.noise_floor_p5_db,
            "snr_db": primary_vad.profile.snr_db,
            "prayer_windows_detected": sum(vr.prayer_windows_count for _, _, vr in range_jobs),
            "prayer_segments_stripped": sum(vr.prayer_stripped_count for _, _, vr in range_jobs),
            "stray_shouts_stripped": sum(vr.stray_shouts_stripped for _, _, vr in range_jobs),
            "room_tone_bridge": any(vr.use_room_tone for _, _, vr in range_jobs),
        }


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 860px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .quran-hero {
            direction: rtl;
            text-align: center;
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(15, 118, 110, 0.25);
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(217, 119, 6, 0.06));
            margin-bottom: 1.25rem;
        }
        .quran-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.85rem;
        }
        .quran-hero p {
            margin: 0.2rem 0;
            opacity: 0.9;
            font-size: 0.96rem;
        }
        .metric-card {
            direction: rtl;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(128, 128, 128, 0.05);
            height: 100%;
        }
        .metric-label {
            font-size: 0.82rem;
            opacity: 0.78;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.28rem;
            font-weight: 700;
            color: #0f766e;
        }
        .metric-sub {
            font-size: 0.76rem;
            opacity: 0.72;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics_card(res: dict[str, Any]) -> None:
    st.markdown("#### 📊 ملخص النتائج | Results Summary")
    c1, c2, c3, c4 = st.columns(4)

    orig_str = format_duration_clock(res["original_duration"])
    final_str = format_duration_clock(res["final_duration"])
    saved_str = f"{res['time_saved_pct']:.1f}%"
    prof_str = format_profile_label(res["profile_kind"])

    ranges_cnt = res.get("ranges_count", 1)
    seg_sub = (
        f"{res['segments_kept']} مقاطع · {ranges_cnt} نطاقات"
        if ranges_cnt > 1
        else f"{res['segments_kept']} مقاطع | segments"
    )
    bridge_lbl = " · Room-Tone" if res.get("room_tone_bridge") else ""

    cards = [
        (c1, "المدة الأصلية | Original Duration", orig_str, f"{res['original_duration']:.1f}s"),
        (c2, "المدة بعد التنقية | Final Duration", final_str, seg_sub),
        (c3, "نسبة التوفير | Time Saved %", saved_str, f"{max(0.0, res['original_duration'] - res['final_duration']):.1f}s موفرة"),
        (c4, "نوع التسجيل | Detected Profile", prof_str, f"Floor {res['noise_floor_p5_db']:.0f} dBFS · SNR {res['snr_db']:.0f} dB{bridge_lbl}"),
    ]
    for col, label, val, sub in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if res["mode"] == "prayer":
        st.info(
            f"🕌 تم فحص **{res['prayer_windows_detected']}** انتقالة ركوع/سجود وحذف "
            f"**{res['prayer_segments_stripped']}** مقطع تكبير/مبلّغ مع الحفاظ على الفاتحة والقراءة.\n\n"
            f"Detected **{res['prayer_windows_detected']}** Ruku/Sujud windows and stripped "
            f"**{res['prayer_segments_stripped']}** Takbir/Muballigh segments."
        )
    elif res["mode"] == "natural" and res["prayer_windows_detected"] >= 2:
        st.info(
            f"💡 **تلميح | Hint:** تم رصد **{res['prayer_windows_detected']}** انتقالة صلاة (ركوع/سجود). "
            "لإزالة التكبيرات وصوت المبلّغ مع إبقاء الفاتحة والسور، اختر وضع **صلاة جهرية | Prayer Mode**.\n\n"
            f"Detected **{res['prayer_windows_detected']}** periodic Ruku/Sujud transitions — switch to "
            "**Prayer Mode** to remove Imam/Muballigh Takbirs."
        )


def _execute_with_progress(**job_kwargs: Any) -> None:
    progress_bar = st.progress(0)
    status_box = st.empty()

    def _on_progress(pct: int, msg: str) -> None:
        progress_bar.progress(pct)
        status_box.caption(f"**{pct}%** — {msg}")

    try:
        res = run_cleaning_job(**job_kwargs, progress_cb=_on_progress)
        st.session_state["quran_clean_result"] = res
        status_box.success("✅ تمت تنقية التلاوة بنجاح | Recitation cleaned successfully!")
    except Exception as exc:
        progress_bar.empty()
        status_box.empty()
        st.error(str(exc))


def _ensure_range_state() -> list[dict[str, Any]]:
    if "clip_ranges" not in st.session_state or not st.session_state["clip_ranges"]:
        st.session_state["clip_ranges"] = [{"id": 0, "start": "", "end": ""}]
        st.session_state["clip_range_next_id"] = 1
    if "clip_range_next_id" not in st.session_state:
        st.session_state["clip_range_next_id"] = len(st.session_state["clip_ranges"])
    return st.session_state["clip_ranges"]


def main() -> None:
    st.set_page_config(
        page_title="أنصتوا | Ansitu — مُنقّي التلاوات القرآنية",
        page_icon="🕌",
        layout="centered",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="quran-hero">
            <h1>🕌 أنصتوا — Ansitu</h1>
            <p>أداة متخصصة لتنقية التلاوات القرآنية من السكتات الطويلة، والتكبيرات في صلوات التراويح والتهجد، مع الحفاظ التام على دفء التسجيلات التراثية وجودة الصوت الأصلية بدون تشويه.</p>
            <p>Zero-quality-loss Quran recitation cleaner: removes dead silences, breaths &amp; prayer transitions while preserving vintage tape warmth and full dynamic range.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_external = st.tabs(
        [
            "📁 رفع ملف صوتي أو فيديو | Upload Audio / Video File",
            "🔗 رابط خارجي (YouTube / SoundCloud / رابط مباشر) | External Media URL",
        ]
    )

    with tab_upload:
        uploaded = st.file_uploader(
            "اختر ملف التلاوة (صوت أو فيديو) | Select audio or video file",
            type=_UPLOAD_TYPES,
            help="Supported: MP3, WAV, M4A, MP4, MKV, WEBM, OPUS, FLAC",
        )

    with tab_external:
        ext_url = st.text_input(
            "أدخل الرابط الخارجي (YouTube / SoundCloud / Archive.org / رابط مباشر) | External Media URL",
            value="",
            placeholder="https://youtube.com/... or https://soundcloud.com/... or https://archive.org/...",
        )
        st.caption(
            "يدعم جميع الروابط المدعومة عبر yt-dlp (يوتيوب، ساوند كلاود، أرشيف، وروابط MP3/M4A المباشرة). في حال حظر يوتيوب خوادم السحابة، قم بتنزيل الملف محلياً واستخدم تبويب «رفع ملف».\n"
            "Supports YouTube, SoundCloud, Archive.org, and direct MP3/M4A URLs via yt-dlp."
        )

    with st.container(border=True):
        st.markdown("#### ⚙️ إعدادات التنقية | Settings")
        col_mode, col_fmt = st.columns(2)
        with col_mode:
            mode = st.radio(
                "وضع التشغيل | Mode",
                options=["natural", "prayer"],
                format_func=lambda m: _MODE_LABELS[m],
                index=0,
            )
        with col_fmt:
            out_format = st.radio(
                "صيغة الإخراج | Output Format",
                options=["m4a", "mp3"],
                format_func=lambda f: _FORMAT_LABELS[f],
                index=0,
            )

        q_options = quality_options_for_source(uploaded.name if uploaded is not None else None)
        if (
            "audio_quality_select" in st.session_state
            and st.session_state["audio_quality_select"] not in q_options
        ):
            del st.session_state["audio_quality_select"]
        quality = st.selectbox(
            "جودة الصوت | Audio Quality",
            options=q_options,
            format_func=lambda q: _QUALITY_LABELS[q],
            index=q_options.index("192"),
            key="audio_quality_select",
        )
        if quality == "lossless_wav":
            st.caption(
                "🎧 سيتم التصدير بصيغة WAV غير مضغوطة (Lossless PCM) مع الحفاظ على كامل جودة التسجيل الأصلي."
            )

        st.markdown("##### ✂️ نطاقات التلاوة (اختياري) | Clip Ranges (Optional · Full file if blank)")
        clip_rows = _ensure_range_state()
        for idx, row in enumerate(clip_rows):
            rid = row["id"]
            if len(clip_rows) > 1:
                c_start, c_end, c_del = st.columns([5, 5, 1])
            else:
                c_start, c_end = st.columns(2)
                c_del = None

            with c_start:
                row["start"] = st.text_input(
                    f"بداية المقطع {idx + 1} | Start #{idx + 1} (MM:SS)",
                    value=row["start"],
                    placeholder="00:00",
                    key=f"range_start_{rid}",
                )
            with c_end:
                row["end"] = st.text_input(
                    f"نهاية المقطع {idx + 1} | End #{idx + 1} (MM:SS)",
                    value=row["end"],
                    placeholder="05:00",
                    key=f"range_end_{rid}",
                )
            if c_del is not None:
                with c_del:
                    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"del_range_{rid}", help="حذف هذا المقطع | Remove range"):
                        clip_rows.pop(idx)
                        st.rerun()

        if st.button("➕ أضف مقطعاً آخر | + Add Range", key="add_range_btn"):
            next_id = int(st.session_state["clip_range_next_id"])
            st.session_state["clip_range_next_id"] = next_id + 1
            clip_rows.append({"id": next_id, "start": "", "end": ""})
            st.rerun()

    selected_ranges = [(r["start"], r["end"]) for r in clip_rows]
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        if st.button(
            "✨ ابدأ تنقية الملف | Clean Uploaded Recitation",
            type="primary",
            width="stretch",
            disabled=uploaded is None,
            key="run_upload_btn",
        ):
            assert uploaded is not None
            _execute_with_progress(
                mode=mode,
                ranges=selected_ranges,
                out_format=out_format,
                quality=quality,
                uploaded_name=uploaded.name,
                uploaded_bytes=uploaded.getvalue(),
            )
    with act_col2:
        if st.button(
            "✨ تحميل وتنقية الرابط الخارجي | Download & Clean URL",
            type="primary",
            width="stretch",
            disabled=not ext_url.strip(),
            key="run_external_btn",
        ):
            _execute_with_progress(
                mode=mode,
                ranges=selected_ranges,
                out_format=out_format,
                quality="192" if quality == "lossless_wav" else quality,
                external_url=ext_url.strip(),
            )

    res = st.session_state.get("quran_clean_result")
    if res is not None:
        st.divider()
        _render_metrics_card(res)
        st.markdown("#### 🎧 معاينة وتحميل التلاوة | Preview & Download")
        st.audio(res["audio_bytes"], format=res["mime_type"])
        dl_col1, dl_col2 = st.columns([3, 1])
        with dl_col1:
            st.download_button(
                label=f"⬇️ تحميل التلاوة المنقّاة ({res['output_filename']}) | Download Cleaned Audio",
                data=res["audio_bytes"],
                file_name=res["output_filename"],
                mime=res["mime_type"],
                type="primary",
                width="stretch",
            )
        with dl_col2:
            st.download_button(
                label="📄 timestamps.json",
                data=res["timestamps_json"].encode("utf-8"),
                file_name="timestamps.json",
                mime="application/json",
                width="stretch",
            )

        with st.expander(f"📋 تفاصيل المقاطع المحتفظ بها ({res['segments_kept']}) | Kept Segments"):
            st.dataframe(res["segments"], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
