"""Bilingual (Arabic-first / English) Streamlit web app for quran_cleaner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from quran_cleaner.media import (
    download_youtube_audio,
    extract_vad_audio,
    is_youtube_url,
    parse_timestamp,
    probe_media,
    require_ffmpeg,
)
from quran_cleaner.processor import render_clean_media
from quran_cleaner.vad import VADConfig, detect_speech_segments

_MODE_LABELS = {
    "natural": "تلاوة طبيعية | Natural Recitation",
    "prayer": "صلاة جهرية | Prayer Mode",
}

_FORMAT_LABELS = {
    "m4a": "M4A — جودة عالية (AAC High Quality)",
    "mp3": "MP3 — متوافق مع كل الأجهزة (Universal)",
}

_MIME_TYPES = {
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
}

_UPLOAD_TYPES = ["mp3", "wav", "m4a", "mp4", "mkv", "webm", "opus", "flac"]


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


def format_youtube_error(exc: Exception) -> str:
    """Return a graceful bilingual message when YouTube blocks datacenter IPs or fails."""
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
            "تعذّر التحميل من يوتيوب لأن خوادم السحابة (Datacenter IP) محجوبة مؤقتاً من قِبل يوتيوب.\n\n"
            "الحل البديل الموصى به: قم بتنزيل الملف الصوتي أو الفيديو على جهازك أولاً، ثم استخدم تبويب "
            "«رفع ملف صوتي أو فيديو | Upload Audio/Video File» بالأعلى.\n\n"
            "YouTube blocked this server's datacenter IP (bot-check / HTTP 403). "
            "Please download the audio/video locally on your device and use the primary "
            "'Upload Audio/Video File' tab instead."
        )
    return (
        "تعذّر تحميل الصوت من رابط يوتيوب. يُفضّل تحميل الملف محلياً ثم رفعه عبر تبويب «رفع ملف».\n\n"
        f"YouTube download failed ({detail}). Try downloading locally and uploading via the File tab."
    )


def run_cleaning_job(
    *,
    mode: str,
    start_text: str,
    end_text: str,
    out_format: str,
    uploaded_name: Optional[str] = None,
    uploaded_bytes: Optional[bytes] = None,
    youtube_url: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Execute end-to-end download/load, VAD analysis, slicing, and in-memory export."""
    def _report(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0, min(100, pct)), msg)

    fmt = out_format.lower().lstrip(".")
    if fmt not in {"m4a", "mp3"}:
        raise ValueError(f"Unsupported output format: {out_format}")

    start_s, end_s = parse_optional_range(start_text, end_text)
    require_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="quran_web_") as tmp_root:
        work_dir = Path(tmp_root)

        if youtube_url is not None:
            url = youtube_url.strip()
            if not is_youtube_url(url):
                raise ValueError(
                    "رابط يوتيوب غير صالح | Please enter a valid youtube.com or youtu.be URL."
                )
            _report(8, "جاري التحميل من يوتيوب... | Downloading audio from YouTube...")
            try:
                input_path = download_youtube_audio(url, work_dir)
            except Exception as exc:
                raise RuntimeError(format_youtube_error(exc)) from exc
            source_stem = input_path.stem
        elif uploaded_bytes is not None and uploaded_name:
            _report(10, "جاري تجهيز الملف... | Preparing uploaded file...")
            safe_name = Path(uploaded_name).name or "recitation.mp3"
            input_path = work_dir / safe_name
            input_path.write_bytes(uploaded_bytes)
            source_stem = input_path.stem
        else:
            raise ValueError(
                "يرجى اختيار ملف أو إدخال رابط يوتيوب | Please provide a media file or YouTube URL."
            )

        _report(22, "جاري فحص خصائص الصوت... | Probing audio properties...")
        info = probe_media(input_path)

        clip_start = start_s if start_s is not None else 0.0
        clip_end = min(end_s, info.duration) if end_s is not None else info.duration
        if clip_start >= info.duration:
            raise ValueError(
                f"وقت البداية ({clip_start:.1f}s) يتجاوز مدة الملف ({info.duration:.1f}s) | "
                f"Start time exceeds file duration."
            )
        if clip_start >= clip_end:
            raise ValueError(
                "وقت البداية يجب أن يكون قبل وقت النهاية | Start must be earlier than end."
            )
        clip_duration = clip_end - clip_start

        _report(30, "جاري استخراج مسار التحليل (16kHz)... | Extracting 16kHz VAD stream...")
        audio_16k = extract_vad_audio(
            input_path,
            start_s=start_s,
            end_s=clip_end if end_s is not None else None,
        )

        config = VADConfig(mode=mode)  # type: ignore[arg-type]

        def _on_vad(frac: float) -> None:
            pct = 38 + int(round(frac * 30))
            _report(pct, "جاري تحليل الصوت والتوقفات (VAD)... | Analyzing speech & pauses...")

        _report(38, "جاري تحليل الصوت والتوقفات (VAD)... | Analyzing speech & pauses...")
        vad_result = detect_speech_segments(
            audio_16k=audio_16k,
            config=config,
            exact_duration=clip_duration,
            clip_start_s=clip_start,
            progress_cb=_on_vad,
        )

        out_filename = f"{source_stem}_clean.{fmt}"
        output_path = work_dir / out_filename
        timestamps_path = work_dir / "timestamps.json"

        def _on_render(frac: float) -> None:
            pct = 70 + int(round(frac * 27))
            _report(pct, "جاري تقطيع ودمج الصوت بجودته الأصلية... | Slicing & crossfading native audio...")

        _report(70, "جاري تقطيع ودمج الصوت بجودته الأصلية... | Slicing & crossfading native audio...")
        result = render_clean_media(
            input_path=input_path,
            output_path=output_path,
            timestamps_path=timestamps_path,
            info=info,
            config=config,
            vad_result=vad_result,
            clip_start_s=start_s,
            clip_end_s=clip_end if end_s is not None else None,
            progress_cb=_on_render,
        )

        _report(100, "اكتملت التنقية بنجاح | Completed successfully!")
        ts_text = timestamps_path.read_text(encoding="utf-8")
        ts_data = json.loads(ts_text)

        return {
            "audio_bytes": output_path.read_bytes(),
            "output_filename": out_filename,
            "mime_type": _MIME_TYPES[fmt],
            "timestamps_json": ts_text,
            "segments": ts_data.get("segments", []),
            "mode": mode,
            "original_duration": result.original_duration,
            "final_duration": result.final_duration,
            "time_saved_pct": result.time_saved_pct,
            "segments_kept": result.segments_kept,
            "profile_kind": vad_result.profile.kind,
            "noise_floor_p5_db": vad_result.profile.noise_floor_p5_db,
            "snr_db": vad_result.profile.snr_db,
            "prayer_windows_detected": vad_result.prayer_windows_count,
            "prayer_segments_stripped": vad_result.prayer_stripped_count,
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

    cards = [
        (c1, "المدة الأصلية | Original Duration", orig_str, f"{res['original_duration']:.1f}s"),
        (c2, "المدة بعد التنقية | Final Duration", final_str, f"{res['segments_kept']} مقاطع | segments"),
        (c3, "نسبة التوفير | Time Saved %", saved_str, f"{max(0.0, res['original_duration'] - res['final_duration']):.1f}s موفرة"),
        (c4, "نوع التسجيل | Detected Profile", prof_str, f"Floor {res['noise_floor_p5_db']:.0f} dBFS · SNR {res['snr_db']:.0f} dB"),
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


def main() -> None:
    st.set_page_config(
        page_title="مُنقّي التلاوات القرآنية | Quran Recitation Cleaner",
        page_icon="🕌",
        layout="centered",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="quran-hero">
            <h1>🕌 مُنقّي التلاوات القرآنية | Quran Recitation Cleaner</h1>
            <p>إزالة فترات السكتات الطويلة والتكبيرات وتنقية التلاوات التراثية والاستوديو مع الحفاظ التام على دفء الصوت وديناميكية القراءة بدون ضغط أو تشويه.</p>
            <p>Zero-quality-loss silence, breath &amp; Takbir trimmer preserving original analog warmth and dynamic range.</p>
        </div>
        """,
        unsafe_allow_html=True,
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

        col_start, col_end = st.columns(2)
        with col_start:
            start_text = st.text_input(
                "بداية المقطع (اختياري) | Start Timestamp (MM:SS)",
                value="",
                placeholder="00:00",
            )
        with col_end:
            end_text = st.text_input(
                "نهاية المقطع (اختياري) | End Timestamp (MM:SS)",
                value="",
                placeholder="05:00",
            )

    tab_upload, tab_youtube = st.tabs(
        [
            "📁 رفع ملف صوتي أو فيديو | Upload Audio / Video File",
            "▶️ رابط يوتيوب | YouTube URL",
        ]
    )

    with tab_upload:
        uploaded = st.file_uploader(
            "اختر ملف التلاوة (صوت أو فيديو) | Select audio or video file",
            type=_UPLOAD_TYPES,
            help="Supported: MP3, WAV, M4A, MP4, MKV, WEBM, OPUS, FLAC",
        )
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
                start_text=start_text,
                end_text=end_text,
                out_format=out_format,
                uploaded_name=uploaded.name,
                uploaded_bytes=uploaded.getvalue(),
            )

    with tab_youtube:
        yt_url = st.text_input(
            "أدخل رابط الفيديو من يوتيوب | Enter YouTube URL",
            value="",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        st.caption(
            "ملاحظة: في بعض استضافات السحابة قد يمنع يوتيوب التحميل التلقائي؛ في هذه الحالة قم بتحميل الملف محلياً واستخدم تبويب «رفع ملف».\n"
            "Note: Cloud datacenter IPs are sometimes blocked by YouTube; if that happens, download locally and use the File Upload tab."
        )
        if st.button(
            "✨ تحميل وتنقية رابط يوتيوب | Download & Clean YouTube",
            type="primary",
            width="stretch",
            disabled=not yt_url.strip(),
            key="run_youtube_btn",
        ):
            _execute_with_progress(
                mode=mode,
                start_text=start_text,
                end_text=end_text,
                out_format=out_format,
                youtube_url=yt_url.strip(),
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
