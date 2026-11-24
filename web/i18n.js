/**
 * Bilingual translations (Arabic RTL default / English LTR) and duration/filename helpers
 * for ansitu web app.
 */

export const TRANSLATIONS = {
  ar: {
    lang: "ar",
    dir: "rtl",
    pageTitle: "أَنصِتُوا",
    brand: "أَنصِتُوا",
    langToggle: "EN",
    langToggleAria: "Switch to English",
    themeLightAria: "تفعيل الوضع الفاتح",
    themeDarkAria: "تفعيل الوضع الداكن",
    installApp: "تثبيت التطبيق",
    iosInstallLine1: "اضغط على زر المشاركة بالأسفل",
    iosInstallLine2: "ثم اختر «إضافة إلى الشاشة الرئيسية»",
    heroTitle: "تلاوات خاشعة، بلا سكتات طويلة",
    heroSubtitle:
      "احذف السكتات الطويلة بين الآيات تلقائياً، مع الحفاظ على كل حرف من التلاوة",
    pickFile: "اختر ملفاً صوتياً",
    dropHint: "أو اسحب الملف وأفلته هنا",
    changeFile: "تغيير الملف",
    startCleaning: "تنقية التلاوة",
    sampleBtn: "أو جرّب بتلاوة جاهزة (الشيخ البهتيمي)",
    sampleLoading: "جاري تجهيز التلاوة التجريبية...",
    formatLabel: "صيغة الملف",
    qualityLabel: "الجودة",
    formatOptions: {
      m4a: "\u2066M4A\u2069 (الأنقى)",
      mp3: "\u2066MP3\u2069 (الأكثر انتشاراً)",
    },
    qualityOptions: {
      "128": "\u2066128 kbps\u2069 (خفيفة)",
      "192": "\u2066192 kbps\u2069 (عالية)",
      "256": "\u2066256 kbps\u2069 (فائقة)",
      lossless_wav: "\u2066WAV\u2069 (بدون فقدان)",
    },
    losslessHint: "سيتم التصدير بصيغة WAV غير مضغوطة مع الحفاظ على كامل الجودة الأصلية.",
    tabStandard: "تلاوة عادية",
    tabPrayer: "تلاوة صلاة",
    trimHint:
      "حدد أوقات البداية والنهاية (مثال: 01:20) لاستبعاد الركوع والسجود أو استخراج مقطع محدد",
    startTime: "البداية",
    endTime: "النهاية",
    rangeNumber: (n) => `الجزء ${n}`,
    addRange: "إضافة جزء آخر",
    removeRange: "حذف",
    stages: {
      loading: "جاري تجهيز محرك الصوت...",
      decoding: "جاري فك ترميز الصوت...",
      analyzing: "جاري تحليل السكتات والتلاوة...",
      splicing: "جاري دمج المقاطع بسلاسة...",
      exporting: "جاري تصدير الملف النهائي...",
    },
    cancel: "إلغاء",
    mobileLongTip:
      "نصيحة: التلاوات الطويلة (أكثر من 45 دقيقة) تُعالج بسرعة أكبر على الحاسوب.",
    resultTitle: "تمت تنقية التلاوة",
    resultSummary: (orig, final, saved) =>
      `قَلّت المدة من ${orig} إلى ${final} (تم توفير ${saved})`,
    downloadBtn: "تحميل التلاوة",
    shareBtn: "مشاركة الأداة",
    shareTitle: "أَنصِتُوا",
    shareText:
      "أداة مجانية تحذف السكتات الطويلة بين الآيات تلقائياً مع الحفاظ الكامل على كل حرف من التلاوة",
    linkCopied: "تم نسخ الرابط",
    processAnother: "معالجة ملف آخر",
    lowSnrTitle: "تعذّر القص التلقائي لحماية التلاوة",
    lowSnrBody:
      "يحتوي هذا التسجيل على ضوضاء خلفية مرتفعة، ولذلك أوقفنا قصّ السكتات تلقائياً تعظيماً للقرآن الكريم وحفاظاً على كل حرف من التلاوة دون أي اقتطاع.",
    chooseAnother: "اختيار ملف آخر",
    feedbackLink: "ملاحظات",
    githubLink: "GitHub",
    feedbackTitle: "ملاحظات أو الإبلاغ عن مشكلة",
    feedbackMessage: "الرسالة",
    feedbackMessagePlaceholder: "اكتب ملاحظتك أو صف المشكلة التي واجهتك...",
    feedbackReciter: "القارئ أو السورة (اختياري)",
    feedbackReciterPlaceholder: "مثال: الشيخ البهتيمي - سورة طه",
    feedbackSend: "إرسال",
    feedbackSending: "جاري الإرسال...",
    feedbackSuccess: "شكراً لك، تم إرسال رسالتك بنجاح.",
    feedbackUnconfigured:
      "لم يتم تفعيل مفتاح Web3Forms بعد. يمكنك إرسال ملاحظتك عبر مستودع GitHub.",
    feedbackError: "تعذّر إرسال الرسالة حالياً. يرجى المحاولة لاحقاً.",
    close: "إغلاق",
    errors: {
      invalidRangeOrder: "يجب أن يكون وقت البداية قبل وقت النهاية.",
      blankRangeRow: "يرجى إدخال وقت البداية أو النهاية لكل جزء مضاف، أو حذفه.",
      invalidTimestamp: "صيغة الوقت غير صحيحة. استخدم MM:SS أو HH:MM:SS.",
      rangeExceedsDuration: "أحد أوقات البداية يتجاوز مدة الملف الصوتي.",
      noSpeechDetected: "لم يتم العثور على مقاطع صوتية واضحة في هذا الملف.",
      sampleMissing: "ملف التلاوة التجريبية غير متاح حالياً.",
      ffmpegFailed: "تعذّر فك ترميز أو تصدير الملف الصوتي. تأكد من سلامة الملف.",
    },
  },
  en: {
    lang: "en",
    dir: "ltr",
    pageTitle: "ansitu",
    brand: "ansitu",
    langToggle: "عربي",
    langToggleAria: "التبديل إلى العربية",
    themeLightAria: "Switch to light mode",
    themeDarkAria: "Switch to dark mode",
    installApp: "Install App",
    iosInstallLine1: "Tap the Share button below",
    iosInstallLine2: "Then choose Add to Home Screen",
    heroTitle: "Reverent recitations, without long pauses",
    heroSubtitle:
      "Automatically remove long pauses between verses while preserving every letter of the recitation.",
    pickFile: "Choose an audio file",
    dropHint: "Or drag and drop an audio file here",
    changeFile: "Change file",
    startCleaning: "Clean recitation",
    sampleBtn: "Or try a sample recitation (Sheikh Al-Bahtimi)",
    sampleLoading: "Loading sample recitation...",
    formatLabel: "Output format",
    qualityLabel: "Audio quality",
    formatOptions: {
      m4a: "M4A (Cleaner)",
      mp3: "MP3 (Universal)",
    },
    qualityOptions: {
      "128": "128 kbps (Light)",
      "192": "192 kbps (High)",
      "256": "256 kbps (Max)",
      lossless_wav: "WAV (Lossless)",
    },
    losslessHint: "Exports as uncompressed Lossless WAV preserving original PCM quality.",
    tabStandard: "Standard Recitation",
    tabPrayer: "Prayer Recitation",
    trimHint:
      "Set start and end times (e.g. 01:20) to exclude ruku' and sujud or extract selected passages",
    startTime: "Start",
    endTime: "End",
    rangeNumber: (n) => `Section ${n}`,
    addRange: "Add another section",
    removeRange: "Remove",
    stages: {
      loading: "Preparing audio engine...",
      decoding: "Decoding audio stream...",
      analyzing: "Analyzing pauses and recitation...",
      splicing: "Splicing segments seamlessly...",
      exporting: "Exporting cleaned audio...",
    },
    cancel: "Cancel",
    mobileLongTip:
      "Tip: Long recitations (over 45 minutes) process faster on desktop.",
    resultTitle: "Recitation cleaned",
    resultSummary: (orig, final, saved) =>
      `Duration reduced from ${orig} to ${final} (${saved} saved)`,
    downloadBtn: "Download audio",
    shareBtn: "Share tool",
    shareTitle: "ansitu",
    shareText:
      "A free tool that automatically trims long silences between verses while preserving every letter of the recitation",
    linkCopied: "Link copied",
    processAnother: "Process another file",
    lowSnrTitle: "Automatic trimming paused to protect the recitation",
    lowSnrBody:
      "This recording has high background noise, so automatic silence removal was paused out of reverence for the Quran and to preserve every letter of the recitation.",
    chooseAnother: "Choose another file",
    feedbackLink: "Feedback",
    githubLink: "GitHub",
    feedbackTitle: "Feedback or report an issue",
    feedbackMessage: "Message",
    feedbackMessagePlaceholder: "Share feedback or describe the issue you encountered...",
    feedbackReciter: "Reciter or Surah (optional)",
    feedbackReciterPlaceholder: "e.g., Sheikh Al-Bahtimi - Surah Taha",
    feedbackSend: "Send",
    feedbackSending: "Sending...",
    feedbackSuccess: "Thank you! Your message has been sent.",
    feedbackUnconfigured:
      "Web3Forms access key is not configured yet. You can also open an issue on GitHub.",
    feedbackError: "Could not send your message right now. Please try again later.",
    close: "Close",
    errors: {
      invalidRangeOrder: "Start timestamp must be earlier than end timestamp.",
      blankRangeRow: "Please enter a start or end timestamp for each section, or remove it.",
      invalidTimestamp: "Invalid timestamp format. Use MM:SS or HH:MM:SS.",
      rangeExceedsDuration: "A section start timestamp exceeds the audio duration.",
      noSpeechDetected: "No clear speech segments were detected in this audio.",
      sampleMissing: "The sample recitation file is not available right now.",
      ffmpegFailed: "Could not decode or export this audio file. Please verify the file.",
    },
  },
};

/**
 * Format seconds as compact `MM:SS` or `HH:MM:SS` matching `app.py`'s `format_duration_clock`.
 */
export function formatClock(seconds) {
  const totalS = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(totalS / 3600);
  const rem = totalS % 3600;
  const minutes = Math.floor(rem / 60);
  const secs = rem % 60;
  const pad = (n) => String(n).padStart(2, "0");
  if (hours > 0) {
    return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
  }
  return `${pad(minutes)}:${pad(secs)}`;
}

/**
 * Parse `MM:SS`, `HH:MM:SS`, or numeric seconds into non-negative float seconds
 * matching `quran_cleaner/media.py`'s `parse_timestamp`.
 */
export function parseTimestamp(value) {
  const raw = String(value ?? "").trim();
  if (!raw) {
    throw new Error("INVALID_TIMESTAMP");
  }

  if (!raw.includes(":")) {
    const secs = Number(raw);
    if (!Number.isFinite(secs) || secs < 0) {
      throw new Error("INVALID_TIMESTAMP");
    }
    return secs;
  }

  const parts = raw.split(":");
  if (parts.length !== 2 && parts.length !== 3) {
    throw new Error("INVALID_TIMESTAMP");
  }

  const nums = parts.map((p) => (p.trim() === "" ? NaN : Number(p)));
  if (nums.some((n) => !Number.isFinite(n))) {
    throw new Error("INVALID_TIMESTAMP");
  }

  if (nums.length === 2) {
    const [mins, secs] = nums;
    if (mins < 0 || secs < 0 || secs >= 60.0 || !Number.isInteger(mins)) {
      throw new Error("INVALID_TIMESTAMP");
    }
    return mins * 60.0 + secs;
  }

  const [hours, mins, secs] = nums;
  if (
    hours < 0 ||
    mins < 0 ||
    secs < 0 ||
    mins >= 60.0 ||
    secs >= 60.0 ||
    !Number.isInteger(hours) ||
    !Number.isInteger(mins)
  ) {
    throw new Error("INVALID_TIMESTAMP");
  }
  return hours * 3600.0 + mins * 60.0 + secs;
}

/**
 * Build smart output download filename, stripping legacy downloader bitrate/format tags
 * and appending the actual exported quality/extension (e.g. `<name>_ansitu_192k.m4a`).
 */
export function buildOutputFilename(originalName, ext, quality = null) {
  const cleanExt = String(ext || "m4a").replace(/^\.+/, "").toLowerCase();
  const base = String(originalName || "").trim();
  let stem = base ? base.replace(/\.[^./\\]+$/, "") : "";

  // Strip closed tags like `(MP3_160K)`, `(M4A_128K)`, `(128k)`, `[AAC_192kbps]`
  stem = stem.replace(
    /[\s_-]*[\(\[]\s*(?:(?:mp3|m4a|aac|opus|ogg|wav|flac|webm)[_\s-]*\d*\s*k?(?:bps)?|\d{2,3}\s*k(?:bps)?)\s*[\)\]]/gi,
    ""
  );
  // Strip truncated trailing downloader tags like `(MP3_1`, `(MP3__1`, `(MP3_16`
  stem = stem.replace(
    /[\s_-]*[\(\[]\s*(?:(?:mp3|m4a|aac|opus|ogg|wav|flac|webm)[_\s-]*\d*|\d{2,3}\s*k(?:bps)?)\s*[\)\]]?\s*$/gi,
    ""
  );

  const safeStem = stem.trim() || "recitation";
  if (quality === "lossless_wav") {
    return `${safeStem}_ansitu_lossless.${cleanExt}`;
  }
  if (quality !== null && quality !== undefined && String(quality).trim() !== "") {
    const cleanQ = String(quality).trim().replace(/k$/i, "");
    return `${safeStem}_ansitu_${cleanQ}k.${cleanExt}`;
  }
  return `${safeStem}_ansitu.${cleanExt}`;
}
