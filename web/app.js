/**
 * Main UI controller for ansitu web app.
 * Connects minimalist RTL/LTR interface, dynamic `splicer.js` quality presets,
 * dedicated processing Web Worker, PWA install prompts, and Web3Forms feedback modal.
 */

import {
  GITHUB_REPO_URL,
  MOBILE_LONG_AUDIO_SECONDS,
  WEB3FORMS_ACCESS_KEY,
  WEB3FORMS_ENDPOINT,
} from "./config.js";
import {
  DEFAULT_QUALITY_PRESET,
  qualityOptionsForSource,
} from "./engine/splicer.js";
import {
  TRANSLATIONS,
  buildOutputFilename,
  formatClock,
  parseTimestamp,
} from "./i18n.js";

const STORAGE_LANG = "ansitu_lang";
const STORAGE_THEME = "ansitu_theme";

const state = {
  lang: localStorage.getItem(STORAGE_LANG) === "en" ? "en" : "ar",
  theme: resolveInitialTheme(),
  view: "input",
  selectedFile: null,
  mode: "standard",
  rangeAction: "remove",
  ranges: [{ id: 1, start: "", end: "" }],
  nextRangeId: 2,
  worker: null,
  downloadUrl: null,
  lastResult: null,
  lastStage: "loading",
  deferredInstallPrompt: null,
  toastTimer: null,
};

const els = {
  html: document.documentElement,
  themeColorMeta: document.querySelector('meta[name="theme-color"]'),
  brandName: document.getElementById("brand-name"),
  langToggle: document.getElementById("lang-toggle"),
  themeToggle: document.getElementById("theme-toggle"),
  themeIconMoon: document.getElementById("theme-icon-moon"),
  themeIconSun: document.getElementById("theme-icon-sun"),
  installBtn: document.getElementById("install-btn"),
  installBtnLabel: document.getElementById("install-btn-label"),
  iosPopover: document.getElementById("ios-install-popover"),
  iosLine1: document.getElementById("ios-line-1"),
  iosLine2: document.getElementById("ios-line-2"),
  iosPopoverClose: document.getElementById("ios-popover-close"),

  views: {
    input: document.getElementById("view-input"),
    processing: document.getElementById("view-processing"),
    result: document.getElementById("view-result"),
    lowSnr: document.getElementById("view-low-snr"),
  },

  heroTitle: document.getElementById("hero-title"),
  heroSubtitle: document.getElementById("hero-subtitle"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  pickFileLabel: document.getElementById("pick-file-label"),
  dropHintLabel: document.getElementById("drop-hint-label"),
  selectedFileBar: document.getElementById("selected-file-bar"),
  selectedFileName: document.getElementById("selected-file-name"),
  selectedFileSize: document.getElementById("selected-file-size"),
  changeFileBtn: document.getElementById("change-file-btn"),
  formatField: document.getElementById("format-field"),
  formatLabel: document.getElementById("format-label"),
  formatSelect: document.getElementById("format-select"),
  qualityLabel: document.getElementById("quality-label"),
  qualitySelect: document.getElementById("quality-select"),
  losslessHint: document.getElementById("lossless-hint"),
  tabStandard: document.getElementById("tab-standard"),
  tabPrayer: document.getElementById("tab-prayer"),
  trimPanel: document.getElementById("trim-panel"),
  rangeActionRemove: document.getElementById("range-action-remove"),
  rangeActionKeep: document.getElementById("range-action-keep"),
  trimHint: document.getElementById("trim-hint"),
  rangesList: document.getElementById("ranges-list"),
  addRangeBtn: document.getElementById("add-range-btn"),
  addRangeLabel: document.getElementById("add-range-label"),
  inputError: document.getElementById("input-error"),
  startBtn: document.getElementById("start-btn"),

  processingFilename: document.getElementById("processing-filename"),
  processingStage: document.getElementById("processing-stage"),
  progressBarWrap: document.getElementById("progress-bar-wrap"),
  progressFill: document.getElementById("progress-fill"),
  progressPct: document.getElementById("progress-pct"),
  mobileLongTip: document.getElementById("mobile-long-tip"),
  cancelBtn: document.getElementById("cancel-btn"),

  resultTitle: document.getElementById("result-title"),
  resultSummary: document.getElementById("result-summary"),
  resultNotice: document.getElementById("result-notice"),
  downloadBtn: document.getElementById("download-btn"),
  shareBtn: document.getElementById("share-btn"),
  anotherBtn: document.getElementById("another-btn"),

  lowSnrTitle: document.getElementById("low-snr-title"),
  lowSnrBody: document.getElementById("low-snr-body"),
  guardAnotherBtn: document.getElementById("guard-another-btn"),

  feedbackOpenBtn: document.getElementById("feedback-open-btn"),
  githubLink: document.getElementById("github-link"),
  feedbackDialog: document.getElementById("feedback-dialog"),
  feedbackForm: document.getElementById("feedback-form"),
  feedbackDialogTitle: document.getElementById("feedback-dialog-title"),
  feedbackCloseX: document.getElementById("feedback-close-x"),
  feedbackMsgLabel: document.getElementById("feedback-msg-label"),
  feedbackMessage: document.getElementById("feedback-message"),
  feedbackReciterLabel: document.getElementById("feedback-reciter-label"),
  feedbackReciter: document.getElementById("feedback-reciter"),
  feedbackStatus: document.getElementById("feedback-status"),
  feedbackSubmitBtn: document.getElementById("feedback-submit-btn"),
  feedbackCancelBtn: document.getElementById("feedback-cancel-btn"),
  toast: document.getElementById("toast"),
};

function resolveInitialTheme() {
  const saved = localStorage.getItem(STORAGE_THEME);
  if (saved === "dark" || saved === "light") {
    return saved;
  }
  return window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function t() {
  return TRANSLATIONS[state.lang];
}

function isMobileBrowser() {
  const coarse =
    window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  const ua = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  return coarse || ua;
}

function isIosSafari() {
  const ua = navigator.userAgent || "";
  const isIos =
    /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  return isIos && /Safari/i.test(ua) && !/CriOS|FxiOS|EdgiOS/i.test(ua);
}

function isStandaloneMode() {
  return (
    (window.matchMedia &&
      window.matchMedia("(display-mode: standalone)").matches) ||
    window.navigator.standalone === true
  );
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showView(name) {
  state.view = name;
  els.views.input.hidden = name !== "input";
  els.views.processing.hidden = name !== "processing";
  els.views.result.hidden = name !== "result";
  els.views.lowSnr.hidden = name !== "lowSnr";
}

function showInputError(msg) {
  if (!msg) {
    els.inputError.hidden = true;
    els.inputError.textContent = "";
    return;
  }
  els.inputError.textContent = msg;
  els.inputError.hidden = false;
}

function showToast(message) {
  if (state.toastTimer) {
    clearTimeout(state.toastTimer);
  }
  els.toast.textContent = message;
  els.toast.hidden = false;
  state.toastTimer = setTimeout(() => {
    els.toast.hidden = true;
  }, 2400);
}

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem(STORAGE_THEME, theme);
  els.html.setAttribute("data-theme", theme);
  const dark = theme === "dark";
  els.themeIconMoon.hidden = dark;
  els.themeIconSun.hidden = !dark;
  els.themeToggle.setAttribute(
    "aria-label",
    dark ? t().themeLightAria : t().themeDarkAria
  );
  if (els.themeColorMeta) {
    els.themeColorMeta.setAttribute("content", dark ? "#0b0f0c" : "#15803d");
  }
}

function getFormatOptionLabel(fmt) {
  return t().formatOptions[fmt] || fmt.toUpperCase();
}

function getQualityOptionLabel(id) {
  return t().qualityOptions[id] || id;
}

function populateFormatOptions() {
  const prev = els.formatSelect.value || "m4a";
  els.formatSelect.innerHTML = "";
  for (const fmt of ["m4a", "mp3"]) {
    const opt = document.createElement("option");
    opt.value = fmt;
    opt.textContent = getFormatOptionLabel(fmt);
    els.formatSelect.appendChild(opt);
  }
  els.formatSelect.value = prev === "mp3" ? "mp3" : "m4a";
}

function populateQualityOptions() {
  const fileName = state.selectedFile ? state.selectedFile.name : null;
  const options = qualityOptionsForSource(fileName);
  const prev = els.qualitySelect.value || DEFAULT_QUALITY_PRESET;

  els.qualitySelect.innerHTML = "";
  for (const id of options) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = getQualityOptionLabel(id);
    els.qualitySelect.appendChild(opt);
  }

  els.qualitySelect.value = options.includes(prev) ? prev : DEFAULT_QUALITY_PRESET;
  syncLosslessState();
}

function syncLosslessState() {
  const isLossless = els.qualitySelect.value === "lossless_wav";
  els.formatSelect.disabled = isLossless;
  els.losslessHint.hidden = !isLossless;
}

function renderRanges() {
  const tr = t();
  els.rangesList.innerHTML = "";
  state.ranges.forEach((row, idx) => {
    const wrap = document.createElement("div");
    wrap.className = "range-row";

    const startField = document.createElement("div");
    startField.className = "field";
    const startLbl = document.createElement("label");
    startLbl.textContent = `${tr.startTime} (${tr.rangeNumber(idx + 1)})`;
    const startInp = document.createElement("input");
    startInp.type = "text";
    startInp.inputMode = "numeric";
    startInp.placeholder = "00:00";
    startInp.value = row.start;
    startInp.addEventListener("input", (e) => {
      row.start = e.target.value;
      showInputError("");
    });
    startField.append(startLbl, startInp);

    const endField = document.createElement("div");
    endField.className = "field";
    const endLbl = document.createElement("label");
    endLbl.textContent = `${tr.endTime} (${tr.rangeNumber(idx + 1)})`;
    const endInp = document.createElement("input");
    endInp.type = "text";
    endInp.inputMode = "numeric";
    endInp.placeholder = "05:00";
    endInp.value = row.end;
    endInp.addEventListener("input", (e) => {
      row.end = e.target.value;
      showInputError("");
    });
    endField.append(endLbl, endInp);

    const rmBtn = document.createElement("button");
    rmBtn.type = "button";
    rmBtn.className = "range-remove-btn";
    rmBtn.textContent = tr.removeRange;
    rmBtn.hidden = state.ranges.length <= 1;
    rmBtn.addEventListener("click", () => {
      state.ranges = state.ranges.filter((r) => r.id !== row.id);
      renderRanges();
    });

    wrap.append(startField, endField, rmBtn);
    els.rangesList.appendChild(wrap);
  });
}

function renderResultSummary() {
  if (!state.lastResult) {
    els.resultNotice.hidden = true;
    return;
  }
  const orig = formatClock(state.lastResult.originalDuration);
  const fin = formatClock(state.lastResult.finalDuration);
  const saved = formatClock(
    Math.max(0, state.lastResult.originalDuration - state.lastResult.finalDuration)
  );
  const tr = t();
  els.resultSummary.textContent = tr.resultSummary(orig, fin, saved);
  if (state.lastResult.silenceTrimSkipped) {
    els.resultNotice.textContent = tr.resultSilenceSkippedNotice;
    els.resultNotice.hidden = false;
  } else {
    els.resultNotice.hidden = true;
    els.resultNotice.textContent = "";
  }
}

function applyLanguage(lang) {
  state.lang = lang === "en" ? "en" : "ar";
  localStorage.setItem(STORAGE_LANG, state.lang);
  const tr = t();

  document.title = tr.pageTitle;
  els.html.setAttribute("lang", tr.lang);
  els.html.setAttribute("dir", tr.dir);

  els.brandName.textContent = tr.brand;
  els.langToggle.textContent = tr.langToggle;
  els.langToggle.setAttribute("aria-label", tr.langToggleAria);
  els.themeToggle.setAttribute(
    "aria-label",
    state.theme === "dark" ? tr.themeLightAria : tr.themeDarkAria
  );
  els.installBtnLabel.textContent = tr.installApp;
  els.iosLine1.textContent = tr.iosInstallLine1;
  els.iosLine2.textContent = tr.iosInstallLine2;

  els.heroTitle.textContent = tr.heroTitle;
  els.heroSubtitle.textContent = tr.heroSubtitle;
  els.pickFileLabel.textContent = tr.pickFile;
  els.dropHintLabel.textContent = tr.dropHint;
  els.changeFileBtn.textContent = tr.changeFile;
  els.startBtn.textContent = tr.startCleaning;
  els.formatLabel.textContent = tr.formatLabel;
  els.qualityLabel.textContent = tr.qualityLabel;
  els.losslessHint.textContent = tr.losslessHint;
  els.tabStandard.textContent = tr.tabStandard;
  els.tabPrayer.textContent = tr.tabPrayer;
  els.rangeActionRemove.textContent = tr.rangeActionRemove;
  els.rangeActionKeep.textContent = tr.rangeActionKeep;
  els.trimHint.textContent =
    state.rangeAction === "keep" ? tr.trimHintKeep : tr.trimHintRemove;
  els.addRangeLabel.textContent = tr.addRange;

  els.processingStage.textContent =
    tr.stages[state.lastStage] || tr.stages.loading;
  els.mobileLongTip.textContent = tr.mobileLongTip;
  els.cancelBtn.textContent = tr.cancel;

  els.resultTitle.textContent = tr.resultTitle;
  renderResultSummary();
  els.downloadBtn.textContent = tr.downloadBtn;
  els.shareBtn.textContent = tr.shareBtn;
  els.anotherBtn.textContent = tr.processAnother;

  els.lowSnrTitle.textContent = tr.lowSnrTitle;
  els.lowSnrBody.textContent = tr.lowSnrBody;
  els.guardAnotherBtn.textContent = tr.chooseAnother;

  els.feedbackOpenBtn.textContent = tr.feedbackLink;
  els.githubLink.textContent = tr.githubLink;
  els.githubLink.href = GITHUB_REPO_URL;
  els.feedbackDialogTitle.textContent = tr.feedbackTitle;
  els.feedbackMsgLabel.textContent = tr.feedbackMessage;
  els.feedbackMessage.placeholder = tr.feedbackMessagePlaceholder;
  els.feedbackReciterLabel.textContent = tr.feedbackReciter;
  els.feedbackReciter.placeholder = tr.feedbackReciterPlaceholder;
  els.feedbackSubmitBtn.textContent = tr.feedbackSend;
  els.feedbackCancelBtn.textContent = tr.close;

  populateFormatOptions();
  populateQualityOptions();
  renderRanges();
}

function setSelectedFile(file) {
  state.selectedFile = file;
  showInputError("");
  if (!file) {
    els.dropzone.hidden = false;
    els.selectedFileBar.hidden = true;
    els.startBtn.disabled = true;
    populateQualityOptions();
    return;
  }
  els.selectedFileName.textContent = file.name;
  els.selectedFileSize.textContent = formatFileSize(file.size);
  els.dropzone.hidden = true;
  els.selectedFileBar.hidden = false;
  els.startBtn.disabled = false;
  populateQualityOptions();
}

function setMode(mode) {
  const isPrayer = mode === "prayer";
  state.mode = isPrayer ? "prayer" : "standard";
  els.tabStandard.classList.toggle("is-active", !isPrayer);
  els.tabStandard.setAttribute("aria-selected", String(!isPrayer));
  els.tabPrayer.classList.toggle("is-active", isPrayer);
  els.tabPrayer.setAttribute("aria-selected", String(isPrayer));
  els.trimPanel.hidden = !isPrayer;
  showInputError("");
}

function setRangeAction(action) {
  const isKeep = action === "keep";
  state.rangeAction = isKeep ? "keep" : "remove";
  els.rangeActionRemove.classList.toggle("is-active", !isKeep);
  els.rangeActionRemove.setAttribute("aria-checked", String(!isKeep));
  els.rangeActionKeep.classList.toggle("is-active", isKeep);
  els.rangeActionKeep.setAttribute("aria-checked", String(isKeep));
  const tr = t();
  els.trimHint.textContent = isKeep ? tr.trimHintKeep : tr.trimHintRemove;
  showInputError("");
}

function collectValidatedRanges() {
  if (state.mode !== "prayer") {
    return [];
  }
  const hasAnyValue = state.ranges.some(
    (r) => r.start.trim() !== "" || r.end.trim() !== ""
  );
  if (!hasAnyValue) {
    return [];
  }

  const parsed = [];
  for (const row of state.ranges) {
    const sTxt = row.start.trim();
    const eTxt = row.end.trim();
    if (state.ranges.length > 1 && !sTxt && !eTxt) {
      throw new Error(t().errors.blankRangeRow);
    }
    let startS = null;
    let endS = null;
    try {
      startS = sTxt ? parseTimestamp(sTxt) : null;
      endS = eTxt ? parseTimestamp(eTxt) : null;
    } catch {
      throw new Error(t().errors.invalidTimestamp);
    }
    if (startS !== null && endS !== null && startS >= endS) {
      throw new Error(t().errors.invalidRangeOrder);
    }
    parsed.push({ startS, endS });
  }
  return parsed;
}

function terminateWorker() {
  if (state.worker) {
    state.worker.terminate();
    state.worker = null;
  }
}

function cleanupWorkerAndDownloadUrl() {
  terminateWorker();
  if (state.downloadUrl) {
    URL.revokeObjectURL(state.downloadUrl);
    state.downloadUrl = null;
  }
}

function mapWorkerError(code) {
  const errs = t().errors;
  switch (code) {
    case "INVALID_RANGE_ORDER":
      return errs.invalidRangeOrder;
    case "RANGE_EXCEEDS_DURATION":
      return errs.rangeExceedsDuration;
    case "ALL_AUDIO_REMOVED":
      return errs.allAudioRemoved;
    case "NO_SPEECH_DETECTED":
      return errs.noSpeechDetected;
    default:
      return errs.ffmpegFailed;
  }
}

function updateProgressUI(stage, pct) {
  state.lastStage = stage || "loading";
  const clamped = Math.max(0, Math.min(100, Math.round(pct || 0)));
  els.processingStage.textContent =
    t().stages[state.lastStage] || t().stages.loading;
  els.progressFill.style.width = `${clamped}%`;
  els.progressPct.textContent = `${clamped}%`;
  els.progressBarWrap.setAttribute("aria-valuenow", String(clamped));
}

async function startProcessing(file) {
  if (!file) return;

  let parsedRanges;
  try {
    parsedRanges = collectValidatedRanges();
  } catch (err) {
    showInputError(err.message);
    return;
  }

  cleanupWorkerAndDownloadUrl();
  showInputError("");
  els.mobileLongTip.hidden = true;
  els.processingFilename.textContent = file.name;
  updateProgressUI("loading", 2);
  showView("processing");

  let buffer;
  try {
    buffer = await file.arrayBuffer();
  } catch {
    showView("input");
    showInputError(t().errors.ffmpegFailed);
    return;
  }

  const worker = new Worker("./worker.js", { type: "module" });
  state.worker = worker;

  worker.onmessage = (event) => {
    const msg = event.data;
    if (!msg || state.worker !== worker) {
      return;
    }

    if (msg.type === "progress") {
      updateProgressUI(msg.stage, msg.pct);
      return;
    }

    if (msg.type === "meta") {
      if (isMobileBrowser() && msg.duration > MOBILE_LONG_AUDIO_SECONDS) {
        els.mobileLongTip.hidden = false;
      }
      return;
    }

    if (msg.type === "low_snr_guarded") {
      terminateWorker();
      showView("lowSnr");
      return;
    }

    if (msg.type === "done") {
      const blob = new Blob([msg.buffer], { type: msg.mimeType });
      state.downloadUrl = URL.createObjectURL(blob);
      // Immediately reclaim ffmpeg.wasm linear memory heap once final Blob URL is created
      terminateWorker();
      state.lastResult = {
        originalDuration: msg.originalDuration,
        finalDuration: msg.finalDuration,
        ext: msg.ext,
        silenceTrimSkipped: Boolean(msg.silenceTrimSkipped),
      };
      els.downloadBtn.href = state.downloadUrl;
      els.downloadBtn.download = buildOutputFilename(file.name, msg.ext, msg.quality);
      renderResultSummary();
      showView("result");
      return;
    }

    if (msg.type === "error") {
      terminateWorker();
      showView("input");
      showInputError(mapWorkerError(msg.code));
    }
  };

  worker.onerror = () => {
    if (state.worker !== worker) return;
    terminateWorker();
    showView("input");
    showInputError(t().errors.ffmpegFailed);
  };

  worker.postMessage(
    {
      type: "process",
      fileBuffer: buffer,
      fileName: file.name,
      outFormat: els.formatSelect.value,
      quality: els.qualitySelect.value,
      ranges: parsedRanges,
      mode: state.mode,
      rangeAction: state.rangeAction,
    },
    [buffer]
  );
  buffer = null;
}

function resetToInput() {
  cleanupWorkerAndDownloadUrl();
  state.lastResult = null;
  els.fileInput.value = "";
  setSelectedFile(null);
  showView("input");
}

async function handleShareClick() {
  const tr = t();
  const shareUrl = window.location.href;
  if (typeof navigator.share === "function" && isMobileBrowser()) {
    try {
      await navigator.share({
        title: tr.shareTitle,
        text: tr.shareText,
        url: shareUrl,
      });
      return;
    } catch (err) {
      if (err && err.name === "AbortError") {
        return;
      }
    }
  }

  try {
    await navigator.clipboard.writeText(shareUrl);
    showToast(tr.linkCopied);
  } catch {
    // Fallback copy via temporary input if clipboard API is unavailable
    const temp = document.createElement("input");
    temp.value = shareUrl;
    document.body.appendChild(temp);
    temp.select();
    document.execCommand("copy");
    temp.remove();
    showToast(tr.linkCopied);
  }
}

function setupPwaInstall() {
  if ("serviceWorker" in navigator) {
    const registerSw = () => {
      navigator.serviceWorker.register("./sw.js").catch(() => {});
    };
    if (document.readyState === "complete") {
      registerSw();
    } else {
      window.addEventListener("load", registerSw, { once: true });
    }
  }

  if (isStandaloneMode()) {
    els.installBtn.hidden = true;
    return;
  }

  if (isIosSafari()) {
    els.installBtn.hidden = false;
  }

  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    state.deferredInstallPrompt = e;
    els.installBtn.hidden = false;
  });

  window.addEventListener("appinstalled", () => {
    state.deferredInstallPrompt = null;
    els.installBtn.hidden = true;
    els.iosPopover.hidden = true;
  });

  els.installBtn.addEventListener("click", async () => {
    if (state.deferredInstallPrompt) {
      state.deferredInstallPrompt.prompt();
      const choice = await state.deferredInstallPrompt.userChoice;
      state.deferredInstallPrompt = null;
      if (choice && choice.outcome === "accepted") {
        els.installBtn.hidden = true;
      }
      return;
    }
    if (isIosSafari()) {
      els.iosPopover.hidden = !els.iosPopover.hidden;
    }
  });

  els.iosPopoverClose.addEventListener("click", () => {
    els.iosPopover.hidden = true;
  });
}

function setupFeedbackModal() {
  const resetFeedbackStatus = () => {
    els.feedbackStatus.hidden = true;
    els.feedbackStatus.textContent = "";
    els.feedbackStatus.className = "feedback-status";
  };

  els.feedbackOpenBtn.addEventListener("click", () => {
    resetFeedbackStatus();
    els.feedbackDialog.showModal();
  });

  const closeDialog = () => els.feedbackDialog.close();
  els.feedbackCloseX.addEventListener("click", closeDialog);
  els.feedbackCancelBtn.addEventListener("click", closeDialog);
  els.feedbackDialog.addEventListener("click", (e) => {
    if (e.target === els.feedbackDialog) {
      closeDialog();
    }
  });

  els.feedbackForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const tr = t();
    resetFeedbackStatus();

    if (
      !WEB3FORMS_ACCESS_KEY ||
      WEB3FORMS_ACCESS_KEY === "YOUR_WEB3FORMS_ACCESS_KEY_HERE"
    ) {
      els.feedbackStatus.textContent = tr.feedbackUnconfigured;
      els.feedbackStatus.hidden = false;
      return;
    }

    els.feedbackSubmitBtn.disabled = true;
    els.feedbackSubmitBtn.textContent = tr.feedbackSending;

    try {
      const payload = {
        access_key: WEB3FORMS_ACCESS_KEY,
        subject: "ansitu feedback",
        message: els.feedbackMessage.value.trim(),
        reciter_or_surah: els.feedbackReciter.value.trim(),
      };
      const res = await fetch(WEB3FORMS_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (res.ok && json && json.success) {
        els.feedbackStatus.textContent = tr.feedbackSuccess;
        els.feedbackStatus.classList.add("is-success");
        els.feedbackStatus.hidden = false;
        els.feedbackForm.reset();
      } else {
        throw new Error("SUBMIT_FAILED");
      }
    } catch {
      els.feedbackStatus.textContent = tr.feedbackError;
      els.feedbackStatus.classList.add("is-error");
      els.feedbackStatus.hidden = false;
    } finally {
      els.feedbackSubmitBtn.disabled = false;
      els.feedbackSubmitBtn.textContent = tr.feedbackSend;
    }
  });
}

function bindEvents() {
  els.themeToggle.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });

  els.langToggle.addEventListener("click", () => {
    applyLanguage(state.lang === "ar" ? "en" : "ar");
  });

  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });

  els.changeFileBtn.addEventListener("click", () => els.fileInput.click());

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (file) {
      setSelectedFile(file);
    }
  });

  els.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropzone.classList.add("is-dragover");
  });
  els.dropzone.addEventListener("dragleave", () => {
    els.dropzone.classList.remove("is-dragover");
  });
  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("is-dragover");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
    }
  });

  els.qualitySelect.addEventListener("change", syncLosslessState);

  els.tabStandard.addEventListener("click", () => setMode("standard"));
  els.tabPrayer.addEventListener("click", () => setMode("prayer"));
  [els.tabStandard, els.tabPrayer].forEach((tab) => {
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const nextMode = state.mode === "standard" ? "prayer" : "standard";
        setMode(nextMode);
        (nextMode === "prayer" ? els.tabPrayer : els.tabStandard).focus();
      }
    });
  });

  els.rangeActionRemove.addEventListener("click", () => setRangeAction("remove"));
  els.rangeActionKeep.addEventListener("click", () => setRangeAction("keep"));

  els.addRangeBtn.addEventListener("click", () => {
    state.ranges.push({ id: state.nextRangeId++, start: "", end: "" });
    renderRanges();
  });

  els.startBtn.addEventListener("click", () => {
    if (state.selectedFile) {
      startProcessing(state.selectedFile);
    }
  });

  els.cancelBtn.addEventListener("click", () => {
    cleanupWorkerAndDownloadUrl();
    showView("input");
  });

  els.anotherBtn.addEventListener("click", resetToInput);
  els.guardAnotherBtn.addEventListener("click", resetToInput);
  els.shareBtn.addEventListener("click", handleShareClick);
}

function init() {
  applyTheme(state.theme);
  applyLanguage(state.lang);
  setMode("standard");
  setRangeAction("remove");
  showView("input");
  bindEvents();
  setupPwaInstall();
  setupFeedbackModal();
}

init();
