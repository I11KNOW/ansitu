/**
 * Offline Service Worker for ansitu PWA.
 * Preserves COOP/COEP headers on cached responses and caches ffmpeg.wasm cores
 * fetched from jsDelivr for offline reuse.
 */

const CACHE_NAME = "ansitu-shell-v2";
const FFMPEG_CACHE = "ansitu-ffmpeg-v1";

const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./worker.js",
  "./config.js",
  "./i18n.js",
  "./engine/vad.js",
  "./engine/splicer.js",
  "./manifest.json",
  "./icons/crescent.svg",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  const valid = new Set([CACHE_NAME, FFMPEG_CACHE]);
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.map((key) => (valid.has(key) ? null : caches.delete(key)))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") {
    return;
  }

  let url;
  try {
    url = new URL(request.url);
  } catch {
    return;
  }

  if (url.hostname === "api.web3forms.com") {
    return;
  }

  // Cache-first for @ffmpeg/core and @ffmpeg/core-mt CDN bundles
  if (
    url.origin === "https://cdn.jsdelivr.net" &&
    url.pathname.startsWith("/npm/@ffmpeg/")
  ) {
    event.respondWith(
      caches.open(FFMPEG_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        if (cached) {
          return cached;
        }
        const response = await fetch(request);
        if (response && response.ok) {
          cache.put(request, response.clone());
        }
        return response;
      })
    );
    return;
  }

  // Network-first for same-origin app shell (keeps updates fresh while preserving offline support)
  if (url.origin === self.location.origin) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.ok && response.type === "basic") {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(request);
          if (cached) {
            return cached;
          }
          if (request.mode === "navigate") {
            const fallback = await caches.match("./index.html");
            if (fallback) {
              return fallback;
            }
          }
          return new Response("Offline", { status: 503, statusText: "Offline" });
        })
    );
  }
});
