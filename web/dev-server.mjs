/**
 * Lightweight zero-dependency static server for local development & testing of `web/`.
 * Sends COOP/COEP headers matching `web/_headers` so `crossOriginIsolated === true`
 * and multi-threaded `ffmpeg.wasm` works locally out of the box.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB_ROOT = path.dirname(fileURLToPath(import.meta.url));

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".wasm": "application/wasm",
};

function applyIsolationHeaders(res) {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  res.setHeader("Cache-Control", "no-cache");
}

export function createStaticServer(rootDir = WEB_ROOT) {
  const resolvedRoot = path.resolve(rootDir);

  return http.createServer((req, res) => {
    applyIsolationHeaders(res);

    const method = req.method || "GET";
    if (method !== "GET" && method !== "HEAD") {
      res.writeHead(405, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Method Not Allowed");
      return;
    }

    const rawUrl = (req.url || "/").split("?")[0];
    let decodedPath;
    try {
      decodedPath = decodeURIComponent(rawUrl);
    } catch {
      res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Bad Request");
      return;
    }

    const relPath = decodedPath === "/" ? "index.html" : decodedPath.replace(/^\/+/, "");
    const targetPath = path.resolve(resolvedRoot, relPath);

    // Block path traversal outside `web/`
    if (
      targetPath !== resolvedRoot &&
      !targetPath.startsWith(resolvedRoot + path.sep)
    ) {
      res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Forbidden");
      return;
    }

    let finalPath = targetPath;
    try {
      const stat = fs.statSync(finalPath);
      if (stat.isDirectory()) {
        finalPath = path.join(finalPath, "index.html");
      }
      const fileStat = fs.statSync(finalPath);
      const ext = path.extname(finalPath).toLowerCase();
      const mime = MIME_TYPES[ext] || "application/octet-stream";

      res.writeHead(200, {
        "Content-Type": mime,
        "Content-Length": String(fileStat.size),
      });

      if (method === "HEAD") {
        res.end();
        return;
      }

      const stream = fs.createReadStream(finalPath);
      stream.on("error", () => {
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
        }
        res.end("Internal Server Error");
      });
      stream.pipe(res);
    } catch {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      if (method === "HEAD") {
        res.end();
      } else {
        res.end("Not Found");
      }
    }
  });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const port = Number(process.env.PORT || 4173);
  const host = process.env.HOST || "127.0.0.1";
  const server = createStaticServer(WEB_ROOT);
  server.listen(port, host, () => {
    console.log(`ansitu web dev server running at http://localhost:${port}`);
    console.log("COOP/COEP headers enabled (crossOriginIsolated = true)");
  });
}
