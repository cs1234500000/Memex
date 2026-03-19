/**
 * Local development static file server.
 * Serves the web/ directory on http://localhost:PORT
 *
 * All API routes (/api/*) are now handled by the Python FastAPI backend.
 * Start it with:  uvicorn memex.api.main:app --reload --port 8000
 *
 * In production, Cloudflare Pages serves web/ directly and
 * web/functions/api/ handles the waitlist endpoints at the edge.
 */

const http = require("http");
const fs = require("fs/promises");
const path = require("path");

const PORT = process.env.PORT || 3000;
const WEB_DIR = path.join(__dirname, "web");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const server = http.createServer(async (req, res) => {
  if (!req.url || req.method !== "GET") {
    res.writeHead(405);
    res.end("Method not allowed");
    return;
  }

  const requestPath = req.url === "/" ? "/index.html" : req.url;
  const sanitized = path.normalize(requestPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(WEB_DIR, sanitized);

  if (!filePath.startsWith(WEB_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  try {
    const stats = await fs.stat(filePath);
    if (stats.isDirectory()) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";
    const fileData = await fs.readFile(filePath);
    res.writeHead(200, { "Content-Type": contentType });
    res.end(fileData);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});

server.listen(PORT, () => {
  console.log(`Memex web  →  http://localhost:${PORT}`);
  console.log(`Python API →  http://localhost:7001  (uvicorn memex.api.main:app --reload --port 7001)`);
});
