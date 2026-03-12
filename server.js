const http = require("http");
const fs = require("fs/promises");
const path = require("path");
const crypto = require("crypto");

const PORT = process.env.PORT || 3000;
const ROOT_DIR = __dirname;
const DATA_DIR = path.join(ROOT_DIR, "data");
const SIGNUPS_FILE = path.join(DATA_DIR, "signups.json");

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

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
  });
  res.end(JSON.stringify(payload));
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

async function ensureDataFile() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  try {
    await fs.access(SIGNUPS_FILE);
  } catch {
    await fs.writeFile(SIGNUPS_FILE, "[]", "utf8");
  }
}

async function readJsonFile(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

async function writeJsonFile(filePath, data) {
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf8");
}

async function handleSignup(req, res) {
  let body = "";
  for await (const chunk of req) {
    body += chunk;
    if (body.length > 1_000_000) {
      return sendJson(res, 413, { error: "Payload too large." });
    }
  }

  let parsed;
  try {
    parsed = JSON.parse(body);
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body." });
  }

  const name = (parsed.name || "").toString().trim();
  const email = (parsed.email || "").toString().trim().toLowerCase();
  const interest = (parsed.interest || "").toString().trim();

  if (!email || !isValidEmail(email)) {
    return sendJson(res, 400, { error: "Valid email is required." });
  }

  await ensureDataFile();
  const signups = await readJsonFile(SIGNUPS_FILE);
  const duplicate = signups.some((entry) => entry.email === email);

  if (duplicate) {
    return sendJson(res, 409, { error: "This email is already registered." });
  }

  signups.push({
    id: crypto.randomUUID(),
    name,
    email,
    interest,
    createdAt: new Date().toISOString(),
  });

  await writeJsonFile(SIGNUPS_FILE, signups);
  return sendJson(res, 201, { ok: true });
}

async function serveStatic(req, res) {
  const requestPath = req.url === "/" ? "/index.html" : req.url || "/index.html";
  const sanitizedPath = path.normalize(requestPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(ROOT_DIR, sanitizedPath);

  if (!filePath.startsWith(ROOT_DIR)) {
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
}

const server = http.createServer(async (req, res) => {
  if (!req.url || !req.method) {
    res.writeHead(400);
    res.end("Bad request");
    return;
  }

  if (req.method === "POST" && req.url === "/api/signup") {
    try {
      await handleSignup(req, res);
    } catch {
      sendJson(res, 500, { error: "Internal server error." });
    }
    return;
  }

  if (req.method === "GET") {
    await serveStatic(req, res);
    return;
  }

  res.writeHead(405);
  res.end("Method not allowed");
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Memex site running on http://localhost:${PORT}`);
});
