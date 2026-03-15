const http = require("http");
const fs = require("fs/promises");
const path = require("path");
const crypto = require("crypto");
const {
  handleGetLatestNews,
  handleRunNews,
  handleGetSubscriptions,
  handleUpdateSubscriptions,
  handleGetKnowledge,
  handleNewsUserSignup,
  handleNewsUserSignin
} = require("./src/api/newsHandlers");

const PORT = process.env.PORT || 3000;
const ROOT_DIR = __dirname;
const DATA_DIR = path.join(ROOT_DIR, "data");
const SIGNUPS_FILE = path.join(DATA_DIR, "signups.json");
const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const SUPABASE_SIGNUPS_TABLE = process.env.SUPABASE_SIGNUPS_TABLE || "waitlist_signups";
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const WAITLIST_FROM_EMAIL = process.env.WAITLIST_FROM_EMAIL || "hello@withmemex.com";
const WAITLIST_CONFIRM_SUBJECT = process.env.WAITLIST_CONFIRM_SUBJECT || "You're on the Memex waitlist";
const WAITLIST_SITE_URL = process.env.WAITLIST_SITE_URL || "https://withmemex.com";

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

function isAdminAuthorized(req) {
  const expectedToken = process.env.WAITLIST_ADMIN_TOKEN || "";
  if (!expectedToken) return false;
  const headerToken = req.headers["x-admin-token"] || "";
  const url = new URL(req.url, "http://localhost");
  const queryToken = url.searchParams.get("token") || "";
  return headerToken === expectedToken || queryToken === expectedToken;
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

function isSupabaseConfigured() {
  return Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);
}

async function trySendConfirmationEmail(email) {
  try {
    await sendWaitlistConfirmationEmail(email);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error("Confirmation email failed:", error.message || error);
  }
}

async function supabaseFindSignupByEmail(email) {
  const url = `${SUPABASE_URL}/rest/v1/${SUPABASE_SIGNUPS_TABLE}?select=id&email=eq.${encodeURIComponent(email)}&limit=1`;
  const response = await fetch(url, {
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
    },
  });
  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Supabase read failed (${response.status}): ${details}`);
  }
  const rows = await response.json();
  return Array.isArray(rows) && rows.length > 0;
}

async function supabaseInsertSignup(signup) {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${SUPABASE_SIGNUPS_TABLE}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      prefer: "return=representation",
    },
    body: JSON.stringify({
      id: signup.id,
      name: signup.name,
      email: signup.email,
      interest: signup.interest,
      created_at: signup.createdAt,
    }),
  });
  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Supabase insert failed (${response.status}): ${details}`);
  }
}

async function supabaseListSignups() {
  const response = await fetch(
    `${SUPABASE_URL}/rest/v1/${SUPABASE_SIGNUPS_TABLE}?select=id,name,email,interest,created_at&order=created_at.desc&limit=1000`,
    {
      headers: {
        apikey: SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      },
    }
  );
  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Supabase list failed (${response.status}): ${details}`);
  }
  const rows = await response.json();
  return (rows || []).map((row) => ({
    id: row.id,
    name: row.name || "",
    email: row.email,
    interest: row.interest || "",
    createdAt: row.created_at,
  }));
}

async function sendWaitlistConfirmationEmail(recipientEmail) {
  if (!RESEND_API_KEY) {
    // eslint-disable-next-line no-console
    console.warn("RESEND_API_KEY is not configured; skipping confirmation email.");
    return;
  }

  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${RESEND_API_KEY}`,
    },
    body: JSON.stringify({
      from: WAITLIST_FROM_EMAIL,
      to: [recipientEmail],
      subject: WAITLIST_CONFIRM_SUBJECT,
      html: `
        <div style="font-family:Arial,sans-serif;line-height:1.6;color:#1a1814">
          <p>You're in - thanks for joining the Memex waitlist.</p>
          <p>We'll reach out from <b>hello@withmemex.com</b> as we open access.</p>
          <p>Website: <a href="${WAITLIST_SITE_URL}">${WAITLIST_SITE_URL}</a></p>
        </div>
      `,
    }),
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Failed to send confirmation email (${response.status}): ${details}`);
  }
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

  const signup = {
    id: crypto.randomUUID(),
    name,
    email,
    interest,
    createdAt: new Date().toISOString(),
  };

  if (isSupabaseConfigured()) {
    const duplicate = await supabaseFindSignupByEmail(email);
    if (duplicate) {
      return sendJson(res, 409, { error: "This email is already registered." });
    }
    await supabaseInsertSignup(signup);
    await trySendConfirmationEmail(signup.email);
    return sendJson(res, 201, { ok: true });
  }

  await ensureDataFile();
  const signups = await readJsonFile(SIGNUPS_FILE);
  const duplicate = signups.some((entry) => entry.email === email);
  if (duplicate) {
    return sendJson(res, 409, { error: "This email is already registered." });
  }
  signups.push(signup);
  await writeJsonFile(SIGNUPS_FILE, signups);
  await trySendConfirmationEmail(signup.email);
  return sendJson(res, 201, { ok: true });
}

async function handleSignupList(req, res) {
  if (!isAdminAuthorized(req)) {
    return sendJson(res, 401, { error: "Unauthorized." });
  }
  if (isSupabaseConfigured()) {
    const signups = await supabaseListSignups();
    return sendJson(res, 200, {
      count: signups.length,
      items: signups,
    });
  }
  await ensureDataFile();
  const signups = await readJsonFile(SIGNUPS_FILE);
  signups.sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")));
  return sendJson(res, 200, {
    count: signups.length,
    items: signups,
  });
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

  if (req.method === "GET" && req.url.startsWith("/api/signup/list")) {
    try {
      await handleSignupList(req, res);
    } catch {
      sendJson(res, 500, { error: "Internal server error." });
    }
    return;
  }

  if (req.method === "GET" && req.url === "/api/news/latest") {
    try {
      await handleGetLatestNews(req, res);
    } catch {
      sendJson(res, 500, { error: "Failed to load latest news report." });
    }
    return;
  }

  if (req.method === "POST" && req.url.startsWith("/api/news/run")) {
    try {
      await handleRunNews(req, res);
    } catch (error) {
      sendJson(res, 500, { error: error.message || "Failed to run news pipeline." });
    }
    return;
  }

  if (req.method === "GET" && req.url.startsWith("/api/news/knowledge")) {
    try {
      await handleGetKnowledge(req, res);
    } catch {
      sendJson(res, 500, { error: "Failed to load user knowledge." });
    }
    return;
  }

  if (req.method === "POST" && req.url === "/api/news/users/signup") {
    try {
      await handleNewsUserSignup(req, res);
    } catch (error) {
      sendJson(res, 500, { error: error.message || "Failed to sign up user." });
    }
    return;
  }

  if (req.method === "POST" && req.url === "/api/news/users/signin") {
    try {
      await handleNewsUserSignin(req, res);
    } catch (error) {
      sendJson(res, 500, { error: error.message || "Failed to sign in user." });
    }
    return;
  }

  if (req.method === "GET" && req.url === "/api/news/subscriptions") {
    try {
      await handleGetSubscriptions(req, res);
    } catch {
      sendJson(res, 500, { error: "Failed to load subscriptions." });
    }
    return;
  }

  if (req.method === "PUT" && req.url === "/api/news/subscriptions") {
    try {
      await handleUpdateSubscriptions(req, res);
    } catch {
      sendJson(res, 500, { error: "Failed to update subscriptions." });
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
