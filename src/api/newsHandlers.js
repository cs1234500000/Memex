const { runNewsPipeline, getLatestReport } = require("../news/pipeline");
const { getSubscriptions, saveSubscriptions } = require("../news/subscriptions");
const { getUserTopicKnowledge, upsertNewsUser, findNewsUser } = require("../news/knowledgeStore");

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8"
  });
  res.end(JSON.stringify(payload));
}

async function readBody(req) {
  let body = "";
  for await (const chunk of req) {
    body += chunk;
    if (body.length > 1_000_000) throw new Error("Payload too large");
  }
  return body;
}

async function handleGetLatestNews(_req, res) {
  const report = await getLatestReport();
  if (!report) return sendJson(res, 404, { error: "No news report found yet." });
  return sendJson(res, 200, report);
}

function getQueryParam(req, key) {
  if (!req?.url) return "";
  const url = new URL(req.url, "http://localhost");
  return String(url.searchParams.get(key) || "");
}

function getUserIdFromRequest(req, bodyPayload) {
  const headerUserId = String(req.headers["x-user-id"] || "").trim();
  const queryUserId = getQueryParam(req, "userId").trim();
  const bodyUserId = String(bodyPayload?.userId || "").trim();
  return headerUserId || queryUserId || bodyUserId || "default";
}

function getUserProfileFromRequest(req, bodyPayload) {
  const headerName = String(req.headers["x-user-name"] || "").trim();
  const headerEmail = String(req.headers["x-user-email"] || "").trim();
  const bodyName = String(bodyPayload?.user?.name || bodyPayload?.userName || "").trim();
  const bodyEmail = String(bodyPayload?.user?.email || bodyPayload?.userEmail || "").trim();
  return {
    name: bodyName || headerName || "",
    email: bodyEmail || headerEmail || ""
  };
}

async function handleRunNews(req, res) {
  let payload = {};
  try {
    const body = await readBody(req);
    payload = body ? JSON.parse(body) : {};
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body." });
  }

  const userId = getUserIdFromRequest(req, payload);
  const userProfile = getUserProfileFromRequest(req, payload);
  const report = await runNewsPipeline({ userId, userProfile });
  return sendJson(res, 200, report);
}

async function handleGetSubscriptions(_req, res) {
  const subscriptions = await getSubscriptions();
  return sendJson(res, 200, subscriptions);
}

async function handleUpdateSubscriptions(req, res) {
  let payload;
  try {
    const body = await readBody(req);
    payload = JSON.parse(body || "{}");
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body." });
  }

  const updated = await saveSubscriptions(payload);
  return sendJson(res, 200, updated);
}

async function handleGetKnowledge(req, res) {
  const userId = getUserIdFromRequest(req, null);
  const topicId = getQueryParam(req, "topicId").trim();
  const data = await getUserTopicKnowledge({
    userId,
    topicId
  });

  if (topicId && !data) {
    return sendJson(res, 404, { error: "No knowledge found for this topic." });
  }
  return sendJson(res, 200, {
    userId,
    topicId: topicId || null,
    data
  });
}

async function handleNewsUserSignup(req, res) {
  let payload;
  try {
    const body = await readBody(req);
    payload = JSON.parse(body || "{}");
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body." });
  }

  const userId = String(payload.userId || "").trim();
  const name = String(payload.name || "").trim();
  const email = String(payload.email || "").trim().toLowerCase();
  if (!userId) return sendJson(res, 400, { error: "userId is required." });

  const user = await upsertNewsUser({ userId, name, email });
  return sendJson(res, 200, { ok: true, user });
}

async function handleNewsUserSignin(req, res) {
  let payload;
  try {
    const body = await readBody(req);
    payload = JSON.parse(body || "{}");
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body." });
  }

  const userId = String(payload.userId || "").trim();
  const email = String(payload.email || "").trim().toLowerCase();
  if (!userId || !email) {
    return sendJson(res, 400, { error: "Both userId and email are required." });
  }
  const user = await findNewsUser({ userId, email });
  if (!user) return sendJson(res, 404, { error: "User not found." });
  return sendJson(res, 200, { ok: true, user });
}

module.exports = {
  handleGetLatestNews,
  handleRunNews,
  handleGetSubscriptions,
  handleUpdateSubscriptions,
  handleGetKnowledge,
  handleNewsUserSignup,
  handleNewsUserSignin
};

