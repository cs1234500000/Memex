const USERS_TABLE = process.env.SUPABASE_NEWS_USERS_TABLE || "news_users";
const TOPICS_TABLE = process.env.SUPABASE_NEWS_TOPICS_TABLE || "news_user_topics";
const TOPIC_RUNS_TABLE = process.env.SUPABASE_NEWS_TOPIC_RUNS_TABLE || "news_user_topic_runs";
const ITEMS_TABLE = process.env.SUPABASE_NEWS_ITEMS_TABLE || "news_user_knowledge_items";

function isSupabaseConfigured() {
  return Boolean(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY);
}

function sanitizeId(value, fallback) {
  const cleaned = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || fallback;
}

function inferContentType(item) {
  const url = String(item?.url || "").toLowerCase();
  if (/\.(mp4|mov|avi|mkv|webm)(\?|$)/.test(url) || /youtube\.com|youtu\.be|vimeo\.com/.test(url)) {
    return "video";
  }
  if (/\.(mp3|wav|m4a|aac|ogg)(\?|$)/.test(url) || /spotify\.com|soundcloud\.com|podcast/.test(url)) {
    return "audio";
  }
  return "article";
}

function summarizeItemText(text) {
  const clean = String(text || "").trim();
  if (!clean) return "";
  return clean.length > 240 ? `${clean.slice(0, 240)}...` : clean;
}

function encode(value) {
  return encodeURIComponent(String(value || ""));
}

async function supabaseRequest(relativePath, options = {}) {
  const baseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const url = `${baseUrl}/rest/v1/${relativePath}`;

  const response = await fetch(url, {
    method: options.method || "GET",
    headers: {
      apikey: serviceRoleKey,
      authorization: `Bearer ${serviceRoleKey}`,
      ...(options.headers || {})
    },
    body: options.body ? JSON.stringify(options.body) : undefined
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Supabase request failed (${response.status}): ${detail}`);
  }
  return response;
}

async function supabaseReadJson(relativePath) {
  const response = await supabaseRequest(relativePath);
  return response.json();
}

async function supabaseUpsert(table, rows, onConflictColumns) {
  if (!rows.length) return [];
  const response = await supabaseRequest(
    `${table}?on_conflict=${encode(onConflictColumns.join(","))}`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        prefer: "resolution=merge-duplicates,return=representation"
      },
      body: rows
    }
  );
  return response.json();
}

async function supabaseUpsertSingle(table, row, onConflictColumns) {
  const result = await supabaseUpsert(table, [row], onConflictColumns);
  return result[0] || null;
}

function parseCountFromContentRange(value) {
  const text = String(value || "");
  const totalPart = text.split("/")[1] || "";
  const parsed = Number(totalPart);
  return Number.isFinite(parsed) ? parsed : 0;
}

async function supabaseCount(table, filters) {
  const filterQuery = Object.entries(filters)
    .map(([key, value]) => `${key}=eq.${encode(value)}`)
    .join("&");
  const response = await supabaseRequest(`${table}?select=*&${filterQuery}`, {
    method: "HEAD",
    headers: {
      prefer: "count=exact"
    }
  });
  return parseCountFromContentRange(response.headers.get("content-range"));
}

async function ensureUserRow({ userId, userProfile }) {
  const now = new Date().toISOString();
  const existingRows = await supabaseReadJson(
    `${USERS_TABLE}?select=display_name,email&user_id=eq.${encode(userId)}&limit=1`
  );
  const existing = existingRows[0] || {};
  const nextDisplayName = String(userProfile?.name || "").trim() || String(existing.display_name || "");
  const nextEmail = String(userProfile?.email || "").trim().toLowerCase() || String(existing.email || "");

  return supabaseUpsertSingle(
    USERS_TABLE,
    {
      user_id: userId,
      display_name: nextDisplayName,
      email: nextEmail,
      last_run_at: now,
      updated_at: now
    },
    ["user_id"]
  );
}

async function upsertNewsUser({ userId, name = "", email = "" }) {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is required for user registration but is not configured.");
  }
  const safeUserId = sanitizeId(userId, "default");
  const normalizedEmail = String(email || "").trim().toLowerCase();
  const row = await ensureUserRow({
    userId: safeUserId,
    userProfile: {
      name: String(name || "").trim(),
      email: normalizedEmail
    }
  });
  return {
    userId: row?.user_id || safeUserId,
    name: row?.display_name || "",
    email: row?.email || normalizedEmail
  };
}

async function findNewsUser({ userId, email }) {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is required for sign in but is not configured.");
  }

  const safeUserId = sanitizeId(userId, "");
  const normalizedEmail = String(email || "").trim().toLowerCase();
  let query = `${USERS_TABLE}?select=user_id,display_name,email&limit=1`;
  if (safeUserId && normalizedEmail) {
    query += `&user_id=eq.${encode(safeUserId)}&email=eq.${encode(normalizedEmail)}`;
  } else if (safeUserId) {
    query += `&user_id=eq.${encode(safeUserId)}`;
  } else if (normalizedEmail) {
    query += `&email=eq.${encode(normalizedEmail)}`;
  } else {
    return null;
  }

  const rows = await supabaseReadJson(query);
  const row = rows[0];
  if (!row) return null;
  return {
    userId: row.user_id,
    name: row.display_name || "",
    email: row.email || ""
  };
}

function normalizeKnowledgeItem({ topicId, source, runId, generatedAt, scoredItem }) {
  const item = scoredItem.item || {};
  const publishedAt = item.createdAt || generatedAt;
  return {
    item_id: String(item.id || `${topicId}_${source}_${publishedAt}`),
    topic_id: topicId,
    source: String(item.source || source || "unknown"),
    content_type: inferContentType(item),
    url: String(item.url || ""),
    author: String(item.author || ""),
    origin: String(item.handle || ""),
    title: summarizeItemText(item.text),
    text_content: String(item.text || ""),
    published_at: publishedAt,
    discovered_at: generatedAt,
    run_id: runId,
    score: Number(scoredItem.score || 0),
    item_summary: String(scoredItem.summary || summarizeItemText(item.text)),
    item_insights: Array.isArray(scoredItem.insights) ? scoredItem.insights : [],
    metrics: {
      likeCount: Number(item?.metrics?.likeCount || 0),
      repostCount: Number(item?.metrics?.repostCount || 0),
      replyCount: Number(item?.metrics?.replyCount || 0)
    }
  };
}

async function upsertTopicKnowledge({ userId, source, reportId, generatedAt, topic }) {
  const safeTopicId = sanitizeId(topic?.topicId, "general");
  const existingTopic = await supabaseReadJson(
    `${TOPICS_TABLE}?select=first_merged_at&user_id=eq.${encode(userId)}&topic_id=eq.${encode(
      safeTopicId
    )}&limit=1`
  );
  const firstMergedAt = existingTopic[0]?.first_merged_at || generatedAt;

  // Ensure parent topic row exists before writing knowledge items (FK safety).
  await supabaseUpsertSingle(
    TOPICS_TABLE,
    {
      user_id: userId,
      topic_id: safeTopicId,
      topic_label: topic?.topicLabel || safeTopicId,
      total_knowledge_items: 0,
      total_runs_merged: 0,
      first_merged_at: firstMergedAt,
      last_merged_at: generatedAt,
      latest_run_id: reportId,
      latest_source: source,
      latest_summary: String(topic?.summary || ""),
      latest_insights: Array.isArray(topic?.insights) ? topic.insights : [],
      updated_at: generatedAt
    },
    ["user_id", "topic_id"]
  );

  const knowledgeItems = (topic?.allItems || [])
    .map((entry) =>
      normalizeKnowledgeItem({
        topicId: safeTopicId,
        source,
        runId: reportId,
        generatedAt,
        scoredItem: entry
      })
    )
    .filter((item) => item.text_content || item.url)
    .map((item) => ({
      ...item,
      user_id: userId,
      updated_at: generatedAt
    }));

  await supabaseUpsert(ITEMS_TABLE, knowledgeItems, ["user_id", "topic_id", "item_id"]);

  await supabaseUpsertSingle(
    TOPIC_RUNS_TABLE,
    {
      user_id: userId,
      topic_id: safeTopicId,
      run_id: reportId,
      generated_at: generatedAt,
      source,
      matched_count: Number(topic?.matchedCount || 0),
      summary: String(topic?.summary || ""),
      insights: Array.isArray(topic?.insights) ? topic.insights : []
    },
    ["user_id", "topic_id", "run_id"]
  );

  const totalKnowledgeItems = await supabaseCount(ITEMS_TABLE, {
    user_id: userId,
    topic_id: safeTopicId
  });
  const totalRunsMerged = await supabaseCount(TOPIC_RUNS_TABLE, {
    user_id: userId,
    topic_id: safeTopicId
  });

  await supabaseUpsertSingle(
    TOPICS_TABLE,
    {
      user_id: userId,
      topic_id: safeTopicId,
      topic_label: topic?.topicLabel || safeTopicId,
      total_knowledge_items: totalKnowledgeItems,
      total_runs_merged: totalRunsMerged,
      first_merged_at: firstMergedAt,
      last_merged_at: generatedAt,
      latest_run_id: reportId,
      latest_source: source,
      latest_summary: String(topic?.summary || ""),
      latest_insights: Array.isArray(topic?.insights) ? topic.insights : [],
      updated_at: generatedAt
    },
    ["user_id", "topic_id"]
  );

  return {
    userId,
    topicId: safeTopicId,
    totalKnowledgeItems
  };
}

async function mergeUserKnowledgeFromReport({ userId, report, userProfile }) {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is required for news knowledge storage but is not configured.");
  }
  const safeUserId = sanitizeId(userId, "default");
  await ensureUserRow({
    userId: safeUserId,
    userProfile
  });

  const mergedTopics = [];
  for (const topic of report.topics || []) {
    const merged = await upsertTopicKnowledge({
      userId: safeUserId,
      source: report.source,
      reportId: report.id,
      generatedAt: report.generatedAt,
      topic
    });
    mergedTopics.push(merged);
  }
  return {
    userId: safeUserId,
    mergedTopics
  };
}

function mapTopicDocument(topicRow, summaries, items) {
  return {
    schemaVersion: 1,
    userId: topicRow.user_id,
    topic: {
      id: topicRow.topic_id,
      label: topicRow.topic_label || topicRow.topic_id
    },
    stats: {
      totalKnowledgeItems: Number(topicRow.total_knowledge_items || 0),
      totalRunsMerged: Number(topicRow.total_runs_merged || 0),
      firstMergedAt: topicRow.first_merged_at || null,
      lastMergedAt: topicRow.last_merged_at || null,
      latestRunId: topicRow.latest_run_id || null,
      latestSource: topicRow.latest_source || null
    },
    latest: {
      summary: topicRow.latest_summary || "",
      insights: Array.isArray(topicRow.latest_insights) ? topicRow.latest_insights : []
    },
    summaries: summaries.map((row) => ({
      runId: row.run_id,
      generatedAt: row.generated_at,
      source: row.source,
      matchedCount: Number(row.matched_count || 0),
      summary: row.summary || "",
      insights: Array.isArray(row.insights) ? row.insights : []
    })),
    knowledgeItems: items.map((row) => ({
      id: row.item_id,
      topicId: row.topic_id,
      source: row.source || "",
      contentType: row.content_type || "article",
      url: row.url || "",
      author: row.author || "",
      origin: row.origin || "",
      title: row.title || "",
      text: row.text_content || "",
      publishedAt: row.published_at,
      discoveredAt: row.discovered_at,
      runId: row.run_id || "",
      score: Number(row.score || 0),
      summary: row.item_summary || summarizeItemText(row.text_content || ""),
      insights: Array.isArray(row.item_insights) ? row.item_insights : [],
      metrics: row.metrics || { likeCount: 0, repostCount: 0, replyCount: 0 }
    }))
  };
}

async function fetchTopicDocument(userId, topicId) {
  const topicRows = await supabaseReadJson(
    `${TOPICS_TABLE}?select=*&user_id=eq.${encode(userId)}&topic_id=eq.${encode(topicId)}&limit=1`
  );
  const topicRow = topicRows[0];
  if (!topicRow) return null;

  const [summaryRows, itemRows] = await Promise.all([
    supabaseReadJson(
      `${TOPIC_RUNS_TABLE}?select=*&user_id=eq.${encode(userId)}&topic_id=eq.${encode(
        topicId
      )}&order=generated_at.desc&limit=100`
    ),
    supabaseReadJson(
      `${ITEMS_TABLE}?select=*&user_id=eq.${encode(userId)}&topic_id=eq.${encode(
        topicId
      )}&order=published_at.desc`
    )
  ]);

  return mapTopicDocument(topicRow, summaryRows, itemRows);
}

async function getUserTopicKnowledge({ userId, topicId }) {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is required for news knowledge retrieval but is not configured.");
  }

  const safeUserId = sanitizeId(userId, "default");
  if (topicId) {
    const safeTopicId = sanitizeId(topicId, "general");
    return fetchTopicDocument(safeUserId, safeTopicId);
  }

  const topicRows = await supabaseReadJson(
    `${TOPICS_TABLE}?select=topic_id&user_id=eq.${encode(safeUserId)}&order=topic_id.asc`
  );
  const docs = await Promise.all(
    topicRows.map((row) => fetchTopicDocument(safeUserId, row.topic_id))
  );
  return docs.filter(Boolean);
}

module.exports = {
  mergeUserKnowledgeFromReport,
  getUserTopicKnowledge,
  upsertNewsUser,
  findNewsUser
};
