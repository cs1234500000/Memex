const path = require("path");
const { DEFAULT_SUBSCRIPTIONS } = require("./defaults");
const { readJson, writeJson } = require("../lib/jsonStore");

const SUBSCRIPTIONS_FILE = path.join(__dirname, "../../data/news/subscriptions.json");

function normalizeDomain(domain) {
  const cleaned = String(domain || "")
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\//, "")
    .replace(/^www\./, "")
    .replace(/\/.*$/, "");
  return cleaned;
}

function normalizeSource(value) {
  const source = String(value || "").trim().toLowerCase();
  return source || DEFAULT_SUBSCRIPTIONS.source;
}

function normalizeTwitterHandles(input) {
  return Array.from(
    new Set(
      (input?.twitterHandles || [])
        .map((item) => String(item || "").replace(/^@/, "").trim())
        .filter(Boolean)
    )
  );
}

function normalizeSourcesConfig(input) {
  const incoming = input?.sources?.newsapi || input?.newsapi || {};
  const domains = Array.from(
    new Set((incoming.domains || []).map((item) => normalizeDomain(item)).filter(Boolean))
  );
  const sourceIds = Array.from(
    new Set((incoming.sourceIds || []).map((item) => String(item || "").trim()).filter(Boolean))
  );
  const language = String(incoming.language || DEFAULT_SUBSCRIPTIONS.sources.newsapi.language)
    .trim()
    .toLowerCase();
  const sortBy = String(incoming.sortBy || DEFAULT_SUBSCRIPTIONS.sources.newsapi.sortBy).trim();

  return {
    newsapi: {
      language: language || DEFAULT_SUBSCRIPTIONS.sources.newsapi.language,
      sortBy: sortBy || DEFAULT_SUBSCRIPTIONS.sources.newsapi.sortBy,
      domains,
      sourceIds
    }
  };
}

function normalizeSubscriptions(input) {
  const twitterHandles = normalizeTwitterHandles(input);
  const interests = (input.interests || [])
    .map((interest, idx) => {
      const id = String(interest.id || `interest_${idx + 1}`).trim();
      const label = String(interest.label || id).trim();
      const keywords = Array.from(
        new Set(
          (interest.keywords || [])
            .map((kw) => String(kw || "").trim().toLowerCase())
            .filter(Boolean)
        )
      );
      return { id, label, keywords };
    })
    .filter((interest) => interest.id && interest.keywords.length);

  const maxItemsValue = Number(input.maxItemsPerRun || input.maxTweetsPerRun);
  const maxItemsPerRun = maxItemsValue > 0
    ? Math.min(maxItemsValue, 100)
    : DEFAULT_SUBSCRIPTIONS.maxItemsPerRun;
  const maxTweetsPerRun = Number(input.maxTweetsPerRun || maxItemsPerRun) > 0
    ? Math.min(Number(input.maxTweetsPerRun || maxItemsPerRun), 100)
    : DEFAULT_SUBSCRIPTIONS.maxTweetsPerRun;

  const source = normalizeSource(input.source);
  const sources = normalizeSourcesConfig(input);

  return {
    source,
    twitterHandles: twitterHandles.length ? twitterHandles : DEFAULT_SUBSCRIPTIONS.twitterHandles,
    sources,
    interests: interests.length ? interests : DEFAULT_SUBSCRIPTIONS.interests,
    maxItemsPerRun,
    maxTweetsPerRun
  };
}

async function getSubscriptions() {
  const existing = await readJson(SUBSCRIPTIONS_FILE, null);
  const normalized = normalizeSubscriptions(existing || DEFAULT_SUBSCRIPTIONS);
  if (!existing) await writeJson(SUBSCRIPTIONS_FILE, normalized);
  return normalized;
}

async function saveSubscriptions(input) {
  const normalized = normalizeSubscriptions(input || {});
  await writeJson(SUBSCRIPTIONS_FILE, normalized);
  return normalized;
}

module.exports = {
  SUBSCRIPTIONS_FILE,
  getSubscriptions,
  saveSubscriptions
};

