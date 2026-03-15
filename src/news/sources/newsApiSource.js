const crypto = require("crypto");

function buildQueryFromInterest(interest) {
  const keywords = (interest?.keywords || []).slice(0, 8).filter(Boolean);
  if (!keywords.length) return "";
  return keywords.map((kw) => `"${kw}"`).join(" OR ");
}

function dedupeByUrl(articles) {
  const map = new Map();
  for (const article of articles) {
    const key = article.url || article.id;
    if (!map.has(key)) map.set(key, article);
  }
  return [...map.values()];
}

function mapArticleToItem(article) {
  const url = String(article?.url || "").trim();
  const idSeed = url || `${article?.title || ""}-${article?.publishedAt || ""}`;
  const id = crypto.createHash("sha1").update(idSeed).digest("hex");
  const title = String(article?.title || "").trim();
  const description = String(article?.description || "").trim();
  const text = [title, description].filter(Boolean).join(". ");
  if (!text) return null;

  const sourceName = article?.source?.name || "newsapi";
  return {
    id,
    text,
    author: String(article?.author || sourceName),
    handle: String(sourceName).toLowerCase().replace(/\s+/g, "_"),
    createdAt: article?.publishedAt || new Date().toISOString(),
    metrics: {
      likeCount: 0,
      repostCount: 0,
      replyCount: 0
    },
    url,
    source: "newsapi"
  };
}

async function fetchArticlesForInterest({ interest, perInterestMax, config, apiKey }) {
  const query = buildQueryFromInterest(interest);
  if (!query) return [];

  const url = new URL("https://newsapi.org/v2/everything");
  url.searchParams.set("q", query);
  url.searchParams.set("language", config.language || "en");
  url.searchParams.set("sortBy", config.sortBy || "publishedAt");
  url.searchParams.set("pageSize", String(Math.min(perInterestMax, 100)));
  url.searchParams.set("searchIn", "title,description,content");

  if (config.domains?.length) {
    url.searchParams.set("domains", config.domains.join(","));
  }
  if (config.sourceIds?.length) {
    url.searchParams.set("sources", config.sourceIds.join(","));
  }

  const response = await fetch(url, {
    headers: {
      "X-Api-Key": apiKey
    }
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`NewsAPI failed (${response.status}): ${detail}`);
  }

  const payload = await response.json();
  return (payload.articles || []).map((article) => mapArticleToItem(article)).filter(Boolean);
}

function compactError(error) {
  const text = String(error?.message || error || "unknown error").replace(/\s+/g, " ").trim();
  return text.length > 220 ? `${text.slice(0, 220)}...` : text;
}

async function fetchNewsApiItems({ subscriptions, maxItems }) {
  const apiKey = String(process.env.NEWSAPI_API_KEY || "").trim();
  if (!apiKey) {
    throw new Error("NEWSAPI_API_KEY is not set.");
  }

  const config = subscriptions?.sources?.newsapi || {};
  const interests = subscriptions?.interests || [];
  if (!interests.length) return { items: [], warnings: [] };

  const perInterestMax = Math.max(5, Math.ceil(maxItems / interests.length));
  const allItems = [];
  const warnings = [];

  for (const interest of interests) {
    try {
      const items = await fetchArticlesForInterest({
        interest,
        perInterestMax,
        config,
        apiKey
      });
      allItems.push(...items);
    } catch (error) {
      warnings.push(`newsapi fetch failed for interest "${interest.label}": ${compactError(error)}`);
    }
  }

  const items = dedupeByUrl(allItems)
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, maxItems);

  return {
    items,
    warnings
  };
}

module.exports = {
  fetchNewsApiItems
};
