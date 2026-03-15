function countKeywordMatches(text, keywords) {
  const lower = text.toLowerCase();
  return keywords.reduce((score, keyword) => {
    if (!keyword) return score;
    return lower.includes(keyword.toLowerCase()) ? score + 1 : score;
  }, 0);
}

function engagementScore(metrics) {
  const like = metrics?.likeCount || 0;
  const repost = metrics?.repostCount || 0;
  const reply = metrics?.replyCount || 0;
  return like + repost * 2 + reply * 1.5;
}

let embedderPromise = null;

function parseEnvNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < a.length; i += 1) {
    const aValue = Number(a[i]) || 0;
    const bValue = Number(b[i]) || 0;
    dot += aValue * bValue;
    aNorm += aValue * aValue;
    bNorm += bValue * bValue;
  }
  if (!aNorm || !bNorm) return 0;
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

function legacyKeywordMatches(items, interests) {
  const matched = [];

  for (const interest of interests) {
    for (const item of items) {
      const keywordScore = countKeywordMatches(item.text || "", interest.keywords || []);
      if (keywordScore <= 0) continue;

      const score = keywordScore * 10 + Math.log1p(engagementScore(item.metrics));
      matched.push({
        topicId: interest.id,
        topicLabel: interest.label,
        score,
        item
      });
    }
  }

  return matched.sort((a, b) => b.score - a.score);
}

async function getLocalEmbedder() {
  if (embedderPromise) return embedderPromise;
  const fastembed = require("fastembed");
  const rawModelName = String(
    process.env.NEWS_LOCAL_EMBEDDING_MODEL || process.env.EMBEDDING_MODEL || "fast-all-MiniLM-L6-v2"
  ).trim();
  const modelAliases = {
    "all-minilm-l6-v2": fastembed.EmbeddingModel.AllMiniLML6V2,
    "fast-all-minilm-l6-v2": fastembed.EmbeddingModel.AllMiniLML6V2
  };
  const modelName = modelAliases[rawModelName.toLowerCase()] || rawModelName;
  const knownModel = Object.values(fastembed.EmbeddingModel).find((value) => value === modelName);
  const model = knownModel || fastembed.EmbeddingModel.AllMiniLML6V2;

  embedderPromise = fastembed.FlagEmbedding.init({
    model,
    cacheDir: process.env.NEWS_LOCAL_EMBEDDING_CACHE_DIR || "local_cache",
    showDownloadProgress: false
  });
  return embedderPromise;
}

function buildTopicQuery(interest) {
  const label = String(interest?.label || "").trim();
  const keywords = (interest?.keywords || []).map((value) => String(value || "").trim()).filter(Boolean);
  return [label, keywords.join(", ")].filter(Boolean).join(". ");
}

async function embedPassages(embedder, texts, batchSize) {
  const vectors = [];
  for await (const batch of embedder.passageEmbed(texts, batchSize)) {
    vectors.push(...batch);
  }
  return vectors;
}

async function semanticHybridMatches(items, interests) {
  const texts = items.map((item) => String(item?.text || ""));
  if (!texts.length || !interests.length) return [];

  const embedder = await getLocalEmbedder();
  const itemEmbeddings = await embedPassages(
    embedder,
    texts,
    parseEnvNumber(process.env.NEWS_EMBEDDING_BATCH_SIZE, 32)
  );
  const topicEmbeddings = await Promise.all(interests.map((interest) => embedder.queryEmbed(buildTopicQuery(interest))));

  const minSimilarity = parseEnvNumber(process.env.NEWS_SEMANTIC_MIN_SIMILARITY, 0.2);
  const semanticWeight = parseEnvNumber(process.env.NEWS_SEMANTIC_WEIGHT, 20);
  const keywordWeight = parseEnvNumber(process.env.NEWS_KEYWORD_WEIGHT, 8);
  const matched = [];

  for (let topicIndex = 0; topicIndex < interests.length; topicIndex += 1) {
    const interest = interests[topicIndex];
    const topicEmbedding = topicEmbeddings[topicIndex];

    for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
      const item = items[itemIndex];
      const keywordScore = countKeywordMatches(item.text || "", interest.keywords || []);
      const similarity = cosineSimilarity(itemEmbeddings[itemIndex], topicEmbedding);
      if (keywordScore <= 0 && similarity < minSimilarity) continue;

      const score =
        similarity * semanticWeight +
        keywordScore * keywordWeight +
        Math.log1p(engagementScore(item.metrics));

      matched.push({
        topicId: interest.id,
        topicLabel: interest.label,
        score,
        item
      });
    }
  }

  return matched.sort((a, b) => b.score - a.score);
}

async function matchItemsByInterest(items, interests) {
  const localEmbeddingsEnabled = String(process.env.NEWS_USE_LOCAL_EMBEDDINGS || "true")
    .trim()
    .toLowerCase() !== "false";

  if (!localEmbeddingsEnabled) {
    return legacyKeywordMatches(items, interests);
  }

  try {
    return await semanticHybridMatches(items, interests);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn("Falling back to keyword relevance:", error?.message || error);
    return legacyKeywordMatches(items, interests);
  }
}

function groupMatchesByTopic(matches) {
  return matches.reduce((acc, match) => {
    const bucket = acc[match.topicId] || {
      topicId: match.topicId,
      topicLabel: match.topicLabel,
      items: []
    };
    bucket.items.push(match);
    acc[match.topicId] = bucket;
    return acc;
  }, {});
}

module.exports = {
  matchItemsByInterest,
  groupMatchesByTopic,
  engagementScore
};

