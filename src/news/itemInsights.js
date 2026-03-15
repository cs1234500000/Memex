function splitSentences(text) {
  return String(text || "")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function summarizeText(text) {
  const sentences = splitSentences(text);
  const base = sentences[0] || String(text || "").trim();
  if (!base) return "No concise summary available.";
  return base.length > 220 ? `${base.slice(0, 220)}...` : base;
}

function matchedKeywords(text, keywords) {
  const lower = String(text || "").toLowerCase();
  return (keywords || []).filter((kw) => lower.includes(String(kw || "").toLowerCase()));
}

function buildInsights(item, { topicLabel, keywords, score }) {
  const matched = matchedKeywords(item?.text, keywords);
  const publishedAt = item?.createdAt ? new Date(item.createdAt).toISOString() : "";
  const insightLines = [
    `Relevance score: ${Number(score || 0).toFixed(2)} for topic "${topicLabel}".`,
    `Source: ${item?.source || "unknown"} via ${item?.handle || "unknown"}; published at ${publishedAt || "unknown time"}.`,
    matched.length
      ? `Matched keywords: ${matched.slice(0, 8).join(", ")}.`
      : "Matched keywords are broad; review manually for semantic relevance.",
    item?.url ? `Original link available for full context and verification.` : "No canonical link available in source payload."
  ];
  return insightLines.filter(Boolean);
}

async function analyzeItem(item, context) {
  return {
    summary: summarizeText(item?.text || ""),
    insights: buildInsights(item, context)
  };
}

module.exports = {
  analyzeItem
};
