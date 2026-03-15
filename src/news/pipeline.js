const path = require("path");
const { ensureDir, writeJson, readJson } = require("../lib/jsonStore");
const { getSubscriptions } = require("./subscriptions");
const { fetchNewsItems } = require("./sources");
const { matchItemsByInterest, groupMatchesByTopic } = require("./relevance");
const { summarizeTopic } = require("./summarizer");
const { analyzeItem } = require("./itemInsights");
const { mergeUserKnowledgeFromReport } = require("./knowledgeStore");

const NEWS_DIR = path.join(__dirname, "../../data/news");
const REPORTS_DIR = path.join(NEWS_DIR, "reports");
const LATEST_REPORT_FILE = path.join(NEWS_DIR, "latest.json");

function buildReportId() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function runNewsPipeline({ userId = "default", userProfile = {} } = {}) {
  const subscriptions = await getSubscriptions();
  const sourceResult = await fetchNewsItems({ subscriptions });
  const items = sourceResult.items;

  const matches = await matchItemsByInterest(items, subscriptions.interests);
  const grouped = groupMatchesByTopic(matches);
  const topics = [];

  for (const interest of subscriptions.interests) {
    const group = grouped[interest.id] || { items: [] };
    const sortedItems = [...group.items].sort((a, b) => b.score - a.score);
    const enrichedEntries = await Promise.all(
      sortedItems.map(async (entry) => {
        const analysis = await analyzeItem(entry.item, {
          topicLabel: interest.label,
          keywords: interest.keywords,
          score: entry.score
        });
        return {
          score: Number(entry.score.toFixed(2)),
          item: entry.item,
          summary: analysis.summary,
          insights: analysis.insights
        };
      })
    );
    const summary = await summarizeTopic(interest.label, group.items);
    topics.push({
      topicId: interest.id,
      topicLabel: interest.label,
      matchedCount: enrichedEntries.length,
      summary: summary.summary,
      insights: summary.insights,
      allItems: enrichedEntries,
      topItems: enrichedEntries.slice(0, 5)
    });
  }

  const report = {
    id: buildReportId(),
    generatedAt: new Date().toISOString(),
    source: sourceResult.source,
    subscriptions,
    totalFetchedItems: items.length,
    warnings: sourceResult.warnings,
    topics
  };

  const knowledgeMerge = await mergeUserKnowledgeFromReport({
    userId,
    report,
    userProfile
  });

  await ensureDir(REPORTS_DIR);
  await writeJson(path.join(REPORTS_DIR, `${report.id}.json`), report);
  await writeJson(LATEST_REPORT_FILE, report);
  const responseTopics = report.topics.map((topic) => {
    const { allItems, ...rest } = topic;
    return rest;
  });

  return {
    ...report,
    userId: knowledgeMerge.userId,
    knowledgeMerge,
    topics: responseTopics
  };
}

async function getLatestReport() {
  return readJson(LATEST_REPORT_FILE, null);
}

module.exports = {
  runNewsPipeline,
  getLatestReport
};

