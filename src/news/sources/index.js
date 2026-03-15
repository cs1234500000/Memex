const { fetchNewsApiItems } = require("./newsApiSource");
const { fetchTwitterApiItems } = require("./twitterApiSource");

const SOURCE_ADAPTERS = {
  newsapi: fetchNewsApiItems,
  twitter: fetchTwitterApiItems
};

async function fetchNewsItems({ subscriptions }) {
  const source = String(subscriptions?.source || "newsapi").toLowerCase();
  const maxItems = Number(subscriptions?.maxItemsPerRun || 60);
  const adapter = SOURCE_ADAPTERS[source];

  if (!adapter) {
    throw new Error(`Unsupported news source "${source}".`);
  }

  const result = await adapter({
    subscriptions,
    maxItems
  });

  return {
    source,
    items: result.items || [],
    warnings: result.warnings || []
  };
}

module.exports = {
  fetchNewsItems
};
