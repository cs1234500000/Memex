const DEFAULT_SUBSCRIPTIONS = {
  source: "newsapi",
  twitterHandles: [
    "OpenAI",
    "AnthropicAI",
    "perplexity_ai",
    "ycombinator",
    "a16z",
    "sama"
  ],
  sources: {
    newsapi: {
      language: "en",
      sortBy: "publishedAt",
      domains: [],
      sourceIds: []
    }
  },
  interests: [
    {
      id: "ai_startups",
      label: "AI startup companies",
      keywords: [
        "ai startup",
        "startup",
        "funding",
        "seed",
        "series a",
        "series b",
        "launch",
        "agent",
        "llm",
        "model",
        "inference",
        "enterprise ai",
        "open source"
      ]
    }
  ],
  maxItemsPerRun: 60,
  maxTweetsPerRun: 60
};

module.exports = {
  DEFAULT_SUBSCRIPTIONS
};

