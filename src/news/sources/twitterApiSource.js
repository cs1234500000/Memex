function buildSearchQuery(handles) {
  const fromParts = handles.map((handle) => `from:${handle}`);
  return `(${fromParts.join(" OR ")}) -is:retweet lang:en`;
}

function getUserMap(includes) {
  const users = (includes && includes.users) || [];
  return users.reduce((acc, user) => {
    acc[user.id] = user;
    return acc;
  }, {});
}

function mapTweet(tweet, userMap) {
  const user = userMap[tweet.author_id] || {};
  const handle = user.username || "unknown";
  return {
    id: String(tweet.id),
    text: tweet.text,
    author: user.name || handle,
    handle,
    createdAt: tweet.created_at || new Date().toISOString(),
    metrics: {
      likeCount: tweet.public_metrics?.like_count || 0,
      repostCount: tweet.public_metrics?.retweet_count || 0,
      replyCount: tweet.public_metrics?.reply_count || 0
    },
    url: `https://x.com/${handle}/status/${tweet.id}`,
    source: "twitter"
  };
}

async function fetchTwitterApiItems({ subscriptions, maxItems }) {
  const bearerToken = String(process.env.TWITTER_BEARER_TOKEN || "").trim();
  if (!bearerToken) {
    throw new Error("TWITTER_BEARER_TOKEN is not set.");
  }

  const handles = (subscriptions?.twitterHandles || [])
    .map((item) => String(item || "").replace(/^@/, "").trim())
    .filter(Boolean);
  if (!handles.length) return { items: [], warnings: [] };

  const query = buildSearchQuery(handles);
  const limit = Math.min(Number(subscriptions?.maxTweetsPerRun || maxItems || 60), 100);
  const url = new URL("https://api.twitter.com/2/tweets/search/recent");
  url.searchParams.set("query", query);
  url.searchParams.set("max_results", String(limit));
  url.searchParams.set("tweet.fields", "author_id,created_at,public_metrics");
  url.searchParams.set("expansions", "author_id");
  url.searchParams.set("user.fields", "name,username");

  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${bearerToken}`
    }
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Twitter API failed (${response.status}): ${detail}`);
  }

  const payload = await response.json();
  const userMap = getUserMap(payload.includes);
  const items = (payload.data || [])
    .map((tweet) => mapTweet(tweet, userMap))
    .slice(0, maxItems);

  return {
    items,
    warnings: []
  };
}

module.exports = {
  fetchTwitterApiItems
};
