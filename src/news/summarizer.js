function topHandles(matches, limit = 3) {
  const counts = new Map();
  for (const match of matches) {
    const handle = match.item.handle || "unknown";
    counts.set(handle, (counts.get(handle) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([handle, count]) => ({ handle, count }));
}

function fallbackSummary(topicLabel, matches) {
  const top = matches.slice(0, 5);
  const handles = topHandles(matches);

  const summary = `${topicLabel}: ${matches.length} relevant items detected across configured sources. ` +
    `Top signals center on ${top.map((entry) => entry.item.text.split(" ").slice(0, 6).join(" ")).slice(0, 2).join(" / ")}.`;

  const insights = [
    `Most active sources: ${handles.map((h) => `@${h.handle} (${h.count})`).join(", ") || "n/a"}.`,
    `Highest-engagement item: "${top[0]?.item.text || "n/a"}".`,
    `Action: review top ${Math.min(top.length, 3)} items for thesis updates and potential follow-up research.`
  ];

  return {
    summary,
    insights
  };
}

async function summarizeWithOpenAI(topicLabel, matches, apiKey) {
  const snippets = matches.slice(0, 12).map((match, idx) => (
    `${idx + 1}. @${match.item.handle}: ${match.item.text}`
  )).join("\n");

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || "gpt-4o-mini",
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "You are a concise market intelligence analyst. Return strict JSON with keys summary and insights (array of 3 strings)."
        },
        {
          role: "user",
          content: `Topic: ${topicLabel}\nItems:\n${snippets}`
        }
      ]
    })
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error (${response.status})`);
  }

  const payload = await response.json();
  const content = payload.choices?.[0]?.message?.content || "{}";
  const parsed = JSON.parse(content);
  return {
    summary: parsed.summary,
    insights: parsed.insights
  };
}

async function summarizeTopic(topicLabel, matches) {
  if (!matches.length) {
    return {
      summary: `No strong signals detected for ${topicLabel} in this run.`,
      insights: ["Expand tracked handles or keywords to increase coverage."]
    };
  }

  const apiKey = process.env.OPENAI_API_KEY || "";
  if (!apiKey) return fallbackSummary(topicLabel, matches);

  try {
    return await summarizeWithOpenAI(topicLabel, matches, apiKey);
  } catch {
    return fallbackSummary(topicLabel, matches);
  }
}

module.exports = {
  summarizeTopic
};

