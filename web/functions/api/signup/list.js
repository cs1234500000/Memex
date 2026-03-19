function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8"
    }
  });
}

function isAuthorized(request, expectedToken) {
  if (!expectedToken) return false;
  const headerToken = request.headers.get("x-admin-token") || "";
  const queryToken = new URL(request.url).searchParams.get("token") || "";
  return headerToken === expectedToken || queryToken === expectedToken;
}

export async function onRequestGet(context) {
  const adminToken = context.env?.WAITLIST_ADMIN_TOKEN || "";
  if (!isAuthorized(context.request, adminToken)) {
    return json({ error: "Unauthorized." }, 401);
  }

  const supabaseUrl = context.env?.SUPABASE_URL || "";
  const serviceRoleKey = context.env?.SUPABASE_SERVICE_ROLE_KEY || "";
  const table = context.env?.SUPABASE_SIGNUPS_TABLE || "waitlist_signups";

  if (supabaseUrl && serviceRoleKey) {
    const url = `${supabaseUrl}/rest/v1/${table}?select=id,name,email,interest,created_at&order=created_at.desc&limit=1000`;
    const res = await fetch(url, {
      headers: {
        apikey: serviceRoleKey,
        authorization: `Bearer ${serviceRoleKey}`
      }
    });
    if (!res.ok) {
      const details = await res.text();
      return json({ error: `Supabase query failed (${res.status}): ${details}` }, 500);
    }
    const rows = await res.json();
    const items = (rows || []).map((row) => ({
      id: row.id,
      name: row.name || "",
      email: row.email,
      interest: row.interest || "",
      createdAt: row.created_at
    }));
    return json({ count: items.length, items });
  }

  if (!context.env?.WAITLIST_KV || typeof context.env.WAITLIST_KV.list !== "function") {
    return json({ error: "Supabase and WAITLIST_KV are not configured." }, 500);
  }

  const entries = [];
  let cursor = undefined;

  do {
    const page = await context.env.WAITLIST_KV.list({
      prefix: "signup:",
      cursor,
      limit: 100
    });

    for (const key of page.keys) {
      const raw = await context.env.WAITLIST_KV.get(key.name);
      if (!raw) continue;
      try {
        entries.push(JSON.parse(raw));
      } catch {
        // Skip malformed entries.
      }
    }
    cursor = page.cursor;
    if (!page.list_complete && entries.length < 1000) {
      continue;
    }
    break;
  } while (cursor);

  entries.sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")));

  return json({
    count: entries.length,
    items: entries
  });
}

