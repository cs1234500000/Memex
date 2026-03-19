const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

async function sendWaitlistConfirmationEmail(env, recipientEmail) {
  const apiKey = env?.RESEND_API_KEY || "";
  const fromEmail = env?.WAITLIST_FROM_EMAIL || "hello@withmemex.com";
  const subject = env?.WAITLIST_CONFIRM_SUBJECT || "You're on the Memex waitlist";
  const siteUrl = env?.WAITLIST_SITE_URL || "https://withmemex.com";

  if (!apiKey) {
    console.warn("RESEND_API_KEY is not configured; skipping confirmation email.");
    return;
  }

  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      from: fromEmail,
      to: [recipientEmail],
      subject,
      html: `
        <div style="font-family:Arial,sans-serif;line-height:1.6;color:#1a1814">
          <p>You're in - thanks for joining the Memex waitlist.</p>
          <p>We'll reach out from <b>hello@withmemex.com</b> as we open access.</p>
          <p>Website: <a href="${siteUrl}">${siteUrl}</a></p>
        </div>
      `
    })
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Failed to send confirmation email (${response.status}): ${details}`);
  }
}

async function trySendConfirmationEmail(env, email) {
  try {
    await sendWaitlistConfirmationEmail(env, email);
  } catch (error) {
    console.error("Confirmation email failed:", error?.message || error);
  }
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8"
    }
  });
}

export async function onRequestOptions() {
  return new Response(null, {
    headers: {
      Allow: "POST, OPTIONS"
    }
  });
}

export async function onRequestPost(context) {
  let payload;
  try {
    payload = await context.request.json();
  } catch {
    return json({ error: "Invalid JSON body." }, 400);
  }

  const name = String(payload?.name || "").trim();
  const email = String(payload?.email || "").trim().toLowerCase();
  const interest = String(payload?.interest || "waitlist").trim();

  if (!EMAIL_RE.test(email)) {
    return json({ error: "Valid email is required." }, 400);
  }

  const signup = {
    id: crypto.randomUUID(),
    name,
    email,
    interest,
    createdAt: new Date().toISOString()
  };

  const supabaseUrl = context.env?.SUPABASE_URL || "";
  const serviceRoleKey = context.env?.SUPABASE_SERVICE_ROLE_KEY || "";
  const table = context.env?.SUPABASE_SIGNUPS_TABLE || "waitlist_signups";

  if (supabaseUrl && serviceRoleKey) {
    const existsUrl = `${supabaseUrl}/rest/v1/${table}?select=id&email=eq.${encodeURIComponent(email)}&limit=1`;
    const existsRes = await fetch(existsUrl, {
      headers: {
        apikey: serviceRoleKey,
        authorization: `Bearer ${serviceRoleKey}`
      }
    });
    if (!existsRes.ok) {
      return json({ error: `Supabase read failed (${existsRes.status}).` }, 500);
    }
    const existingRows = await existsRes.json();
    if (Array.isArray(existingRows) && existingRows.length > 0) {
      return json({ error: "This email is already registered." }, 409);
    }

    const insertRes = await fetch(`${supabaseUrl}/rest/v1/${table}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        apikey: serviceRoleKey,
        authorization: `Bearer ${serviceRoleKey}`,
        prefer: "return=representation"
      },
      body: JSON.stringify({
        id: signup.id,
        name: signup.name,
        email: signup.email,
        interest: signup.interest,
        created_at: signup.createdAt
      })
    });
    if (!insertRes.ok) {
      const details = await insertRes.text();
      return json({ error: `Supabase insert failed (${insertRes.status}): ${details}` }, 500);
    }
    await trySendConfirmationEmail(context.env, email);
    return json({ ok: true });
  }

  // Fallback persistence: KV if Supabase is not configured.
  if (context.env?.WAITLIST_KV && typeof context.env.WAITLIST_KV.put === "function") {
    const emailKey = `signup_email:${email}`;
    const existing = await context.env.WAITLIST_KV.get(emailKey);
    if (existing) {
      return json({ error: "This email is already registered." }, 409);
    }

    const signupKey = `signup:${signup.createdAt}:${signup.id}`;
    await context.env.WAITLIST_KV.put(signupKey, JSON.stringify(signup));
    await context.env.WAITLIST_KV.put(emailKey, signupKey);
  } else {
    // Last fallback so endpoint still works during setup.
    console.log("WAITLIST signup", signup);
  }

  await trySendConfirmationEmail(context.env, email);
  return json({ ok: true });
}

