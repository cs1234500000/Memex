# Memex Architecture Map

This repo uses a **Node-first architecture** with Cloudflare Pages Functions in production.

## Folder responsibilities

- `src/`
  - Node.js domain modules for news intelligence (`src/news/*`)
  - Node-local API handlers for news endpoints (`src/api/newsHandlers.js`)
  - Used by `server.js`

- `server.js`
  - Local Node server for:
    - Static site hosting (`index.html`)
    - News endpoints (`/api/news/*`)
    - Signup endpoints (`/api/signup`, `/api/signup/list`)
  - Good for local full-stack development in one process

- `functions/api/`
  - Cloudflare Pages Functions (production serverless endpoints)
  - Handles live site endpoints on `withmemex.com` (currently signup endpoints)

## Runtime choices in practice

1. **Local development**
   - Run `npm start`
   - Uses `server.js` + `src/*`
2. **Production**
   - Deploy static assets + `functions/api/*` on Cloudflare Pages

## Current production truth

- `withmemex.com` live API uses `functions/api/*` (Cloudflare Pages Functions).
- Local Node runtime is for development and testing before deployment.
