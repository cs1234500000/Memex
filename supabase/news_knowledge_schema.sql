-- Memex news knowledge schema (Supabase/Postgres)
-- Run this once in Supabase SQL editor.

create table if not exists public.news_users (
  user_id text primary key,
  display_name text not null default '',
  email text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  last_run_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.news_user_topics (
  user_id text not null references public.news_users(user_id) on delete cascade,
  topic_id text not null,
  topic_label text not null default '',
  total_knowledge_items integer not null default 0,
  total_runs_merged integer not null default 0,
  first_merged_at timestamptz,
  last_merged_at timestamptz,
  latest_run_id text,
  latest_source text,
  latest_summary text not null default '',
  latest_insights jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  primary key (user_id, topic_id)
);

create table if not exists public.news_user_topic_runs (
  user_id text not null references public.news_users(user_id) on delete cascade,
  topic_id text not null,
  run_id text not null,
  generated_at timestamptz not null,
  source text not null,
  matched_count integer not null default 0,
  summary text not null default '',
  insights jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  primary key (user_id, topic_id, run_id),
  foreign key (user_id, topic_id) references public.news_user_topics(user_id, topic_id) on delete cascade
);

create table if not exists public.news_user_knowledge_items (
  user_id text not null references public.news_users(user_id) on delete cascade,
  topic_id text not null,
  item_id text not null,
  source text not null default '',
  content_type text not null default 'article',
  url text not null default '',
  author text not null default '',
  origin text not null default '',
  title text not null default '',
  text_content text not null default '',
  published_at timestamptz,
  discovered_at timestamptz not null default now(),
  run_id text not null default '',
  score double precision not null default 0,
  item_summary text not null default '',
  item_insights jsonb not null default '[]'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  primary key (user_id, topic_id, item_id),
  foreign key (user_id, topic_id) references public.news_user_topics(user_id, topic_id) on delete cascade
);

create index if not exists idx_news_topics_user on public.news_user_topics(user_id);
create index if not exists idx_news_items_user_topic_published
  on public.news_user_knowledge_items(user_id, topic_id, published_at desc);
create index if not exists idx_news_runs_user_topic_generated
  on public.news_user_topic_runs(user_id, topic_id, generated_at desc);

alter table public.news_user_knowledge_items
  add column if not exists item_summary text not null default '';

alter table public.news_user_knowledge_items
  add column if not exists item_insights jsonb not null default '[]'::jsonb;
