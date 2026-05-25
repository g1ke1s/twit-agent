-- Voice Agent v2 — Supabase SQL Schema
-- Run this in your Supabase SQL editor to create all required tables.

-- Runs table
create table if not exists runs (
  run_id            text primary key,
  user_id           text not null,
  status            text not null default 'starting',
  draft             jsonb,
  final_output      text,
  formatted_variants jsonb,
  trace             jsonb,
  validation        jsonb,
  updated_at        timestamptz default now()
);
create index if not exists runs_user_id on runs(user_id);
create index if not exists runs_status  on runs(status);

-- Voice profiles
create table if not exists voice_profiles (
  user_id      text primary key,
  profile_json jsonb not null,
  updated_at   timestamptz default now()
);

-- Feedback / human edits (gold-label data for voice learning)
create table if not exists feedback (
  id              bigserial primary key,
  run_id          text not null,
  user_id         text not null,
  original_draft  text,
  approved_output text,
  decision        text,
  created_at      timestamptz default now()
);
create index if not exists feedback_user_id on feedback(user_id);

-- Shared runs (public read-only share links)
create table if not exists shared_runs (
  share_id          text primary key,
  run_id            text not null,
  user_id           text not null,
  final_output      text,
  formatted_variants jsonb,
  sources           jsonb,
  created_at        timestamptz default now()
);

-- RLS policies — shared_runs is readable by everyone (public share links)
alter table shared_runs enable row level security;
create policy "public_read_shared_runs"
  on shared_runs for select using (true);

-- All other tables: service role only (backend uses service key)
alter table runs           enable row level security;
alter table voice_profiles enable row level security;
alter table feedback       enable row level security;