# Looker Studio Dashboard Guide

This document explains how the Conversation Insights dashboard is wired
up. It is intentionally short: the heavy lifting happens in the SQL
views under `migrations/0010_looker_views.sql`.

## 1. Connect Looker Studio to PostgreSQL

1. Open https://lookerstudio.google.com and create a new report.
2. Add a data source: **PostgreSQL connector**.
3. Use the same `POSTGRES_*` credentials from `.env` (host, port,
   user, password, database). In production, prefer a read-only
   analytics user that can only `SELECT` from the views.
4. Pick the views listed below. Do not connect to the raw tables - the
   views are the contract.

## 2. Views to import

| View                          | Powers                                  | Suggested chart type   |
| ----------------------------- | --------------------------------------- | ---------------------- |
| `vw_daily_intent_volume`      | Daily intent mix                        | Stacked bar chart      |
| `vw_intent_summary`           | KPI tiles (pricing, handoff, fallback)  | Scorecards             |
| `vw_hubspot_sync_outcomes`    | CRM sync health                         | Pie + time series      |
| `vw_conversation_insights`    | Lead leaderboard                        | Table                  |
| `vw_conversation_volume_hourly` | Hour-of-day heatmap                   | Heatmap                |

## 3. Recommended widgets

- **KPI scorecards** from `vw_intent_summary`:
  - `SUM(pricing_conversations)` -> "Pricing leads"
  - `SUM(handoff_conversations)` -> "Handoff requests"
  - `SUM(fallback_conversations) / SUM(total_conversations)` -> "Fallback rate"
- **Stacked bar**: `day` x `message_count`, stacked by `intent`
  (data: `vw_daily_intent_volume`).
- **Heatmap**: rows = `day_of_week` (0-6), columns = `hour_of_day`
  (0-23), metric = `SUM(message_count)`.
- **Table**: top companies, channels, and pain points from
  `vw_conversation_insights` with a filter on `last_intent IN
  ('pricing', 'handoff')`.

## 4. Filters

Expose a date-range control and a channel multi-select control at the
report level. The SQL views already aggregate to day/hour granularity
so Looker Studio will only need to apply the standard date filter.

## 5. Refresh

Looker Studio will cache results for the duration of the report session.
To force a refresh after a new conversation, hit the refresh button in
the top bar. For a near real-time experience, schedule a 5-minute
refresh in the data source settings.

## 6. SQL view maintenance

When the `conversation_messages.metadata` JSONB shape changes, update
`vw_conversation_insights` to expose the new fields. Keep the view
defensive: any field that may be null should be wrapped in
`COALESCE(..., '[]'::jsonb)` or `->> 'field'`.
