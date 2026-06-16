-- =============================================================================
-- OmniAgent Flow - Analytics views for Looker Studio dashboards
-- =============================================================================
-- This migration adds SQL views that the conversation insights dashboard
-- (Looker Studio) consumes. Run after the conversation schema in
-- src.services.conversation_service has been initialized.
--
-- Usage:
--   psql "$POSTGRES_DSN" -f migrations/0010_looker_views.sql
--
-- Each view is documented with the dashboard widget it powers.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- View: vw_daily_intent_volume
--   Powers: stacked bar chart "Intent volume by day"
--   Filters: date range, channel
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_daily_intent_volume AS
SELECT
    DATE_TRUNC('day', cm.created_at) AS day,
    c.channel,
    COALESCE(cm.intent, 'unknown')    AS intent,
    COUNT(*)                          AS message_count
FROM conversation_messages cm
INNER JOIN conversations c ON c.id = cm.conversation_id
WHERE cm.role = 'user'
  AND cm.intent IS NOT NULL
GROUP BY day, c.channel, cm.intent
ORDER BY day DESC, message_count DESC;


-- -----------------------------------------------------------------------------
-- View: vw_intent_summary
--   Powers: KPI tiles "Total conversations", "Pricing leads", "Handoff rate"
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_intent_summary AS
SELECT
    DATE_TRUNC('day', c.updated_at) AS day,
    c.channel,
    COUNT(DISTINCT c.id)            AS total_conversations,
    COUNT(DISTINCT CASE
        WHEN c.last_intent = 'pricing'   THEN c.id
    END)                            AS pricing_conversations,
    COUNT(DISTINCT CASE
        WHEN c.last_intent = 'consultation' THEN c.id
    END)                            AS consultation_conversations,
    COUNT(DISTINCT CASE
        WHEN c.last_intent = 'handoff' THEN c.id
    END)                            AS handoff_conversations,
    COUNT(DISTINCT CASE
        WHEN c.last_intent = 'fallback' THEN c.id
    END)                            AS fallback_conversations
FROM conversations c
GROUP BY day, c.channel
ORDER BY day DESC;


-- -----------------------------------------------------------------------------
-- View: vw_hubspot_sync_outcomes
--   Powers: pie chart "HubSpot sync status", line chart "Sync failure trend"
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_hubspot_sync_outcomes AS
SELECT
    DATE_TRUNC('day', h.created_at) AS day,
    h.status,
    h.action,
    h.intent,
    COUNT(*)                        AS sync_count
FROM hubspot_lead_syncs h
GROUP BY day, h.status, h.action, h.intent
ORDER BY day DESC, sync_count DESC;


-- -----------------------------------------------------------------------------
-- View: vw_conversation_insights
--   Powers: leaderboard "Top companies", "Top channels", "Top pain points"
--   Aggregates metadata JSONB fields so the BI tool does not have to.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_conversation_insights AS
SELECT
    c.id                            AS conversation_id,
    c.channel,
    c.external_sender_id,
    c.last_intent,
    c.updated_at                    AS last_activity_at,
    cm.metadata ->> 'company'       AS company,
    cm.metadata ->> 'customer_name' AS customer_name,
    cm.metadata ->> 'phone'         AS phone,
    cm.metadata ->> 'email'         AS email,
    cm.metadata ->> 'urgency'       AS urgency,
    cm.metadata ->> 'language'      AS language,
    cm.metadata ->> 'product_interest' AS product_interest,
    cm.metadata ->> 'budget'        AS budget,
    ARRAY(
        SELECT jsonb_array_elements_text(
            COALESCE(cm.metadata -> 'channels', '[]'::jsonb)
        )
    )                               AS channels,
    ARRAY(
        SELECT jsonb_array_elements_text(
            COALESCE(cm.metadata -> 'pain_points', '[]'::jsonb)
        )
    )                               AS pain_points
FROM conversations c
LEFT JOIN LATERAL (
    SELECT metadata
    FROM conversation_messages
    WHERE conversation_id = c.id
    ORDER BY created_at DESC
    LIMIT 1
) cm ON TRUE;


-- -----------------------------------------------------------------------------
-- View: vw_conversation_volume_hourly
--   Powers: heatmap "Conversation volume by hour of day"
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_conversation_volume_hourly AS
SELECT
    DATE_TRUNC('hour', cm.created_at)       AS hour_bucket,
    EXTRACT(DOW    FROM cm.created_at)::int  AS day_of_week,
    EXTRACT(HOUR   FROM cm.created_at)::int  AS hour_of_day,
    c.channel,
    COUNT(*)                                AS message_count
FROM conversation_messages cm
INNER JOIN conversations c ON c.id = cm.conversation_id
WHERE cm.role = 'user'
GROUP BY hour_bucket, day_of_week, hour_of_day, c.channel
ORDER BY hour_bucket DESC;
