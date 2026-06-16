# OmniAgent Flow - Interview Study Guide

This document is the single source of truth for explaining the
OmniAgent Flow project in a technical interview. It mirrors the
project structure and walks through the **why** behind every decision
so you can answer follow-up questions without flipping through the
codebase.

Use the table of contents to jump to the topic the interviewer is
probing. Each section ends with sample Q&A so you can rehearse.

## Table of contents

1. [Project elevator pitch](#1-project-elevator-pitch)
2. [System architecture](#2-system-architecture)
3. [Phase 1 - Core backend & async webhook](#3-phase-1---core-backend--async-webhook)
4. [Phase 2 - Agentic AI & advanced RAG](#4-phase-2---agentic-ai--advanced-rag)
5. [Phase 3 - CRM & real-time notifications](#5-phase-3---crm--real-time-notifications)
6. [Phase 4 - Observability, testing, dashboards](#6-phase-4---observability-testing-dashboards)
7. [Operational playbook](#7-operational-playbook)
8. [Common interview questions](#8-common-interview-questions)

---

## 1. Project elevator pitch

**One-sentence pitch:** A production-shaped multi-channel AI customer
support agent that ingests webhook events, classifies customer intent
with a LangGraph state machine, retrieves grounded answers via hybrid
RAG, syncs qualified leads to HubSpot, alerts humans on Telegram, and
ships its traces to LangFuse.

**Why this project exists for your CV:**
- It demonstrates the **full async pipeline** pattern most companies
  want from a senior backend engineer: webhook -> queue -> worker ->
  external APIs.
- It mixes **GenAI engineering** (LangGraph, structured outputs, RAG,
  evaluation) with **traditional backend** (FastAPI, Celery, Redis,
  PostgreSQL, Docker).
- It includes **observability** and **testing**, which most portfolio
  projects skip but every interview loop asks about.

**Tech stack at a glance:**

| Layer            | Tool                                            |
| ---------------- | ----------------------------------------------- |
| HTTP API         | FastAPI (async)                                 |
| Task queue       | Celery + Redis broker                          |
| Short-term memory| Redis (list + TTL 1800s)                        |
| Long-term memory | PostgreSQL (conversations, messages, sync log)  |
| Agent            | LangGraph state machine                         |
| LLM              | OpenAI structured outputs                       |
| Vector DB        | Qdrant (hybrid dense + BM25 + reranker)         |
| CRM              | HubSpot v3 contacts API                        |
| Notifications    | Telegram Bot HTTP API                          |
| Observability    | LangFuse traces + scores                       |
| Containerisation | Docker + docker-compose                        |
| Testing          | pytest + pytest-asyncio                         |

---

## 2. System architecture

```
              Facebook Messenger / Web Channel
                          |
                          v  POST /api/webhook
              +---------------------------+
              |       FastAPI app         |
              |  (returns HTTP 200 fast)  |
              +-----------+---------------+
                          |
                          v  .delay()
              +---------------------------+
              |   Celery worker (Redis)   |
              +-----------+---------------+
                          |
              +-----------+-----------+
              | generate_agent_result  |
              |  - RAG retrieval       |
              |  - LangGraph run       |
              |  - LangFuse trace      |
              +-----------+-----------+
                          |
        +-----------------+------------------+
        v                 v                  v
  Redis session     PostgreSQL         LangFuse cloud
  (short-term)      (durable history)  (traces + scores)
        |
        v
  +---------------------+
  | sync_hubspot_lead   |
  +---------------------+
        |
        v
  +-------------------------+
  | send_telegram_... (RT)  |
  +-------------------------+
```

### Why an asynchronous pipeline?

A Facebook webhook expects an HTTP 200 in **under 5 seconds**; if you
call an LLM synchronously you risk timeouts, retried deliveries, and
duplicate processing. Pushing the payload into Celery guarantees
**at-least-once** processing with **fast ack** to the upstream
channel. We isolate slow work (LLM, vector search, HubSpot) from the
hot HTTP path. This is the same pattern used by payment webhooks
(Stripe), messaging (Twilio), and CI/CD systems (GitHub Actions).

---

## 3. Phase 1 - Core backend & async webhook

### 3.1 Folder layout

```
src/
  main.py            - FastAPI app + lifespan
  config.py          - Pydantic settings (env-driven)
  database.py        - asyncpg + redis-asyncio pools
  api/webhook.py     - GET (verify) + POST (enqueue)
  workers/
    celery_app.py    - broker / backend config
    tasks.py         - process_incoming_message
  services/
    session_service.py   - Redis session history
    conversation_service.py - PostgreSQL persistence
    ai_service.py       - LangGraph orchestration
    intent_service.py   - structured output extraction
    rag_service.py      - hybrid retrieval
    hubspot_service.py  - CRM sync
    telegram_service.py - real-time alerts
    observability_service.py - LangFuse wrapper
    evaluation_service.py   - faithfulness/relevance scorers
```

### 3.2 Webhook verification

Facebook uses the **hub.mode / hub.verify_token / hub.challenge**
handshake. The webhook returns the `hub.challenge` only when the
token matches. We never accept inbound traffic without the handshake
because anyone could post fake messages otherwise.

### 3.3 Push-to-queue pattern

```python
@router.post("")
async def receive_webhook(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
    task = process_incoming_message.delay(payload)
    return {"status": "success", "task_id": task.id}
```

The handler does **two** things:
1. Enqueue the raw payload to Celery.
2. Return `200 OK` with the task id.

This is the strictest possible contract for an external webhook:
**never** do work synchronously, **never** return error codes for
recoverable failures. The client just needs the acknowledgement.

### 3.4 Redis session history

`session_service.py` keeps a sliding window of the last 10 messages
in a Redis list with a 30-minute TTL. The window size is the
hyper-parameter that balances context richness against token cost.

Key points you can defend in an interview:
- We store JSON strings in a Redis list, not a hash. Lists preserve
  insertion order and let us `LTRIM` to enforce a cap.
- We use `RPUSH` + `LTRIM -N -1` to keep only the newest N entries.
- `EXPIRE` is re-issued on every write so the session lives for 30
  minutes since the last message, **not** since the first.

### 3.5 Celery task

The worker uses `asyncio.run` inside the synchronous Celery task.
This is a pragmatic choice: the task body is small and contains no
shared event loop, so spinning up a fresh loop per task is cheaper
than juggling the worker loop manually. For higher throughput, a
Celery `gevent` pool would let us reuse one loop.

---

## 4. Phase 2 - Agentic AI & advanced RAG

### 4.1 Why LangGraph?

A bare LLM call cannot reliably do **routing**. LangGraph gives us:

- A typed `AgentState` that is mutated by nodes.
- A `route_agent_action` function that maps `intent -> response_node`.
- Conditional edges so the graph literally branches on the
  classifier's output.

This is the right primitive for any "AI that has to take a different
path based on the user's situation" workflow. It also makes the
graph serializable, which is what LangFuse traces render.

### 4.2 Structured outputs + Pydantic

```python
class CustomerIntentExtraction(BaseModel):
    intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0)
    customer_name: str | None
    company: str | None
    phone: str | None
    email: str | None
    budget: str | None
    ...
```

We pass `text_format=CustomerIntentExtraction` to OpenAI. The model
returns JSON that Pydantic validates; we get a typed object back. No
string parsing, no regex on LLM output, no retry loops. We also keep
a deterministic fallback extractor (`_build_fallback_extraction`)
that runs on keyword heuristics when the OpenAI key is missing, so
the system stays demo-able offline.

### 4.3 RAG pipeline

`rag_service.py` implements **hybrid search** end-to-end:

1. **Dense retrieval**: hash-based embeddings (deterministic for the
   demo) + cosine similarity in Qdrant. The query embeds, the top-K
   candidates are returned.
2. **BM25 sparse retrieval**: a Python implementation of the BM25
   ranking function over a scrolled corpus. Weights can be tuned via
   `RAG_DENSE_WEIGHT` and `RAG_BM25_WEIGHT`.
3. **Hybrid fusion**: weighted sum of normalised dense and BM25
   scores. `RAG_CANDIDATE_LIMIT` controls the candidate pool size.
4. **Reranking**: optional cross-encoder via `fastembed`
   (`BAAI/bge-reranker-base`). When `RAG_ENABLE_RERANKER=true` the
   top-K candidates are re-scored by a stronger model to reduce
   hallucination.

Why both? Dense retrieval is great at semantic matches
("how do I unsubscribe?" ~ "cancel my plan") but bad at exact keyword
matches (product SKUs, error codes). BM25 is the opposite. Hybrid
+ reranker is the production-grade answer that every serious
RAG system converges on.

### 4.4 Putting RAG into the graph

We call `hybrid_search_knowledge(user_message, limit=3)` **before**
invoking the agent and inject the context as a synthetic
`role: system` message. The LangGraph nodes stay pure: they only see
the state and respond. This separation lets us test the graph
without RAG and unit-test the retriever in isolation.

---

## 5. Phase 3 - CRM & real-time notifications

### 5.1 HubSpot sync

`sync_hubspot_lead` follows the standard "search then upsert" pattern:

1. Build a strongly-typed `HubSpotLeadPayload` from the metadata
   extracted by the LangGraph.
2. POST to `/crm/v3/objects/contacts/search` filtering by email,
   then by phone.
3. If a contact exists -> PATCH the properties. If not -> POST to
   create. The `action` returned is "created" or "updated".
4. Always log a `hubspot_lead_syncs` row to PostgreSQL so we can audit
   every sync attempt, including the failures.

Defensive choices:
- We use an `httpx.AsyncClient` and a `HubSpotHTTPClient` Protocol
  so unit tests can inject a fake client.
- We close the client in a `finally` block to avoid connection
  leaks.
- The service is **fail-soft**: any error returns a structured
  `HubSpotLeadSyncResult` so the worker can still finish the rest of
  the pipeline and the alert is sent.

### 5.2 Telegram real-time alerts

`send_telegram_notification` is the simplest piece of the system. It
sends an HTML-formatted message to a configured chat via the
`sendMessage` Bot API. The four event types are:

- `hubspot_sync_failed` -> CRM write failed; needs investigation.
- `handoff_requested`   -> customer asked for a human.
- `hot_lead_captured`   -> pricing intent + CRM synced.
- `new_message`         -> catch-all.

We pick the event type from `(intent, hubspot_status)` so the on-call
channel can filter easily. We escape every user-controlled field with
`html.escape` because Telegram interprets HTML.

Why not Slack? Slack is heavy (OAuth, scopes, signing secrets).
Telegram Bot API is one HTTP call - perfect for a demo and for any
small team that just wants push notifications.

---

## 6. Phase 4 - Observability, testing, dashboards

### 6.1 LangFuse observability

`observability_service.py` wraps the LangFuse SDK behind a tiny
abstraction. The module exports:

- `is_observability_enabled()` - quick check.
- `atrace_agent_run` - async context manager that opens a trace.
- `record_intent_generation` - records a generation span with model,
  usage, latency.
- `record_evaluation_score` - attaches a numeric score (0-1) to the
  trace.

The module ships a **null client** so tests and offline development
work even without a LangFuse account. Switching it on is a one-env-
variable change: `LANGFUSE_ENABLED=true`.

### 6.2 LLM evaluation

`evaluation_service.py` ships two scorers that the LangFuse
dashboard consumes:

- **Faithfulness**: how much of the answer is grounded in the
  retrieved context? We compute it as token overlap with the
  retrieved docs. With an OpenAI key, we instead call the model and
  ask for a 0-1 score with a rubric.
- **Answer relevance**: is the answer actually about the question?
  We blend token overlap with the user question and a keyword bonus
  for known intent terms.

Both fall back to deterministic heuristics when the LLM is
unavailable. The result is folded into the agent's metadata, written
to PostgreSQL, and pushed to LangFuse as a score.

### 6.3 Testing

The test suite lives in `tests/` and runs with pytest. We split it
by service so failures are easy to localise:

- `test_webhook.py`         - HTTP contract, GET/POST, 403 on bad
                              token, Celery enqueue.
- `test_session_service.py` - Redis-backed list operations.
- `test_rag_service.py`     - hybrid search, BM25, reranker toggle.
- `test_intent_service.py`  - structured output + fallback.
- `test_agent.py`           - LangGraph routing.
- `test_hubspot_service.py` - upsert flow with a fake client.
- `test_telegram_service.py`- HTML escaping, send/skip/fail.
- `test_conversation_service.py` - schema + helpers.
- `test_observability_service.py` - null client behaviour.
- `test_evaluation_service.py`    - fallback scorers.
- `test_tasks.py`           - Celery task pipeline.
- `test_app_factory.py`     - lifespan + healthcheck.

Run with `python -m pytest tests/ -v`. The current run is 45 tests,
all green, in under 2 seconds.

### 6.4 Looker Studio dashboards

We do **not** stream the events into a warehouse. The dashboards
read directly from PostgreSQL via SQL views. The views live in
`migrations/0010_looker_views.sql`:

- `vw_daily_intent_volume`      - stacked bar of intents per day.
- `vw_intent_summary`           - KPIs (pricing, handoff, fallback).
- `vw_hubspot_sync_outcomes`    - CRM sync health.
- `vw_conversation_insights`    - lead leaderboard (channels, pain
                                  points flattened from JSONB).
- `vw_conversation_volume_hourly` - hour-of-day heatmap.

The Looker Studio report guide is in `docs/looker_studio.md`. The
contract between the BI tool and the database is the set of views;
we can change the underlying tables without breaking the dashboard.

---

## 7. Operational playbook

### Local development

```bash
# Start infra + app
docker compose up --build

# Run the worker (separate terminal)
celery -A src.workers.tasks worker --loglevel=info
```

### Smoke tests

```bash
# Webhook verification
python test_webhook.py

# Session flow
python test_session.py

# Full pipeline (webhook -> queue -> agent -> DB)
python test_queue.py

# RAG / agent / conversation / HubSpot unit tests
python -m pytest tests/ -v
```

### Production deployment notes

- `WEBHOOK_VERIFY_TOKEN` must be rotated and stored in a secret
  manager. The token is the **only** thing standing between an
  attacker and your Celery queue.
- The Postgres credentials should come from a managed secret store
  (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault). The
  `docker-compose.yml` already fails fast if any of them is empty.
- `LANGFUSE_*` keys are optional. Without them the system still
  works - it just stops sending traces.
- `HUBSPOT_SYNC_ENABLED=true` plus a private app token is required
  for live CRM writes.
- `TELEGRAM_NOTIFICATIONS_ENABLED=true` requires a bot token and
  the chat id of the target group (negative number for groups).

---

## 8. Common interview questions

### Q1. Why is the webhook async?

> Facebook expects an HTTP 200 in <5s. LLM calls + RAG + HubSpot
> writes can take seconds. If we answered synchronously we'd risk
> timeouts and duplicate processing. We push to Celery and ack in
> <500ms. The worker handles slow work, retries, and failures.

### Q2. Why Redis for sessions and Postgres for messages?

> Sessions are bounded, short-lived, and need O(1) reads/writes
> with TTL. Redis is built for that. Conversation history must
> survive container restarts and be queryable by date/intent for
> analytics. That's a relational use case, hence Postgres with
> JSONB for metadata flexibility.

### Q3. How do you prevent the LLM from hallucinating?

> Three layers:
> 1. **Structured outputs** force the LLM to answer against a
>    Pydantic schema. It cannot invent free-form fields.
> 2. **Hybrid RAG + reranker** grounds the response in the
>    knowledge base. We inject the retrieved docs as a system
>    message before the agent runs.
> 3. **Faithfulness evaluation** scores every turn and ships the
>    score to LangFuse so we can spot regressions in the dashboard.

### Q4. Why LangGraph instead of a single prompt?

> A single prompt cannot reliably route. LangGraph lets us declare
> an explicit `AgentState`, a `classify_intent` node, and four
> response branches. The graph is testable, traceable, and
> composable. It is the right primitive for any branching
> AI workflow (routing, escalation, handoff).

### Q5. How do you test an LLM-powered system without flake?

> We test **what we own**, not the LLM. The agent's routing is
> tested with the structured-output extractor disabled and a
> deterministic fallback classifier. The RAG service is tested
> with deterministic hash embeddings. The HubSpot and Telegram
> services use protocol-based fake clients. Only the OpenAI call
> itself is left as a network dependency, and that one is behind
> a `try/except` that falls back to the deterministic path.

### Q6. What would you change for production?

> - Replace hash embeddings with a real model (text-embedding-3-
>   small or BGE) and store vectors in Qdrant with HNSW.
> - Use Celery's `gevent` worker pool for higher concurrency.
> - Add a retry policy with exponential backoff for HubSpot and
>   OpenAI.
> - Move the SQL views into a read replica with a dedicated
>   analytics user.
> - Replace the null LangFuse client with a queue-based exporter
>   so traces survive worker restarts.

### Q7. How does the system stay fail-soft?

> Every external call (HubSpot, Telegram, OpenAI, LangFuse) is
> wrapped in a `try/except`. The pipeline never crashes because
> Telegram is down or HubSpot is rate-limiting. Failures are
> logged via `loguru` and persisted in PostgreSQL so we can
> reconcile them later.

### Q8. Walk me through a real message.

> 1. Customer sends "Tôi muốn xin báo giá" on Messenger.
> 2. Facebook POSTs to `/api/webhook`. FastAPI parses it, pushes
>    the payload onto Celery, returns 200 OK with the task id.
> 3. The worker pulls the task, loads the session from Redis
>    (last 10 messages), and pulls the top-3 RAG docs.
> 4. `ai_service.generate_agent_result` opens a LangFuse trace,
>    invokes the LangGraph state machine, classifies the intent
>    as `pricing`, and runs the `pricing_response` node.
> 5. The response is saved to Redis (with TTL refresh) and
>    PostgreSQL with the structured metadata.
> 6. The HubSpot service creates a contact with `lifecyclestage =
>    lead`. The Telegram service fires a `hot_lead_captured`
>    alert to the sales channel.
> 7. The LangFuse trace receives the faithfulness and relevance
>    scores, then the trace is flushed.
