# OmniAgent Flow - Tài liệu ôn tập phỏng vấn

Tài liệu này giải thích dự án OmniAgent Flow (phiên bản đơn giản)
trong buổi phỏng vấn kỹ thuật. Nó bám sát cấu trúc code và giải
thích **vì sao** mỗi quyết định được đưa ra.

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Chi tiết từng module](#3-chi-tiết-từng-module)
4. [Các quyết định thiết kế quan trọng](#4-các-quyết-định-thiết-kế-quan-trọng)
5. [Playbook vận hành](#5-playbook-vận-hành)
6. [Câu hỏi phỏng vấn thường gặp](#6-câu-hỏi-phỏng-vấn-thường-gặp)

---

## 1. Tổng quan dự án

**Một câu:** Telegram chatbot RAG — crawl website công ty, index vào
Qdrant, LLM trả lời tiếng Việt có citation. Không URL → entity
detection hỏi user. Câu hỏi trùng → cache Redis trả ngay.

**Ba tính năng chính:**

1. **URL detection + Crawl**: user gửi URL → sitemap discovery →
   crawl4ai (Chromium headless) → chunk → embed (`text-embedding-3-small`)
   → Qdrant.
2. **RAG + LLM chat**: user hỏi → top-3 chunks → GPT-4o-mini trả lời
   tiếng Việt có citation `[n]`.
3. **Fallback UX**: RAG không tìm → xin email. Câu hỏi trùng → cache 1h.

**Stack thực tế trong CV:**
> FastAPI async · Redis (session, cache, TTL) · Qdrant (vector store) ·
> OpenAI (embedding + LLM) · crawl4ai (web scraping) · pytest + pytest-asyncio

---

## 2. Kiến trúc hệ thống

```
Telegram ──► FastAPI POST /api/telegram/webhook
                     │
           ┌─────────┼──────────┐
           ▼         ▼          ▼
       Entity    RAG search   Telegram
       detection               send
           │         │          ▲
           │         ▼          │
           │    Qdrant          │
           │         │          │
           │         ▼          │
           │    OpenAI GPT-4o-mini
           │         │
           │         ▼
           │      Redis (session + cache)
           │
           ▼
      crawl4ai ──► Qdrant
```

**Đặc điểm quan trọng:** Webhook **đồng bộ** (không Celery) — Telegram
không yêu cầu 5s timeout như Facebook. Với mô hình này, ta gọi LLM
trực tiếp trong request với streaming response (in-place edit Telegram
message) để user thấy typing indicator.

---

## 3. Chi tiết từng module

### 3.1 `src/crawler.py` — Web scraping

```
discover_urls(url)
  │ sitemap.xml → _parse_sitemap_xml (hỗ trợ sitemap index lồng nhau)
  │ robots.txt → Sitemap: directive
  ▼
filter_same_domain(urls, base_url)  ← loại Twitter/LinkedIn
  ▼
crawl_full_website(url, max_pages=50)
  │ crawl4ai AsyncWebCrawler (Chromium headless)
  │ fit_markdown (loại bỏ boilerplate: nav, footer, cookie banner)
  ▼
list[CrawlResult]  {url, title, markdown, success, error_message}
```

Điểm đáng chú ý:
- **sitemap discovery**: tự động tìm `/sitemap.xml`, `sitemap_index.xml`,
  và `Sitemap:` directive trong `robots.txt`.
- **retry 2 lần** với backoff 1.5s — không retry vô hạn.
- **fail isolation**: mỗi URL có `success`/`error_message` riêng, một 404
  không làm sập batch.
- **chunk_markdown()**: chia markdown thành chunks 1000 chars, overlap 100
  chars, tách câu cho paragraphs dài.

### 3.2 `src/rag.py` — Vector store

```
markdown + url + title
  │
  ▼ chunk_markdown()
list[chunk_text]
  │
  ▼ _embed_in_batches()  ← 64 chunks/call, average vector khi chunk bị split
list[vector 1536-dim]
  │
  ▼ _upsert_points() → Qdrant
```

```
query
  │
  ▼ embed_query() → vector 1536-dim
  ▼ Qdrant query_points(limit=top_k) → list[RagSearchResult]
  │   payload: {text, url, title, page_chunk_index, page_chunk_total}
  ▼ format_context() → string (giới hạn 6000 chars)
```

Điểm đáng chú ý:
- **`_split_oversize_text()`**: chunk > 7000 chars được tách trước khi
  embed để tránh vượt limit 8192 tokens của OpenAI embed API.
- **`_embed_in_batches()`**: average vector khi 1 chunk bị split → 1
  chunk split → nhiều vectors → average = 1 vector cuối.
- **`replace=True`**: drop collection trước khi re-crawl → idempotent.
- **`_ensure_collection()`**: tạo collection nếu chưa có (cosine similarity).

### 3.3 `src/chat.py` — Chat orchestration

```
chat(sender_id, user_message)        ← non-streaming
chat_stream(sender_id, user_message) ← streaming

Flow:
  1. session.get_history()       ← Redis list (sliding window 10 msg)
  2. session.cache_get(question)  ← cache HIT → return ngay, không LLM
  3. rag.search(query)             ← top-3 chunks
  4. _build_messages(history, context, question)
  5. OpenAI GPT-4o-mini (stream=True/False)
  6. session.cache_set(question, reply)  ← cache MISS → lưu cho sau
  7. session.save_message()        ← refresh TTL + append history
  8. _should_capture_email()       ← RAG trống + reply chứa "email" + "để lại"
```

**Streaming**: Telegram có limit 4096 chars/message. Ta:
1. POST placeholder message → lấy `message_id`.
2. Stream tokens → sau mỗi 1.0s throttle → `editMessageText` (in-place).
3. Fallback nếu `sendMessage` fail trên edit.

### 3.4 `src/session.py` — Redis state

```
session:{sender_id}    → Redis list  │ TTL 1800s (sliding window 10 msg)
pending_crawl:{id}     → Redis string │ TTL 300s
pending_email:{id}      → Redis string │ TTL 300s
email:{id}              → Redis string │ TTL 30 ngày
cache:{normalized_q}   → Redis string │ TTL 3600s
```

Cache key là `_normalize_for_cache(text)` = lowercase + collapse
whitespace, nên "Công ty A" và "công ty a" share slot.

### 3.5 `src/entity.py` — URL + Company detection

```
extract_url(message)    ← regex + heuristic (TLD ≥ 2 chars, no @)
extract_email(message)  ← RFC-5322-ish regex
detect_company(message) ← OpenAI tool calling (no streaming, 5s timeout)
  │ Cue-word pre-filter: "công ty", "doanh nghiệp", "shop", brand, ...
  │ Skip LLM nếu không có cue word → cheap fast path
analyse(message)        ← extract_url + extract_email + detect_company
```

### 3.6 `src/api/telegram_webhook.py` — Webhook handler

```
POST /api/telegram/webhook
  │ verify Telegram HMAC signature
  ▼
analyse(message)  → EntityResult(url?, company?, email?)
  │
  ├─ URL trong message
  │   └─ crawl + index + RAG → reply_streaming()
  │
  ├─ pending_crawl + non-URL
  │   └─ "bạn có thể gửi URL công ty không?"
  │
  ├─ company detected + no URL
  │   └─ session.set_pending_crawl() + "URL công ty nào ạ?"
  │
  └─ email capture mode
      └─ extract_email() → session.save_email() → ack
```

---

## 4. Các quyết định thiết kế quan trọng

### 4.1 Vì sao không Celery?

Facebook Messenger yêu cầu HTTP 200 trong **< 5 giây** (webhook spec);
Telegram **không** có constraint đó. Với Telegram, ta có thể gọi LLM
đồng bộ và stream response để user thấy typing indicator. Celery chỉ
cần thiết khi webhook spec bắt buộc response nhanh — đây không phải
trường hợp đó.

### 4.2 Vì sao Redis cho session mà không PostgreSQL?

- **Latency**: Redis ~1ms, Postgres roundtrip ~5-20ms mỗi request.
- **TTL tự động**: session tự hết hạn sau 30 phút không cần cron.
- **Sliding window**: Redis list với `RPUSH + LTRIM` là pattern O(1)
  cho cửa sổ trượt; Postgres cần `DELETE` thủ công.
- **Redis được dùng chung** cho cache, pending markers, email —
  không cần thêm service.

### 4.3 Vì sao Qdrant mà không ChromaDB?

- **Production-ready**: Qdrant hỗ trợ HNSW, quantization, tenant
  isolation, Kubernetes-native.
- **REST + gRPC API**: monitor bằng Prometheus dễ hơn.
- **On-disk HNSW**: không cần toàn bộ vector trong RAM.

### 4.4 Vì sao `text-embedding-3-small` (1536 dims)?

OpenAI embedding model rẻ nhất, đủ tốt cho demo. `text-embedding-3-small`
(1536 dims) thay vì `ada-002` (1536 dims) vì model mới hơn và rẻ hơn.
Qdrant collection size phải khớp `rag_embedding_size=1536`.

### 4.5 Vì sao streaming + in-place edit?

Telegram có 2 lựa chọn:
1. **Gửi 1 message cuối**: user thấy "bot is typing..." nhưng không
   thấy reply cho đến khi hoàn tất (có thể 5-10 giây).
2. **Stream + edit in-place**: user thấy reply "chạy" real-time.

Ta chọn (2) với throttle 1.0s giữa các edit để tránh Telegram rate
limit. Với token nhanh → 2 edits; token chậm → 3-4 edits.

---

## 5. Playbook vận hành

### Development

```powershell
# Infra
docker compose up -d redis qdrant

# App
uvicorn src.main:app --reload

# Tests
python -m pytest tests/ -v
```

### Crawl một website

```powershell
python crawl_websites.py https://example.com --max-pages 10 --replace
```

### Cấu hình Telegram webhook

```powershell
# Set webhook URL với Telegram Bot API
curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://your-domain.com/api/telegram/webhook"
```

### Chạy với Docker Compose

```powershell
docker compose up --build
```

---

## 6. Câu hỏi phỏng vấn thường gặp

### Câu 1. Tại sao lại cần entity detection trước khi crawl?

> Không phải user nào cũng biết gửi URL ngay. Nhiều người hỏi
> "Công ty Stripe có gì?" trước. Ta dùng cue-word pre-filter
> ("công ty", "doanh nghiệp", brand...) để skip LLM call cho câu hỏi
> generic, và chỉ gọi OpenAI tool calling khi có dấu hiệu rõ ràng.
> Đây là pattern **avoid unnecessary LLM call** mà production system nào
> cũng nên áp dụng.

### Câu 2. LLM ảo giác (hallucination) được xử lý thế nào?

> Ba lớp phòng thủ:
> 1. **System prompt** yêu cầu LLM trả lời dựa trên "Knowledge Base",
>    và fallback phrase khi không tìm thấy.
> 2. **Citation `[n]`**: mỗi chunk được đánh số trong context, user
>    có thể tra nguồn.
> 3. **Email capture**: câu hỏi out-of-scope → xin email → human follow-up.

### Câu 3. Làm sao tránh Telegram rate limit khi stream?

> Telegram `editMessageText` giới hạn ~1 edit/giây cho cùng message.
> Ta throttle 1.0s giữa các edit (`_STREAM_EDIT_INTERVAL_SECONDS`).
> Mỗi edit chỉ gửi text mới — không re-send toàn bộ message.

### Câu 4. Cache hoạt động thế nào với streaming?

> Cache được check **trước khi** bất kỳ RAG hay LLM call nào.
> Cache hit → `session.save_message()` vẫn được gọi (để lịch sử hội
> thoại đúng) → return/yield cached reply → KHÔNG gọi LLM.
> Cache miss → chạy bình thường → `session.cache_set()` sau khi có reply.

### Câu 5. Nếu Redis chết thì sao?

> - `cache_get`/`cache_set` → `except RedisError` → log → return `None` /
>   swallow error. User vẫn nhận reply (chậm hơn vì không cache).
> - `session.get_history` → fallback `history = []` → hội thoại mất ngữ
>   cảnh nhưng bot vẫn trả lời được.
> - Redis cho Redis thực chất: dùng `redis.asyncio` (aio),
>   tất cả operations là `async` để không block event loop.

### Câu 6. Sự khác biệt giữa `chat()` và `chat_stream()`?

> - **`chat()`**: non-streaming, trả về string. Dùng cho fallback path
>   khi streaming fail hoặc khi latency không quan trọng.
> - **`chat_stream()`**: async generator, yield partial text. Webhook
>   dùng để show real-time typing indicator.
> - Cả hai chia code chung qua `_retrieve_context()`, `_build_messages()`,
>   `_persist_turn()` — không duplicate logic.

### Câu 7. Sẽ thay đổi gì cho production?

> - **Embedding model**: thay `text-embedding-3-small` bằng
>   `text-embedding-3-large` (3072 dims) hoặc open-source (BGE,
>   E5) để cải thiện retrieval quality.
> - **Qdrant**: bật HNSW `ef_construct=256`, quantization (scalar/Product)
>   để giảm RAM khi scale.
> - **Crawl4ai**: thêm `word_count_threshold`, `exclude_external_links`
>   để giảm noise.
> - **Celery**: nếu Telegram đổi spec yêu cầu <5s response, đẩy LLM
>   call vào Celery queue tương tự kiến trúc cũ.
> - **Observability**: thêm structured logging (JSON) + Prometheus metrics
>   cho cache hit rate, crawl success rate, LLM latency p95.
