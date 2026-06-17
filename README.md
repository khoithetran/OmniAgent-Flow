# OmniAgent Flow

Telegram chatbot RAG cho phép crawl bất kỳ website công ty nào và trả
lời câu hỏi dựa trên nội dung website đó.

## Kiến trúc đơn giản

```
Telegram ──► FastAPI webhook
                   │
            ┌──────┴──────┐
            ▼             ▼
         Redis         Qdrant
      (session +        (RAG
       pending)          KB)
            ▲
            │
         OpenAI
   (embedding + LLM)
```

## Tech Stack

| Tầng          | Công cụ                        |
|--------------|-------------------------------|
| HTTP API      | FastAPI (async)                |
| Session/Cache | Redis (list + TTL)             |
| Vector DB     | Qdrant (`text-embedding-3-small`) |
| LLM + Embedding | OpenAI GPT-4o-mini / text-embedding-3-small |
| Crawler       | crawl4ai (Chromium headless)   |
| Container     | Docker + Docker Compose        |
| Testing       | pytest + pytest-asyncio (96 tests) |

## Cách hoạt động

1. **User nhắn URL công ty** → bot crawl toàn bộ site qua sitemap, chunk, embed, index vào Qdrant.
2. **User hỏi về công ty đó** → RAG tìm top-3 chunks, GPT-4o-mini trả lời tiếng Việt có citation.
3. **User hỏi câu đã hỏi** → cache Redis trả ngay, không gọi LLM.
4. **RAG không tìm thấy** → bot xin email, lưu Redis 30 ngày.
5. **Không có URL** → entity detection (OpenAI tool calling) hỏi user gửi URL.

## Quick Start

```powershell
# 1. Copy env
Copy-Item .env.example .env

# 2. Fill in .env:
#    OPENAI_API_KEY=sk-...
#    TELEGRAM_BOT_TOKEN=... (from @BotFather)
#    REDIS_HOST=localhost
#    QDRANT_HOST=localhost

# 3. Start infra
docker compose up -d redis qdrant

# 4. Run app
uvicorn src.main:app --reload

# 5. (optional) Crawl a website and index it
python crawl_websites.py https://example.com --max-pages 10 --replace

# 6. Run tests
python -m pytest tests/ -v
```

## API Endpoints

| Method | Path              | Mô tả                            |
|--------|-------------------|----------------------------------|
| GET    | `/`               | Healthcheck                      |
| GET    | `/api/telegram/webhook` | Telegram webhook verify    |
| POST   | `/api/telegram/webhook` | Nhận tin nhắn Telegram      |
| POST   | `/api/crawl`     | Crawl + index website (admin)     |
| DELETE | `/api/crawl`      | Xoá toàn bộ knowledge base       |
| GET    | `/api/crawl/status` | KB stats (pages, chunks)      |

## Environment Variables

| Variable               | Mặc định       | Mô tả                   |
|-----------------------|----------------|-------------------------|
| `OPENAI_API_KEY`       | —              | Bắt buộc                |
| `TELEGRAM_BOT_TOKEN`   | —              | Từ @BotFather          |
| `REDIS_HOST`           | `localhost`    |                         |
| `QDRANT_HOST`          | `localhost`    |                         |
| `SESSION_TTL_SECONDS`  | `1800`        | 30 phút, sliding window |
| `CACHE_TTL_SECONDS`    | `3600`        | 1 giờ, same-question cache |
| `RAG_TOP_K`            | `3`           | Số chunks trả về        |

## Project Structure

```
src/
  main.py              - FastAPI app + lifespan (Redis, Qdrant, OpenAI)
  config.py            - Pydantic settings (.env driven)
  session.py           - Redis: chat history, pending markers, LLM cache
  crawler.py           - crawl4ai: sitemap discovery, crawl, chunk
  rag.py               - Qdrant: index, search, format_context
  chat.py              - Chat orchestration: RAG + LLM + cache + session
  entity.py             - URL extraction, email, company detection (OpenAI)
  api/
    telegram_webhook.py - Webhook handler + entity detection + Telegram reply

crawl_websites.py      - One-shot crawl+index script
tests/
  conftest.py          - Fixtures: mock_redis, mock_settings, cache clear
  test_session.py      - 20 tests: cache, pending, email, history
  test_crawler.py       - 25 tests: chunking, sitemap, filtering
  test_rag.py           - 15 tests: split, upsert, ensure, format
  test_chat.py          - 36 tests: helpers, chat(), chat_stream()
```

## Documentation

- [PLAN.md](PLAN.md) - Lộ trình triển khai chi tiết theo phase.
- [EXPLAINATION.md](EXPLAINATION.md) - Tài liệu ôn tập phỏng vấn.
