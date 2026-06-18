# OmniAgent Flow

RAG chatbot cho phép crawl bất kỳ website công ty nào và trả lời câu
hỏi dựa trên nội dung website đó. Có 2 giao diện:

- **Gradio web UI** (mặc định, `python app_gradio.py`)
- **Telegram bot** (legacy, đóng băng, chạy qua `uvicorn src.main:app`)

## Kiến trúc đơn giản

```
[Gradio / Telegram]
        │
        ▼
   FastAPI / Gradio
        │
   ┌────┴────┐
   ▼         ▼
 Redis     Qdrant
(session + (RAG KB)
 cache)
   ▲         │
   │         ▼
   └───── OpenAI
       (embedding + LLM)
```

## Tech Stack

| Tầng          | Công cụ                        |
|--------------|-------------------------------|
| Web UI        | Gradio 6.x (chat interface + model selector) |
| HTTP API      | FastAPI (Telegram webhook, admin) |
| Session/Cache | Redis (list + TTL)             |
| Vector DB     | Qdrant (`text-embedding-3-small`) |
| LLM + Embedding | OpenAI (gpt-4o-mini, gpt-4o, o4-mini, gpt-4o-realtime) |
| Crawler       | crawl4ai (Chromium headless)   |
| Container     | Docker + Docker Compose        |
| Testing       | pytest + pytest-asyncio (96 tests) |

## Cách hoạt động (Gradio)

1. **Chưa có tài liệu**: chat dùng general LLM, trả lời câu hỏi thông thường.
2. **User nhập URL + bấm Fetch**: hệ thống crawl toàn bộ site, chunk, embed, index vào Qdrant.
3. **Đã có tài liệu**: chat dùng RAG, chỉ trả lời từ tài liệu. Không tìm thấy → "Không tìm thấy thông tin này trong tài liệu." (KHÔNG bịa).
4. **Bấm X trên panel** → xóa tài liệu, quay về chế độ general.
5. **Model selector** (4 button): chọn `gpt-4o-mini`, `gpt-4o`, `o4-mini`, hoặc `gpt-4o-realtime` cho câu hỏi tiếp theo.

## Quick Start

```powershell
# 1. Copy env
Copy-Item .env.example .env

# 2. Fill in .env:
#    OPENAI_API_KEY=sk-...
#    REDIS_HOST=localhost
#    QDRANT_HOST=localhost

# 3. Start infra
docker compose up -d redis qdrant

# 4. Run Gradio UI (mặc định port 7860)
python app_gradio.py

# 5. Mở browser: http://127.0.0.1:7860

# 6. (optional) Telegram bot — legacy, đóng băng
uvicorn src.main:app --port 8000

# 7. Run tests
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
  chat.py              - Chat orchestration: chat(), chat_stream(),
                          chat_general_stream(), chat_rag_stream()
  entity.py             - URL extraction, email, company detection
  api/
    telegram_webhook.py - Telegram webhook handler (legacy, đóng băng)

app_gradio.py          - Gradio web UI (mặc định, port 7860)
crawl_websites.py      - One-shot crawl+index script
tests/
  conftest.py          - Fixtures: mock_redis, mock_settings, cache clear
  test_session.py      - 20 tests: cache, pending, email, history
  test_crawler.py       - 25 tests: chunking, sitemap, filtering
  test_rag.py           - 15 tests: split, upsert, ensure, format
  test_chat.py          - 36 tests: helpers, chat(), chat_stream()
test_app_gradio_smoke.py - 5 smoke tests for Gradio handlers
```

## Documentation

- [PLAN.md](PLAN.md) - Lộ trình triển khai chi tiết theo phase.
- [EXPLAINATION.md](EXPLAINATION.md) - Tài liệu ôn tập phỏng vấn.
