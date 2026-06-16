# OmniAgent Flow

He thong AI Agent da kenh va tu dong hoa cham soc khach hang doanh nghiep.
Kien truc bat dong bo tiep nhan webhook, dua tin nhan vao Celery queue, quan ly
session bang Redis, su dung LangGraph cho bo phan loai intent, hybrid RAG
(Qdrant + BM25 + reranker) de tra loi, dong bo lead sang HubSpot, day canh bao
realtime qua Telegram Bot, va gui trace/evaluation len LangFuse.

## Tech Stack

- Python 3.10+
- FastAPI
- Celery + Redis
- PostgreSQL
- Docker Compose
- LangGraph (state machine)
- OpenAI Structured Outputs
- Qdrant (hybrid search + BM25 + reranker)
- HubSpot CRM API
- Telegram Bot API
- LangFuse (traces + evaluation)
- pytest + pytest-asyncio

## Quick Start

1. Tao file moi truong:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Cap nhat cac gia tri nhay cam trong `.env`:

   - Dat `WEBHOOK_VERIFY_TOKEN` thanh token rieng.
   - Dat `POSTGRES_USER` va `POSTGRES_DB` theo moi truong.
   - Tao `POSTGRES_PASSWORD` dai, ngau nhien bang password manager hoac secret
     manager. Khong commit file `.env`.
   - (Tuy chon) Dien `OPENAI_API_KEY`, `HUBSPOT_ACCESS_TOKEN`,
     `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, va `LANGFUSE_*` de bat
     tung tinh nang.

3. Khoi dong he thong:

   ```powershell
   docker compose up --build
   ```

FastAPI se chay tai `http://localhost:8000`.

Docker Compose se dung ngay neu thieu mot trong cac bien `POSTGRES_USER`,
`POSTGRES_PASSWORD` hoac `POSTGRES_DB`. Trong production, cac gia tri nay can
duoc cap tu secret manager cua nen tang trien khai.

## Development

```powershell
uvicorn src.main:app --reload
celery -A src.workers.tasks worker --loglevel=info
```

## Testing

```powershell
python -m pytest tests/ -v
```

Bo test gom 45 test (webhook contract, session, RAG, agent, intent,
HubSpot, Telegram, conversation, observability, evaluation, Celery
task pipeline, app lifespan).

## Documentation

- [PLAN.md](PLAN.md) - Lo trinh trien khai chi tiet theo phase.
- [EXPLAINATION.md](EXPLAINATION.md) - Tai lieu on tap phong van.
- [docs/looker_studio.md](docs/looker_studio.md) - Huong dan ket noi
  Looker Studio voi cac SQL view trong `migrations/0010_looker_views.sql`.
