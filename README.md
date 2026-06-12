# OmniAgent Flow

He thong AI Agent da kenh va tu dong hoa cham soc khach hang doanh nghiep.
Kien truc bat dong bo tiep nhan webhook, dua tin nhan vao Celery queue, quan ly
session bang Redis va san sang mo rong voi RAG, CRM cung cac kenh thong bao.

## Tech Stack

- Python 3.10+
- FastAPI
- Celery + Redis
- PostgreSQL
- Docker Compose
- Qdrant/ChromaDB (planned)

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

Lo trinh trien khai chi tiet nam trong [PLAN.md](PLAN.md).
