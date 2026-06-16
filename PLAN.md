# PROJECT IMPLEMENTATION PLAN: OMNIAGENT FLOW

Mục tiêu: Telegram chatbot RAG cho phép crawl bất kỳ website công ty nào và trả lời câu hỏi dựa trên nội dung website đó. Stack đơn giản: FastAPI + Redis + Qdrant + OpenAI + crawl4ai. Không Celery, không LangGraph, không HubSpot, không LangFuse, không PostgreSQL.

## Phase 1: Nền tảng đơn giản [x]

- [x] Task 1.1: Viết lại cấu trúc dự án gọn lại. Cấu hình `docker-compose.yml` với 3 services: app, redis, qdrant. Cập nhật `requirements.txt` với `crawl4ai` + `openai`.
- [x] Task 1.2: Xóa các file không cần thiết: `src/agents/`, `src/workers/`, `src/api/webhook.py`, `src/services/{hubspot,intent,observability,evaluation,conversation,telegram,ai,rag}_service.py`, `src/database.py`, `tests/test_*` cũ. Thêm `EXPLAINATION.md` vào `.gitignore`.
- [x] Task 1.3: Viết `src/session.py` (Redis sliding window 10 tin nhắn, TTL 1800s) + helper `set_pending_crawl` / `pop_pending_crawl` / `peek_pending_crawl` (TTL 5 phút, key `pending_crawl:{sender_id}`).
- [x] Task 1.4: Rà soát `src/config.py`, giữ lại chỉ các biến cần thiết (APP_*, REDIS_*, SESSION_*, PENDING_CRAWL_*, OPENAI_*, QDRANT_*, RAG_*, TELEGRAM_*). Cập nhật `.env` đồng bộ.

## Phase 2: Crawler + RAG [ ]

- [ ] Task 2.1: Viết `src/crawler.py` dùng crawl4ai để lấy markdown từ website. Hỗ trợ cả 1 URL và nhiều URL (batch). Dùng `fit_markdown` để có output gọn cho LLM.
- [ ] Task 2.2: Viết `src/rag.py` để chunk markdown (1000 tokens/chunk), embed bằng `text-embedding-3-small`, lưu vào Qdrant. Hỗ trợ hàm `search(query, top_k=3)`.
- [ ] Task 2.3: Test thực tế: crawl github.com, hỏi "What is FastAPI?" → verify RAG tìm đúng chunk.

## Phase 3: Chatbot [ ]

- [ ] Task 3.1: Viết `src/chat.py` gồm 1 hàm `chat(sender_id, user_message)`: lấy session → search RAG → gọi OpenAI → save session → trả reply.
- [ ] Task 3.2: Viết lại `src/main.py` gồm: Telegram webhook (GET verify + POST nhận), endpoint admin `POST /api/crawl` và `DELETE /api/crawl`, healthcheck.
- [ ] Task 3.3: Entity detection + URL confirmation flow:
  - Nhận diện tên công ty/tổ chức qua OpenAI Function Calling khi user nhắn về công ty nhưng không cung cấp URL.
  - Bot reply xác nhận: "Bạn có đang hỏi về [tên công ty] không? Vui lòng gửi URL website để tôi tìm hiểu thêm."
  - Lưu trạng thái chờ URL trong Redis (TTL 5 phút, key `pending_crawl:{sender_id}`).
  - Khi user gửi URL → crawl ngay + RAG search + trả lời.
  - Khi user nhắn khác trong lúc chờ URL → ignore, vẫn chờ.
  - Hết TTL → xóa trạng thái.
- [ ] Task 3.4: Test end-to-end trên Telegram: crawl website → nhắn câu hỏi → bot reply đúng nội dung.

## Phase 4: Cải thiện UX [ ]

- [ ] Task 4.1: Streaming response cho OpenAI + typing indicator cho Telegram. Người dùng thấy bot "đang gõ" rồi từng chữ hiện ra.
- [ ] Task 4.2: Fallback khi RAG không tìm thấy: bot xin lỗi và hỏi khách để lại email. Lưu email vào Redis đơn giản.
- [ ] Task 4.3: Cache: nếu câu hỏi giống trong 1 giờ, trả cache luôn, không gọi LLM.

## Phase 5: Test + Demo [ ]

- [ ] Task 5.1: Viết pytest cho `crawler.py`, `rag.py`, `chat.py`, `session.py`. Mock OpenAI + crawl4ai trong test.
- [ ] Task 5.2: Crawl 2-3 website thật (agency/marketing SaaS). Viết script `crawl_websites.py` chạy 1 lần để setup.
- [ ] Task 5.3: Cập nhật `README.md` + `EXPLAINATION.md` mô tả kiến trúc mới.
