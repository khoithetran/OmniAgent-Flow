# PROJECT IMPLEMENTATION PLAN: OMNIAGENT FLOW

Mục tiêu: Telegram chatbot RAG cho phép crawl bất kỳ website công ty nào và trả lời câu hỏi dựa trên nội dung website đó. Stack đơn giản: FastAPI + Redis + Qdrant + OpenAI + crawl4ai. Không Celery, không LangGraph, không HubSpot, không LangFuse, không PostgreSQL.

## Phase 1: Nền tảng đơn giản [x]

- [x] Task 1.1: Viết lại cấu trúc dự án gọn lại. Cấu hình `docker-compose.yml` với 3 services: app, redis, qdrant. Cập nhật `requirements.txt` với `crawl4ai` + `openai`.
- [x] Task 1.2: Xóa các file không cần thiết: `src/agents/`, `src/workers/`, `src/api/webhook.py`, `src/services/{hubspot,intent,observability,evaluation,conversation,telegram,ai,rag}_service.py`, `src/database.py`, `tests/test_*` cũ. Thêm `EXPLAINATION.md` vào `.gitignore`.
- [x] Task 1.3: Viết `src/session.py` (Redis sliding window 10 tin nhắn, TTL 1800s) + helper `set_pending_crawl` / `pop_pending_crawl` / `peek_pending_crawl` (TTL 5 phút, key `pending_crawl:{sender_id}`).
- [x] Task 1.4: Rà soát `src/config.py`, giữ lại chỉ các biến cần thiết (APP_*, REDIS_*, SESSION_*, PENDING_CRAWL_*, OPENAI_*, QDRANT_*, RAG_*, TELEGRAM_*). Cập nhật `.env` đồng bộ.

## Phase 2: Crawler + RAG [x]

- [x] Task 2.1: Viết `src/crawler.py` dùng crawl4ai để lấy markdown từ website. Hỗ trợ cả 1 URL và nhiều URL (batch). Dùng `fit_markdown` để có output gọn cho LLM. Bao gồm helper `chunk_markdown()` cho việc tách chunk (overlap 50-100 chars), `CrawlResult` dataclass với `success`/`error_message`, retry 2 lần với backoff, fail isolation giữa các URL trong batch. Thêm `discover_urls()` đọc sitemap.xml (hỗ trợ sitemap index lồng nhau + `Sitemap:` directive trong robots.txt), `filter_same_domain()` chỉ giữ URL cùng host, `crawl_full_website()` kết hợp discover + filter + cap `max_pages`. Test thực tế: example.com + iana.org → cả 2 OK, markdown extracted, chunking hoạt động đúng. Crawl `artificialanalysis.ai` với `max_pages=5` → 5 pages nội bộ OK, không lẫn link Twitter/LinkedIn.
- [x] Task 2.2: Viết `src/rag.py` để chunk markdown (1000 tokens/chunk), embed bằng `text-embedding-3-small`, lưu vào Qdrant. Hỗ trợ hàm `search(query, top_k=3)`. Bao gồm: `init_qdrant()` / `init_openai()` khởi tạo client global trong FastAPI lifespan, `_ensure_collection()` tạo collection với vector size 1536, `index_markdown()` cho 1 page, `index_crawl_results()` cho batch (tự reset collection khi `replace=True`), `search()` trả về `RagSearchResult` với score + metadata, `format_context()` ghép context cho LLM (giới hạn 6000 chars). Có helper `_split_oversize_text()` để tránh vượt giới hạn 8192 tokens/input của OpenAI, và `_embed_in_batches()` để average vector khi chunk bị split. Test thực tế: crawl 3 pages artificialanalysis.ai → 76 chunks → search "What AI models are available?" → 3 hits relevance score 0.62-0.63, nội dung đúng về AI models/providers.
- [x] Task 2.3: Test thực tế với `fastapi.tiangolo.com` (GitHub repo không có sitemap.xml truyền thống). Crawl 10 pages từ sitemap → 203 chunks → test 4 queries: "What is FastAPI?" (score 0.636, trỏ về trang chính với quote giới thiệu), "How to install FastAPI?" (score 0.604, trỏ về phần install/deployment), "What are the main features?" (score 0.305, trỏ về features page), "How to handle authentication?" (score 0.352, trỏ về auth section). Kết luận: RAG pipeline hoạt động đầu cuối, retrieval quality cao cho query cụ thể, có thể cải thiện cho query ngắn chung chung.

## Phase 3: Chatbot [ ]

- [x] Task 3.1: Viết `src/chat.py` gồm 1 hàm `chat(sender_id, user_message)`: lấy session → search RAG → gọi OpenAI → save session → trả reply. Bao gồm `SYSTEM_PROMPT` hướng dẫn LLM trả lời tiếng Việt có dấu, có citation [n], fallback khi không tìm thấy, hỏi email khi out-of-scope. Helper `_sanitise_history()` filter entries không hợp lệ, `_call_openai()` wrap OpenAI với error handling graceful.
- [x] Task 3.2: Viết lại `src/main.py` gồm: Telegram webhook (GET verify + POST nhận) qua `src/api/telegram_webhook.py`, endpoint admin `POST /api/crawl` (crawl + index), `DELETE /api/crawl` (clear KB), `GET /api/crawl/status` (KB stats), healthcheck. Lifespan khởi tạo Redis + Qdrant + OpenAI clients. Pydantic models `CrawlRequest`/`CrawlResponse`/`CrawlStatusResponse`. Thêm `TELEGRAM_TIMEOUT_SECONDS` vào config.
- [x] Task 3.3: Entity detection + URL confirmation flow:
  - Tạo `src/entity.py` với `extract_url()` (regex + heuristic validate, exclude email) và `detect_company()` (OpenAI tool calling với cue-word pre-filter để skip LLM call cho generic question, Vietnamese diacritics normalized).
  - Cue words bao gồm cả có dấu và không dấu: "công ty"/"cong ty", "doanh nghiệp", "shop", "cửa hàng", "brand", "thương hiệu", "agency", "startup", "company", "về công ty", "dịch vụ", "sản phẩm"...
  - Sửa `telegram_webhook.py` thêm flow ưu tiên: (1) URL trong message → crawl + index + RAG, (2) pending_crawl mode + non-URL → re-ask, (3) company detected + no URL → set pending + ask, (4) generic → chat pipeline.
  - Tự động split Telegram message chunks (limit 4000 chars) vì LLM reply có thể dài hơn 4096 limit của Telegram.
- [x] Task 3.4: Test end-to-end toàn diện với 4 test case: (1) "Cong ty Stripe co gi?" → awaiting_url + set pending, (2) "Day ne https://stripe.com" → crawl + index + reply, (3) "Thoi tiet hom nay" → chat pipeline bình thường, (4) User trong pending mode + non-URL → re-ask. Tất cả PASS.

## Phase 4: Cải thiện UX [ ]

- [x] Task 4.1: Streaming response cho OpenAI + typing indicator cho Telegram:
  - Thêm `chat_stream()` async generator trong `src/chat.py` (parallel với `chat()` cũ, không phá API).
  - Tách helper: `_retrieve_context()`, `_build_messages()`, `_extract_delta()`, `_persist_turn()` để share code giữa 2 hàm.
  - Throttle 1.0s giữa các edit (`_STREAM_EDIT_INTERVAL_SECONDS`) để tránh Telegram rate limit.
  - Webhook post placeholder message, edit in-place khi stream yield. Có fallback nếu sendMessage fail.
  - Test 2 case: stream nhanh (delay 0.05s/tokens) → 2 edits, stream chậm (0.4s/tokens) → 3 edits nhờ throttle. Cả 2 đều hiển thị đúng text cuối cùng.
- [x] Task 4.2: Fallback khi RAG không tìm thấy: bot xin lỗi và hỏi khách để lại email. Lưu email vào Redis đơn giản.
  - Thêm `entity.extract_email()` với regex RFC-5322-ish.
  - Thêm `session.set_pending_email` / `pop_pending_email` / `peek_pending_email` (TTL 5 phút) + `session.save_email` / `get_email` (TTL 30 ngày).
  - `chat._should_capture_email()` quyết định: RAG trống + LLM reply chứa "email" + "để lại" → set pending_email.
  - Webhook flow: kiểm tra pending_email đầu tiên → nếu có email trong message, save + ack + pop. Nếu không, nhắc lại user.
  - 3 test case pass: fallback sets marker, non-email keeps marker, valid email gets captured.
- [x] Task 4.3: Cache: nếu câu hỏi giống trong 1 giờ, trả cache luôn, không gọi LLM.

## Phase 5: Test + Demo [ ]

- [x] Task 5.1: Viết pytest cho `crawler.py`, `rag.py`, `chat.py`, `session.py`. Mock OpenAI + crawl4ai trong test.
- [ ] Task 5.2: Crawl 2-3 website thật (agency/marketing SaaS). Viết script `crawl_websites.py` chạy 1 lần để setup.
- [ ] Task 5.3: Cập nhật `README.md` + `EXPLAINATION.md` mô tả kiến trúc mới.
