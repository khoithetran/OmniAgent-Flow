# PROJECT IMPLEMENTATION PLAN: OMNIAGENT FLOW

## Phase 1: Core Backend & Architecture [ ]

- [x] Task 1.1: Khởi tạo cấu trúc thư mục dự án và file `Dockerfile`, `docker-compose.yml`.
- [x] Task 1.2: Thiết lập FastAPI Webhook Base xác thực token và tiếp nhận payload.
- [x] Task 1.3: Cấu hình Redis Connection và hàm helper quản lý Session History (TTL 30p).
- [x] Task 1.4: Tích hợp Celery + Redis Broker để tạo kiến trúc hàng đợi Message Queue.

## Phase 2: AI Core & RAG Pipeline [ ]

- [x] Task 2.1: Viết Celery Worker kết nối OpenAI/Anthropic API xử lý tin nhắn bất đồng bộ.
- [ ] Task 2.2: Thiết kế System Prompt nâng cao để phân tách Intent của khách hàng sang định dạng JSON.
- [ ] Task 2.3: Thiết lập Qdrant Vector DB local và viết Pipeline Semantic Search trích xuất tri thức.

## Phase 3: Integration & Automation [ ]

- [ ] Task 3.1: Kết nối Backend FastAPI với HubSpot Developer API để đồng bộ dữ liệu Lead.
- [ ] Task 3.2: Viết webhook đẩy thông báo sự kiện Realtime qua Telegram Bot.

## Phase 4: Production Ready & Monitoring [ ]

- [ ] Task 4.1: Viết Unit Test cho hệ thống API và Worker.
- [ ] Task 4.2: Kết nối dữ liệu PostgreSQL/Google Sheets với Looker Studio để vẽ Dashboard báo cáo.
