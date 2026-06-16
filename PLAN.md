# PROJECT IMPLEMENTATION PLAN: OMNIAGENT FLOW

## Phase 1: Core Backend & Architecture [ ]

- [x] Task 1.1: Khởi tạo cấu trúc thư mục dự án và file `Dockerfile`, `docker-compose.yml`.
- [x] Task 1.2: Thiết lập FastAPI Webhook Base xác thực token và tiếp nhận payload.
- [x] Task 1.3: Cấu hình Redis Connection và hàm helper quản lý Session History (TTL 30p).
- [x] Task 1.4: Tích hợp Celery + Redis Broker để tạo kiến trúc hàng đợi Message Queue xử lý tin nhắn bất đồng bộ.

## Phase 2: Agentic AI Core & Advanced RAG Pipeline [ ]

- [x] Task 2.1: Xây dựng luồng xử lý Agentic AI bằng **LangGraph**, cấu hình trạng thái (State) và phân nhánh hành động của Agent.
- [x] Task 2.2: Tích hợp tính năng **Structured Outputs** (OpenAI/Anthropic) kết hợp **Pydantic** để phân tách chính xác Intent và Extract Metadata của khách hàng sang định dạng JSON.
- [x] Task 2.3: Thiết lập Qdrant Vector DB local và xây dựng Pipeline **Advanced RAG**:
- Triển khai *Hybrid Search* (Dense Retrieval + BM25).
- Tích hợp *Reranking* (Cohere Rerank hoặc BGE-Reranker) để tối ưu hóa tri thức trích xuất.

## Phase 3: Integration & Conversation Insights [x]

- [x] Task 3.1: Thiết lập cơ sở dữ liệu **PostgreSQL** để lưu trữ lịch sử hội thoại, Metadata và các thuộc tính Intent đã phân tách từ Phase 2.
- [x] Task 3.2: Kết nối Backend với HubSpot Developer API để tự động đồng bộ và cập nhật dữ liệu Lead dựa trên Insight thu được từ cuộc trò chuyện.
- [x] Task 3.3: Viết webhook đẩy thông báo sự kiện Realtime qua Telegram Bot.

## Phase 4: AI Observability & Monitoring [x]

- [x] Task 4.1: Tích hợp nền tảng AI Observability **LangFuse** để giám sát hệ thống LLM:
- Theo dõi lượng Token tiêu thụ, chi phí API và độ trễ (Latency) của từng bước trong LangGraph.
- Thiết lập bộ tiêu chí *LLM Evaluation* để đánh giá tự động độ trung thực (Faithfulness) và mức độ liên quan (Answer Relevance) của câu trả lời.
- [x] Task 4.2: Viết Unit Test cho hệ thống API, Agent và Worker.
- [x] Task 4.3: Kết nối dữ liệu PostgreSQL với Looker Studio để vẽ Dashboard báo cáo về phân tích hành vi người dùng (Conversation Insights).
