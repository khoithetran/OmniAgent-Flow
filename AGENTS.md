# OMNIAGENT FLOW - SYSTEM RULES & INSTRUCTIONS

## 1. Project Overview

Hệ thống AI Agent đa kênh và Tự động hóa Chăm sóc Khách hàng Doanh nghiệp.
Kiến trúc bất đồng bộ (Asynchronous) xử lý dữ liệu từ Webhook đến Message Queue, tích hợp RAG và CRM.

## 2. Tech Stack & Constraints

- Language: Python >= 3.10 (Bắt buộc dùng Type Hints cho mọi function).
- Framework: FastAPI (Async def cho toàn bộ endpoints).
- Message Queue: Celery + Redis (Xử lý tác vụ gọi LLM ngầm).
- Vector DB: Qdrant / ChromaDB (Dùng cho RAG Pipeline).
- Database: PostgreSQL (Lưu lịch sử hội thoại vĩnh viễn), Redis (Lưu session tạm thời).
- Containerization: Docker & Docker Compose.

## 3. Strict Development Rules (Ràng buộc cứng)

- KHÔNG ĐƯỢC lưu trạng thái hội thoại (Session) trong biến toàn cục (In-memory dict). Bắt buộc dùng Redis với TTL = 1800 giây.
- KHÔNG ĐƯỢC gọi API của OpenAI/Claude trực tiếp trong hàm xử lý Webhook. Webhook nhận tin nhắn phải lập tức push vào Celery Queue và trả về `HTTP 200 OK` cho client < 500ms.
- KHÔNG ĐƯỢC tự bịa đặt cấu trúc API. Tuân thủ nghiêm ngặt chuẩn RESTful API.
- Error Handling: Mọi block try-except bắt buộc phải log lỗi chi tiết qua thư viện `loguru`, không dùng `print()`.

## 4. Operation Commands

- Chạy môi trường Dev: `uvicorn src.main:app --reload`
- Chạy Celery Worker: `celery -A src.workers.tasks worker --loglevel=info`
- Build hệ thống: `docker-compose up --build`
