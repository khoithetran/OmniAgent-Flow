---
title: OmniAgent Flow
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
suggested_storage: large
pinned: false
---

# 🤖 OmniAgent Flow

**OmniAgent Flow** là ứng dụng RAG Chatbot thông minh cho phép tra cứu và trả lời câu hỏi chính xác dựa trên **dữ liệu Website (Crawl)** hoặc **Tệp văn bản tải lên (PDF, Word, Excel, Markdown)**.

---

## 🌟 Tính năng Chính

- 🤖 **AI Agent (ReAct Tool Loop)**: LLM tự suy luận và kích hoạt các công cụ (`search_knowledge_base`, `get_document_metadata`, `calculate`, `get_current_time`).
- 📊 **RAGAS Evaluation Dashboard**: Đánh giá định lượng chất lượng RAG qua 4 chỉ số `Faithfulness`, `Answer Relevance`, `Context Precision`, `Context Recall`.
- 🌐 **Tra cứu Website**: Nhập URL bất kỳ để cào dữ liệu và tạo Knowledge Base tự động.
- 📁 **Xử lý Tài liệu Đa định dạng**: Hỗ trợ đọc và trích xuất dữ liệu từ tệp **PDF, Word (.docx), Excel (.xlsx), Markdown (.md), Text (.txt)**.
- 🧩 **4 Chiến lược Phân mảnh (Chunking)**: Hỗ trợ `Fixed-size`, `Recursive`, `Parent-Child (Phân tầng)` và `Tokenizer-aware`.
- 🔍 **Tìm kiếm Lai 2 Giai đoạn (Two-Stage Retrieval)**: Kết hợp Tìm kiếm ngữ nghĩa (Dense Vector) + Tìm kiếm từ khóa (BM25 Sparse) + Re-ranking với Cross-Encoder.
- 💬 **Giao diện Web mượt mà**: Giao diện Gradio hỗ trợ phản hồi dạng Streaming realtime và Trích dẫn nguồn (Citations).

---

## 🛠️ Công nghệ Sử dụng (Tech Stack)

| Tầng | Công nghệ / Thư viện |
|---|---|
| **Giao diện (UI)** | Gradio 5.x (Streaming SSE, Layout 2 Hàng ngang, RAGAS Dashboard) |
| **Mô hình LLM & Agent** | Anthropic Claude API (`claude-3-5-sonnet`), ReAct Pattern Tool Loop |
| **RAG Evaluation** | RAGAS Framework (Faithfulness, Answer Relevance, Context Precision/Recall) |
| **Vector DB** | Qdrant Vector Database + In-Memory Fallback |
| **Đọc Tài liệu** | PyMuPDF (PDF), python-docx (Word), openpyxl (Excel), BeautifulSoup4 |
| **Retrieval Optimization** | Rank-BM25 (Keyword Search), Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`), Tiktoken |
| **Backend & Cache** | Python 3.12, FastAPI, Redis, Docker |

---

## 🚀 Quick Start (Hướng dẫn Nhanh)

```powershell
# 1. Copy tệp cấu hình mẫu
Copy-Item .env.example .env

# 2. Cấu hình API Key trong tệp .env:
#    ANTHROPIC_API_KEY=sk-ant-...
#    OPENAI_API_KEY=sk-...

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 4. Khởi chạy giao diện Gradio Web UI
python app_gradio.py

# 5. Mở trình duyệt truy cập: http://127.0.0.1:7860
```
