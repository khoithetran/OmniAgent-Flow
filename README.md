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

# OmniAgent Flow

**OmniAgent Flow** là hệ thống Enterprise RAG Chatbot đa định dạng, cho phép crawl dữ liệu website hoặc tải lên các tệp văn bản doanh nghiệp (PDF, Word, Excel, Markdown) và tra cứu thông tin chính xác bằng kiến trúc **Two-Stage Retrieval** nâng cao.

Giao diện chính: **Gradio Web UI** (`python app_gradio.py`).

---

## 🏗️ Kiến trúc Hệ thống RAG (Two-Stage Retrieval)

```
[User Query] / [Document Files / Web URLs]
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Document Parsing & Structure Extraction             │
  │    • PDF (PyMuPDF)      • Word (python-docx)           │
  │    • Excel (openpyxl)   • Web (crawl4ai / BeautifulSoup) │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Multi-Strategy Chunking                             │
  │    • Fixed-size         • Recursive Character          │
  │    • Parent-Child       • Tokenizer-aware (tiktoken)   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Stage 1: Hybrid Retrieval (High Recall)             │
  │    • Dense Search  : OpenAI text-embedding-3-small     │
  │    • Sparse Search : BM25 Keyword Search               │
  │    • Rank Fusion   : Reciprocal Rank Fusion (RRF)      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Stage 2: Reranking (High Precision)                 │
  │    • Cross-Encoder (ms-marco-MiniLM-L-6-v2)           │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 5. Generation & Grounded Citation                      │
  │    • OpenAI LLM (gpt-4o-mini / gpt-4o / o4-mini)       │
  │    • RAG Prompt Grounding + Source Citations [n]       │
  └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack & Kỹ thuật Nổi bật

| Tầng | Công cụ / Kỹ thuật |
|---|---|
| **Web UI** | Gradio 5.x (Streaming SSE, Model Selector, Document Upload & Strategy Selector) |
| **Document Parsing** | PyMuPDF (PDF), python-docx (Word), openpyxl (Excel), BeautifulSoup4 |
| **Chunking Engine** | 4 chiến lược: Fixed-size, Recursive, Parent-Child (Hierarchical), Tokenizer-aware (`tiktoken`) |
| **Vector DB** | Qdrant (`text-embedding-3-small`, 1536 dims) + In-Memory Fallback |
| **Sparse Search** | Rank-BM25 (Exact keyword match) + Reciprocal Rank Fusion (RRF) |
| **Reranking** | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **LLM Inference** | OpenAI API (`gpt-4o-mini`, `gpt-4o`, `o4-mini`) |
| **HTTP API** | FastAPI (Lifespan resource manager) |
| **Session / Cache** | Redis (sliding window session + same-question cache) |

---

## ✨ Tính năng Nổi bật

1. **Đọc tài liệu đa định dạng (Document Intelligence)**:
   * **PDF**: Trích xuất text theo thứ tự đọc tự nhiên bằng PyMuPDF.
   * **Word (`.docx`)**: Duyệt cây XML để giữ nguyên thứ tự đoạn văn và định dạng bảng.
   * **Excel (`.xlsx`)**: Ánh xạ tiêu đề cột với ô dữ liệu `Header: Value` giúp LLM hiểu bảng tính.
2. **4 Chiến lược Phân mảnh (Chunking Strategies)**:
   * **Fixed-size**: Cắt cố định theo số ký tự.
   * **Recursive**: Ưu tiên cắt theo đoạn văn (`\n\n`) -> dòng (`\n`) -> câu (`. `).
   * **Parent-Child**: 2 tầng chunk (Child 200 chars để search chính xác, Parent 1000 chars cho LLM đủ ngữ cảnh).
   * **Tokenizer-aware**: Cắt chính xác theo BPE token (`cl100k_base`) tránh tràn Context Window.
3. **Tìm kiếm 2 Giai đoạn (Two-Stage Retrieval)**:
   * **Hybrid Search**: Kết hợp Vector Search ngữ nghĩa + BM25 tìm chính xác tên riêng/mã số.
   * **Cross-Encoder Reranking**: Chấm điểm chú ý cặp (Query, Chunk) bằng mô hình Reranker giúp nâng cao độ chính xác trích xuất.

---

## 🚀 Quick Start

```powershell
# 1. Copy env file
Copy-Item .env.example .env

# 2. Điền thông tin vào .env:
#    OPENAI_API_KEY=sk-...
#    REDIS_HOST=localhost
#    QDRANT_HOST=localhost

# 3. Khởi động Infra (Redis + Qdrant Docker)
docker compose up -d redis qdrant

# 4. Cài đặt dependencies
pip install -r requirements.txt

# 5. Chạy giao diện Gradio UI (mặc định port 7860)
python app_gradio.py

# 6. Truy cập trình duyệt: http://127.0.0.1:7860
```

---

## 📂 Structure Dự án

```
src/
  doc_loader.py        - Unified loader: PDF, DOCX, XLSX, MD, TXT -> DocPage
  chunker.py           - 4 Chunking strategies: Fixed, Recursive, Parent-Child, Tokenizer
  hybrid_search.py     - Sparse BM25 search + Reciprocal Rank Fusion (RRF)
  reranker.py          - Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
  rag.py               - Qdrant index/search pipeline + Two-stage retrieval
  chat.py              - Chat orchestration: chat_stream(), chat_rag_stream()
  simple_crawler.py    - CPU-based lightweight HTML crawler & Markdown extractor
  crawler.py           - crawl4ai Chromium crawler (dùng cho JS-heavy sites)
  session.py           - Redis session, pending markers, LLM cache
  config.py            - Pydantic settings (.env driven)
  main.py              - FastAPI app + lifespan

app_gradio.py          - Gradio Web UI (Upload file, URL fetch, Strategy & Rerank controls)
requirements.txt       - Dependencies
tests/                 - Unit tests & Smoke tests
```

---

## 📄 Documentation

- [PLAN.md](PLAN.md) - Lộ trình nâng cấp hệ thống chi tiết theo Phase.
