# SPEC: OmniAgent Flow — Gradio Interface

> **Status:** FINAL (đã xác nhận với user ngày 17/06/2026)
> **Scope:** Phase 6 — Thay thế Telegram bot bằng giao diện Gradio web.

---

## 1. Mục tiêu

Thay thế Telegram bot bằng giao diện Gradio web. User nhập URL công ty → crawl + index → chat tra cứu RAG. Không có tài liệu → general LLM. Có tài liệu → RAG-only, không tự bịa thông tin.

**Điểm khác biệt chính với Telegram:**
- Streaming mượt như ChatGPT (Gradio SSE native, không throttle)
- Model selector tích hợp ngay trên khung nhập
- Panel bên phải: URL đã fetch + X button để clear

---

## 2. Layout — Right sidebar

```
┌──────────────────────────────────────────────┬─────────────────────────┐
│                                              │                         │
│  [gr.Chatbot — chiếm phần còn lại]          │  🔗 stripe.com    [✕]  │
│                                              │  ─────────────────────  │
│  Bot: Xin chào! Tôi sẵn sàng tra cứu...    │  ✅ 12 trang, 76 chunks│
│                                              │                         │
│  User: Công ty này có mấy nhân viên?       │  [ Clear KB ]           │
│                                              │                         │
│  Bot: [RAG reply với citation [1]]           │                         │
│                                              │                         │
├──────────────────────────────────────────────┤                         │
│  Model: [gpt-4o-mini] [gpt-4o] [o4-mini]   │                         │
│         [gpt-4o-realtime]                    │                         │
├──────────────────────────────────────────────┤                         │
│  [Nhập câu hỏi...                      ] [➡]│                         │
└──────────────────────────────────────────────┴─────────────────────────┘
```

- **Cột phải (~25%)**: URL đã fetch (nếu có) + X button + status + Clear KB. Độ cao = chatbot height.
- **Cột trái (~75%)**: Chatbot + model selector + input.
- **Model selector**: row of clickable buttons (gr.Button), active model có variant="primary".
- **Khi chưa crawl**: cột phải hiện placeholder "Chưa có tài liệu" + URL input + Fetch button.

---

## 3. Model Selector

### 3.1 Model List (4 models)

| ID | Tên hiển thị | Provider | Default |
|----|-------------|----------|---------|
| `gpt-4o-mini` | gpt-4o-mini | OpenAI | ✓ (mặc định) |
| `gpt-4o` | gpt-4o | OpenAI | — |
| `o4-mini` | o4-mini | OpenAI | — |
| `gpt-4o-realtime` | gpt-4o-realtime | OpenAI | — |

**Lưu ý**: tên model phải khớp với OpenAI API. Nếu model không tồn tại trong API (ví dụ `gpt-4o-realtime` chưa phổ biến), exception sẽ được catch và hiển thị fallback.

### 3.2 UI

- 4 button nằm ngang (gr.Row)
- Active model: `variant="primary"`
- Inactive: `variant="secondary"`
- Click button → cập nhật state → dùng model đó cho câu hỏi tiếp theo

---

## 4. Chat Behavior

### 4.1 Mode matrix

| Trạng thái KB | User hỏi | Bot trả lời |
|---------------|-----------|-------------|
| Chưa crawl | Bất kỳ | General LLM |
| Chưa crawl | Hỏi về công ty | General LLM + gợi ý nhập URL |
| Đã crawl | Câu trong tài liệu | RAG + citation [n] |
| Đã crawl | Câu không trong tài liệu | "Không tìm thấy thông tin này trong tài liệu." |
| Đã crawl | Câu hỏi chung | "Không tìm thấy thông tin này trong tài liệu." |
| Clear KB (vừa bấm X) | Bất kỳ | General LLM + cảnh báo 1 lần |

### 4.2 Hai System Prompt

**`SYSTEM_PROMPT_GENERAL`** (chưa crawl):
```
Bạn là trợ lý ảo thân thiện, trả lời ngắn gọn bằng tiếng Việt có dấu.
Nếu câu hỏi liên quan đến công ty/tổ chức cụ thể, hãy gợi ý người dùng
nhập URL website để được tra cứu chính xác hơn.
```

**`SYSTEM_PROMPT_RAG`** (đã crawl):
```
Bạn là trợ lý ảo chỉ trả lời dựa trên "Knowledge Base" được cung cấp.
MỖI phát biểu phải gắn citation theo số thứ tự trong ngoặc vuông, ví dụ: [1].
Nếu thông tin không có trong Knowledge Base, hãy trả lời đúng:
"Không tìm thấy thông tin này trong tài liệu được cung cấp."
TUYỆT ĐỐI KHÔNG bịa đặt, suy đoán, hay diễn giải thêm.
```

### 4.3 Welcome message

Khi mở chat lên, chatbot hiển thị sẵn 1 message chào và hướng dẫn:

> Xin chào! Tôi có thể giúp gì cho bạn?
> Nếu bạn cần thông tin chính xác từ nguồn có sẵn, hãy dùng chức năng Fetch & Index trước khi đặt câu hỏi.

Message này là **static greeting** — không phải từ LLM, được set trực tiếp vào `gr.Chatbot(value=[...])`.

### 4.4 Warning messages (1 lần)

**Sau khi crawl xong** → bot gửi 1 message:
> ⚠️ **Lưu ý**: Tôi chỉ trả lời dựa trên nội dung tài liệu đã cung cấp. Tôi sẽ không trả lời các câu hỏi ngoài phạm vi này.

**Sau khi clear KB (bấm X)** → bot gửi 1 message:
> ⚠️ **Đã xóa tài liệu**. Tôi đang ở chế độ kiến thức chung. Các câu hỏi về công ty có thể không chính xác.

---

## 5. Panel bên phải

### 5.1 Khi chưa crawl

```
┌──────────────────────────┐
│  📄 Nguồn tài liệu      │
│  ──────────────────────  │
│  Chưa có tài liệu.      │
│  Nhập URL bên dưới để   │
│  bắt đầu crawl.         │
│                          │
│  URL:                    │
│  ┌────────────────────┐  │
│  │ https://...        │  │
│  └────────────────────┘  │
│  [  Fetch & Index  ]     │
└──────────────────────────┘
```

### 5.2 Khi đã crawl

```
┌──────────────────────────┐
│  🔗 stripe.com     [✕]  │
│  ──────────────────────  │
│  ✅ 12 trang, 76 chunks  │
│                          │
│  [  Clear KB       ]    │
└──────────────────────────┘
```

- **X button**: clear KB → warning message → quay về state chưa crawl
- **URL input + Fetch button**: vẫn hiện ở dưới cùng panel, cho phép fetch URL mới (replace KB cũ)

---

## 6. State Management (gr.State)

```python
gr.State(value={
    "kb_ready": False,
    "kb_domain": "",          # "stripe.com" hiển thị trên panel
    "kb_pages": 0,
    "kb_chunks": 0,
    "selected_model": "gpt-4o-mini",
    "warning_shown": False,   # đã hiện warning sau crawl chưa
})
```

---

## 7. Module Changes

### 7.1 `src/chat.py`

| Thay đổi | Chi tiết |
|-----------|----------|
| Thêm `SYSTEM_PROMPT_GENERAL` | Prompt cho mode chưa crawl |
| Rename `SYSTEM_PROMPT` → `SYSTEM_PROMPT_RAG` | Prompt cho mode đã crawl |
| Thêm `chat_general_stream(sender_id, msg, model)` | LLM streaming, không RAG |
| Thêm `chat_rag_stream(sender_id, msg, model)` | RAG + LLM streaming |
| Refactor `chat_stream()` | Thêm `kb_ready` param; quyết định dùng prompt nào |

### 7.2 `app_gradio.py` (mới)

```
app_gradio.py — ~200 dòng
├── MODELS = {...}          # định nghĩa 4 model
├── _SYSTEM_PROMPT_GENERAL
├── _SYSTEM_PROMPT_RAG
├── build_app()            # gr.Blocks layout
│   ├── Chatbot component
│   ├── Right panel
│   ├── Model selector row (4 buttons)
│   └── Textbox input + Send button
├── async def handle_fetch(state, url)
├── async def handle_chat(state, msg, history)
└── main: gr.Blocks(...).launch()
```

### 7.3 `requirements.txt`

Thêm: `gradio>=5.0`

### 7.4 `src/main.py`

Cập nhật lifespan để khởi tạo resources cho cả Telegram (giữ nguyên) và Gradio.

### 7.5 `PLAN.md`

Thêm Phase 6: Gradio Interface.

---

## 8. Error Handling

| Error | UI response |
|-------|------------|
| URL sai format | Panel: `⚠️ URL không hợp lệ. Vui lòng nhập URL bắt đầu bằng http:// hoặc https://` |
| URL không truy cập được | Panel: `❌ Không truy cập được URL này` |
| Crawl 0 pages | Panel: `❌ Không crawl được trang nào` |
| Partial crawl | Panel: `⚠️ Chỉ crawl được {n} trang` |
| OpenAI error | Chat bot msg: `Xin lỗi, đã xảy ra lỗi khi gọi LLM.` |
| Model không tồn tại | Chat bot msg: `Model {name} không khả dụng. Dùng gpt-4o-mini.` |

---

## 9. Out of Scope

- Telegram bot (đóng băng, không xóa)
- Email capture
- Multi-URL KB
- URL auto-detection trong chat
- Real-time crawl progress (chỉ hiện spinner)

---

## 10. Acceptance Criteria

```
[ ] Layout đúng: chat trái, panel phải, model selector trên input
[ ] Mở chat: hiện welcome message mặc định (không cần user nhập gì)
[ ] Model selector: 4 button, active highlighted
[ ] Chưa crawl: chat general LLM, panel hiện "Chưa có tài liệu"
[ ] Nhập URL sai → cảnh báo trong panel
[ ] Bấm Fetch → spinner → status cập nhật
[ ] Crawl xong → panel hiện domain + số chunks + X button
[ ] Crawl xong → 1 warning message trong chat
[ ] Đã crawl: hỏi có trong tài liệu → RAG + citation
[ ] Đã crawl: hỏi không có → "Không tìm thấy..."
[ ] Bấm X → KB cleared → warning message → quay về general
[ ] Streaming mượt (Gradio SSE, không throttle)
[ ] Refresh trang → state giữ nguyên (Gradio server-side)
```

---

## 11. File Changes Summary

| File | Action |
|------|--------|
| `SPEC.md` | 🆕 Create |
| `app_gradio.py` | 🆕 Create (~200 dòng) |
| `src/chat.py` | ✏️ Modify — thêm general mode + model param |
| `requirements.txt` | ✏️ Add `gradio>=5.0` |
| `PLAN.md` | ✏️ Add Phase 6: Gradio Interface |
| `tests/test_chat.py` | ✏️ Add test cho general mode |
