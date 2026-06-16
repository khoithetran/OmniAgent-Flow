# OmniAgent Flow - Tài liệu ôn tập phỏng vấn

Tài liệu này là nguồn tham chiếu duy nhất để giải thích dự án
OmniAgent Flow trong buổi phỏng vấn kỹ thuật. File này bám sát cấu
trúc dự án và đi vào **lý do** đằng sau từng quyết định, để bạn có
thể trả lời các câu hỏi mở rộng mà không cần lật lại toàn bộ code.

Dùng mục lục bên dưới để nhảy nhanh đến chủ đề mà nhà tuyển dụng đang
hỏi. Mỗi phần kết thúc bằng phần hỏi - đáp mẫu để bạn tự luyện tập.

## Mục lục

1. [Giới thiệu ngắn về dự án](#1-giới-thiệu-ngắn-về-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Phase 1 - Backend lõi & webhook bất đồng bộ](#3-phase-1---backend-lõi--webhook-bất-đồng-bộ)
4. [Phase 2 - Agentic AI & Advanced RAG](#4-phase-2---agentic-ai--advanced-rag)
5. [Phase 3 - CRM & thông báo realtime](#5-phase-3---crm--thông-báo-realtime)
6. [Phase 4 - Observability, testing, dashboard](#6-phase-4---observability-testing-dashboard)
7. [Playbook vận hành](#7-playbook-vận-hành)
8. [Các câu hỏi phỏng vấn thường gặp](#8-các-câu-hỏi-phỏng-vấn-thường-gặp)

---

## 1. Giới thiệu ngắn về dự án

**Một câu tóm tắt:** Hệ thống AI Agent chăm sóc khách hàng đa kênh
theo hình mẫu production, nhận webhook, phân loại intent khách hàng
bằng state machine LangGraph, truy xuất câu trả lời có cơ sở qua
hybrid RAG, đồng bộ lead chất lượng sang HubSpot, cảnh báo con người
trên Telegram, và đẩy trace sang LangFuse.

**Vì sao dự án này có ích cho CV của bạn:**
- Thể hiện được **pattern pipeline bất đồng bộ hoàn chỉnh** mà hầu
  hết công ty đều muốn ở một kỹ sư backend senior: webhook -> queue
  -> worker -> external APIs.
- Kết hợp được **GenAI engineering** (LangGraph, structured outputs,
  RAG, evaluation) với **backend truyền thống** (FastAPI, Celery,
  Redis, PostgreSQL, Docker).
- Có sẵn **observability** và **testing** - hai phần mà hầu hết dự
  án portfolio bỏ qua nhưng mọi vòng phỏng vấn đều hỏi.

**Tổng quan tech stack:**

| Tầng                  | Công cụ                                       |
| --------------------- | --------------------------------------------- |
| HTTP API              | FastAPI (async)                               |
| Task queue            | Celery + Redis broker                         |
| Bộ nhớ ngắn hạn       | Redis (list + TTL 1800s)                      |
| Bộ nhớ dài hạn        | PostgreSQL (conversations, messages, sync log)|
| Agent                 | LangGraph state machine                       |
| LLM                   | OpenAI structured outputs                     |
| Vector DB             | Qdrant (hybrid dense + BM25 + reranker)       |
| CRM                   | HubSpot v3 contacts API                      |
| Thông báo             | Telegram Bot HTTP API                        |
| Observability         | LangFuse traces + scores                     |
| Container hóa         | Docker + docker-compose                      |
| Testing               | pytest + pytest-asyncio                       |

---

## 2. Kiến trúc hệ thống

```
              Facebook Messenger / Web Channel
                          |
                          v  POST /api/webhook
              +---------------------------+
              |       FastAPI app         |
              |  (returns HTTP 200 fast)  |
              +-----------+---------------+
                          |
                          v  .delay()
              +---------------------------+
              |   Celery worker (Redis)   |
              +-----------+---------------+
                          |
              +-----------+-----------+
              | generate_agent_result  |
              |  - RAG retrieval       |
              |  - LangGraph run       |
              |  - LangFuse trace      |
              +-----------+-----------+
                          |
        +-----------------+------------------+
        v                 v                  v
  Redis session     PostgreSQL         LangFuse cloud
  (short-term)      (durable history)  (traces + scores)
        |
        v
  +---------------------+
  | sync_hubspot_lead   |
  +---------------------+
        |
        v
  +-------------------------+
  | send_telegram_... (RT)  |
  +-------------------------+
```

### Vì sao lại cần pipeline bất đồng bộ?

Webhook của Facebook yêu cầu trả về HTTP 200 trong **dưới 5 giây**;
nếu gọi LLM đồng bộ, bạn sẽ đối mặt với timeout, retry, và xử lý
trùng lặp. Đẩy payload vào Celery đảm bảo xử lý **ít nhất một lần
(at-least-once)** với **xác nhận nhanh** cho kênh phía trên. Chúng ta
cô lập phần việc chậm (LLM, vector search, HubSpot) khỏi đường HTTP
nóng. Đây chính là pattern mà webhook thanh toán (Stripe), nhắn tin
(Twilio), và hệ thống CI/CD (GitHub Actions) đều dùng.

---

## 3. Phase 1 - Backend lõi & webhook bất đồng bộ

### 3.1 Cấu trúc thư mục

```
src/
  main.py            - FastAPI app + lifespan
  config.py          - Pydantic settings (env-driven)
  database.py        - asyncpg + redis-asyncio pools
  api/webhook.py     - GET (verify) + POST (enqueue)
  workers/
    celery_app.py    - broker / backend config
    tasks.py         - process_incoming_message
  services/
    session_service.py        - Lịch sử session trên Redis
    conversation_service.py   - Lưu trữ PostgreSQL
    ai_service.py             - Điều phối LangGraph
    intent_service.py         - Trích xuất structured output
    rag_service.py            - Hybrid retrieval
    hubspot_service.py        - Đồng bộ CRM
    telegram_service.py       - Cảnh báo realtime
    observability_service.py  - Wrapper LangFuse
    evaluation_service.py     - Chấm điểm faithfulness/relevance
```

### 3.2 Xác thực webhook

Facebook dùng cơ chế bắt tay **hub.mode / hub.verify_token /
hub.challenge**. Webhook chỉ trả về `hub.challenge` khi token khớp.
Chúng ta không bao giờ nhận traffic vào nếu không qua bước bắt tay
này, vì nếu không bất kỳ ai cũng có thể gửi tin nhắn giả.

### 3.3 Pattern đẩy vào queue

```python
@router.post("")
async def receive_webhook(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
    task = process_incoming_message.delay(payload)
    return {"status": "success", "task_id": task.id}
```

Handler chỉ làm **đúng hai** việc:
1. Đẩy raw payload vào Celery.
2. Trả về `200 OK` kèm task id.

Đây là hợp đồng nghiêm ngặt nhất có thể cho một webhook bên ngoài:
**không bao giờ** xử lý đồng bộ, **không bao giờ** trả mã lỗi cho
lỗi có thể phục hồi. Client chỉ cần xác nhận đã nhận.

### 3.4 Lịch sử session trên Redis

`session_service.py` giữ một cửa sổ trượt 10 tin nhắn gần nhất
trong một Redis list với TTL 30 phút. Độ rộng cửa sổ là siêu tham
số cân bằng giữa độ phong phú của ngữ cảnh và chi phí token.

Những điểm có thể bảo vệ trong phỏng vấn:
- Chúng ta lưu chuỗi JSON trong Redis list, không phải hash. List
  giữ thứ tự chèn và cho phép `LTRIM` để áp trần.
- Chúng ta dùng `RPUSH` + `LTRIM -N -1` để chỉ giữ N mục mới nhất.
- `EXPIRE` được gọi lại ở mỗi lần ghi, nên session sống 30 phút kể
  từ **tin nhắn cuối**, không phải từ tin nhắn đầu.

### 3.5 Celery task

Worker dùng `asyncio.run` bên trong Celery task đồng bộ. Đây là
lựa chọn thực dụng: thân task nhỏ và không có event loop chia sẻ,
nên tạo một loop mới mỗi task rẻ hơn là phải quản lý loop của worker
thủ công. Nếu cần throughput cao hơn, dùng Celery `gevent` pool để
tái sử dụng một loop.

---

## 4. Phase 2 - Agentic AI & Advanced RAG

### 4.1 Vì sao chọn LangGraph?

Một lệnh gọi LLM trần không thể routing đáng tin cậy. LangGraph
cho chúng ta:

- Một `AgentState` có kiểu rõ ràng, được mutate bởi các node.
- Hàm `route_agent_action` ánh xạ `intent -> response_node`.
- Cạnh có điều kiện để graph **phân nhánh thật sự** theo output
  của bộ phân loại.

Đây là nguyên thủy đúng cho mọi workflow "AI cần đi theo đường
khác nhau tuỳ tình huống người dùng". Nó cũng làm cho graph có thể
serialize - chính là thứ mà trace LangFuse render.

### 4.2 Structured outputs + Pydantic

```python
class CustomerIntentExtraction(BaseModel):
    intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0)
    customer_name: str | None
    company: str | None
    phone: str | None
    email: str | None
    budget: str | None
    ...
```

Chúng ta truyền `text_format=CustomerIntentExtraction` cho OpenAI.
Model trả về JSON, Pydantic validate, chúng ta nhận lại một object
đã có kiểu. Không cần parse chuỗi, không cần regex trên output LLM,
không cần vòng retry. Chúng ta cũng giữ một bộ trích xuất dự phòng
tất định (`_build_fallback_extraction`) chạy bằng heuristic từ
khoá khi thiếu OpenAI key, để hệ thống vẫn demo được offline.

### 4.3 Pipeline RAG

`rag_service.py` hiện thực **hybrid search** end-to-end:

1. **Dense retrieval**: embedding dựa trên hash (tất định cho
   bản demo) + cosine similarity trong Qdrant. Câu truy vấn được
   embed, top-K candidate được trả về.
2. **BM25 sparse retrieval**: hiện thực BM25 bằng Python trên một
   corpus đã cuộn. Trọng số tinh chỉnh qua `RAG_DENSE_WEIGHT` và
   `RAG_BM25_WEIGHT`.
3. **Hybrid fusion**: tổng có trọng số của điểm dense và BM25 đã
   chuẩn hoá. `RAG_CANDIDATE_LIMIT` kiểm soát kích thước tập ứng
   viên.
4. **Reranking**: cross-encoder tuỳ chọn qua `fastembed`
   (`BAAI/bge-reranker-base`). Khi `RAG_ENABLE_RERANKER=true`,
   top-K candidate được chấm lại bằng một model mạnh hơn để giảm
   ảo giác.

Vì sao phải dùng cả hai? Dense retrieval giỏi về match ngữ nghĩa
("làm sao để huỷ?" ~ "cancel plan") nhưng kém về match từ khoá
chính xác (mã SKU, mã lỗi). BM25 thì ngược lại. Hybrid + reranker
là câu trả lời cấp production mà mọi hệ thống RAG nghiêm túc đều
hội tụ về.

### 4.4 Gắn RAG vào graph

Chúng ta gọi `hybrid_search_knowledge(user_message, limit=3)`
**trước khi** kích hoạt agent, rồi inject ngữ cảnh như một tin
nhắn `role: system` giả lập. Các node LangGraph giữ thuần khiết:
chúng chỉ thấy state và phản hồi. Sự tách bạch này cho phép test
graph mà không cần RAG, và test riêng bộ truy xuất.

---

## 5. Phase 3 - CRM & thông báo realtime

### 5.1 Đồng bộ HubSpot

`sync_hubspot_lead` đi theo pattern "search rồi upsert" chuẩn:

1. Xây `HubSpotLeadPayload` có kiểu chặt chẽ từ metadata do
   LangGraph trích xuất.
2. POST tới `/crm/v3/objects/contacts/search` lọc theo email,
   sau đó theo phone.
3. Nếu contact đã tồn tại -> PATCH properties. Nếu chưa -> POST
   để tạo. `action` trả về là "created" hoặc "updated".
4. Luôn ghi một dòng `hubspot_lead_syncs` vào PostgreSQL để audit
   mọi lần sync, kể cả khi thất bại.

Các lựa chọn phòng thủ:
- Dùng `httpx.AsyncClient` và `HubSpotHTTPClient` Protocol để
  unit test có thể inject client giả.
- Đóng client trong khối `finally` để tránh rò rỉ kết nối.
- Service là **fail-soft**: mọi lỗi trả về
  `HubSpotLeadSyncResult` có cấu trúc, để worker vẫn chạy nốt
  phần còn lại của pipeline và cảnh báo vẫn được gửi.

### 5.2 Cảnh báo realtime qua Telegram

`send_telegram_notification` là mảnh đơn giản nhất của hệ thống.
Nó gửi một tin nhắn HTML vào chat đã cấu hình qua `sendMessage`
Bot API. Có bốn loại sự kiện:

- `hubspot_sync_failed` -> Ghi CRM thất bại; cần điều tra.
- `handoff_requested`    -> Khách hàng yêu cầu gặp người.
- `hot_lead_captured`    -> Intent báo giá + CRM đã sync.
- `new_message`          -> Mặc định cho phần còn lại.

Chúng ta chọn event type từ `(intent, hubspot_status)` để kênh
on-call dễ lọc. Mọi trường do người dùng kiểm soát đều được escape
bằng `html.escape` vì Telegram diễn giải HTML.

Vì sao không phải Slack? Slack nặng nề (OAuth, scopes, signing
secrets). Telegram Bot API chỉ là một HTTP call - hoàn hảo cho
bản demo và cho bất kỳ team nhỏ nào chỉ cần push notification.

---

## 6. Phase 4 - Observability, testing, dashboard

### 6.1 LangFuse observability

`observability_service.py` bọc LangFuse SDK sau một lớp trừu
tượng mỏng. Module export:

- `is_observability_enabled()` - kiểm tra nhanh.
- `atrace_agent_run` - async context manager mở trace.
- `record_intent_generation` - ghi generation span với model,
  usage, latency.
- `record_evaluation_score` - gắn điểm số (0-1) vào trace.

Module tích hợp sẵn **null client** để test và dev offline vẫn
chạy được khi không có tài khoản LangFuse. Bật lên chỉ tốn một
biến env: `LANGFUSE_ENABLED=true`.

### 6.2 Đánh giá LLM

`evaluation_service.py` cung cấp hai scorer mà dashboard LangFuse
sẽ tiêu thụ:

- **Faithfulness**: câu trả lời bám vào context được truy xuất tới
  mức nào? Tính bằng token overlap với tài liệu được truy xuất.
  Khi có OpenAI key, gọi model và xin điểm 0-1 kèm rubric.
- **Answer relevance**: câu trả lời có thật sự đúng câu hỏi không?
  Pha trộn token overlap với câu hỏi người dùng và thưởng từ khoá
  cho các cụm từ intent đã biết.

Cả hai đều fallback về heuristic tất định khi LLM không khả dụng.
Kết quả được gấp vào metadata của agent, ghi vào PostgreSQL, và
đẩy sang LangFuse thành score.

### 6.3 Testing

Bộ test nằm trong `tests/` và chạy bằng pytest. Chúng ta tách
theo từng service để dễ khoanh vùng lỗi:

- `test_webhook.py`              - Hợp đồng HTTP, GET/POST, 403 khi
                                   token sai, enqueue Celery.
- `test_session_service.py`      - Thao tác list trên Redis.
- `test_rag_service.py`          - Hybrid search, BM25, bật/tắt
                                   reranker.
- `test_intent_service.py`       - Structured output + fallback.
- `test_agent.py`                - Routing LangGraph.
- `test_hubspot_service.py`      - Luồng upsert với client giả.
- `test_telegram_service.py`     - Escape HTML, send/skip/fail.
- `test_conversation_service.py` - Schema + helpers.
- `test_observability_service.py` - Hành vi null client.
- `test_evaluation_service.py`   - Fallback scorers.
- `test_tasks.py`                - Pipeline Celery task.
- `test_app_factory.py`          - Lifespan + healthcheck.

Chạy bằng `python -m pytest tests/ -v`. Hiện tại chạy 45 test, tất
cả xanh, dưới 2 giây.

### 6.4 Dashboard Looker Studio

Chúng ta **không** stream sự kiện vào warehouse. Dashboard đọc
thẳng từ PostgreSQL qua các SQL view. Các view nằm trong
`migrations/0010_looker_views.sql`:

- `vw_daily_intent_volume`       - Stacked bar intent theo ngày.
- `vw_intent_summary`            - KPI (pricing, handoff, fallback).
- `vw_hubspot_sync_outcomes`     - Sức khoẻ sync CRM.
- `vw_conversation_insights`     - Bảng xếp hạng lead (channels,
                                   pain points được trải phẳng từ
                                   JSONB).
- `vw_conversation_volume_hourly`- Heatmap theo giờ trong ngày.

Hướng dẫn report Looker Studio nằm trong
`docs/looker_studio.md`. Hợp đồng giữa BI tool và database chính
là tập view này; ta có thể đổi bảng bên dưới mà không phá
dashboard.

---

## 7. Playbook vận hành

### Phát triển local

```bash
# Khởi động infra + app
docker compose up --build

# Chạy worker (terminal riêng)
celery -A src.workers.tasks worker --loglevel=info
```

### Smoke test

```bash
# Xác thực webhook
python test_webhook.py

# Luồng session
python test_session.py

# Pipeline đầy đủ (webhook -> queue -> agent -> DB)
python test_queue.py

# Unit test RAG / agent / conversation / HubSpot
python -m pytest tests/ -v
```

### Ghi chú triển khai production

- `WEBHOOK_VERIFY_TOKEN` phải được rotate và lưu trong secret
  manager. Token này là **thứ duy nhất** ngăn cản kẻ tấn công
  truy cập Celery queue của bạn.
- Thông tin đăng nhập Postgres nên đến từ một secret store được
  quản lý (AWS Secrets Manager, GCP Secret Manager, HashiCorp
  Vault). `docker-compose.yml` đã fail-fast nếu bất kỳ biến nào
  bị rỗng.
- Key `LANGFUSE_*` là tuỳ chọn. Không có chúng hệ thống vẫn chạy
  - chỉ là dừng gửi trace.
- `HUBSPOT_SYNC_ENABLED=true` kèm private app token là bắt buộc
  cho ghi CRM thật.
- `TELEGRAM_NOTIFICATIONS_ENABLED=true` yêu cầu bot token và
  chat id của nhóm đích (số âm cho group).

---

## 8. Các câu hỏi phỏng vấn thường gặp

### Câu 1. Vì sao webhook lại bất đồng bộ?

> Facebook kỳ vọng HTTP 200 trong <5s. LLM + RAG + ghi HubSpot
> có thể tốn vài giây. Nếu trả lời đồng bộ, ta đối mặt với
> timeout và xử lý trùng lặp. Chúng ta đẩy vào Celery và xác
> nhận trong <500ms. Worker lo phần việc chậm, retry, và xử lý
> lỗi.

### Câu 2. Vì sao Redis cho session mà Postgres cho message?

> Session có giới hạn, ngắn hạn, và cần đọc/ghi O(1) với TTL.
> Redis sinh ra để làm việc đó. Lịch sử hội thoại phải sống
> sót qua restart container và truy vấn được theo ngày/intent để
> phân tích. Đó là use case quan hệ, nên dùng Postgres với
> JSONB cho linh hoạt về metadata.

### Câu 3. Làm sao ngăn LLM ảo giác?

> Ba lớp:
> 1. **Structured outputs** ép LLM trả lời theo schema Pydantic.
>    Nó không thể bịa ra trường tự do.
> 2. **Hybrid RAG + reranker** grounding câu trả lời trong
>    knowledge base. Chúng ta inject tài liệu truy xuất như
>    system message trước khi agent chạy.
> 3. **Đánh giá Faithfulness** chấm điểm từng lượt và đẩy sang
>    LangFuse để phát hiện regression trên dashboard.

### Câu 4. Vì sao LangGraph thay vì một prompt duy nhất?

> Một prompt không thể routing đáng tin cậy. LangGraph cho phép
> khai báo `AgentState` rõ ràng, một node `classify_intent`, và
> bốn nhánh phản hồi. Graph test được, trace được, và compose
> được. Đây là nguyên thủy đúng cho mọi workflow AI phân nhánh
> (routing, escalation, handoff).

### Câu 5. Làm sao test hệ thống có LLM mà không bị flake?

> Chúng ta test **thứ mình sở hữu**, không test LLM. Routing
> của agent được test bằng cách tắt structured-output extractor
> và dùng fallback classifier tất định. RAG service test với
> hash embedding tất định. HubSpot và Telegram dùng fake client
> dựa trên Protocol. Chỉ duy nhất lệnh gọi OpenAI còn lại là
> phụ thuộc mạng, và lệnh đó nằm sau `try/except` có fallback
> về đường tất định.

### Câu 6. Sẽ thay đổi gì cho production?

> - Thay hash embedding bằng model thật (text-embedding-3-small
>   hoặc BGE) và lưu vector trong Qdrant với HNSW.
> - Dùng Celery `gevent` worker pool để tăng concurrency.
> - Thêm retry policy với exponential backoff cho HubSpot và
>   OpenAI.
> - Chuyển SQL view sang read replica với user phân tích riêng.
> - Thay null LangFuse client bằng exporter dạng queue để trace
>   sống sót qua restart worker.

### Câu 7. Hệ thống giữ được tính fail-soft như thế nào?

> Mọi cuộc gọi bên ngoài (HubSpot, Telegram, OpenAI, LangFuse)
> đều được bọc trong `try/except`. Pipeline không bao giờ sập
> vì Telegram chết hay HubSpot rate-limit. Lỗi được log qua
> `loguru` và lưu vào PostgreSQL để sau này đối chiếu lại.

### Câu 8. Đi qua một message thật giúp tôi.

> 1. Khách gửi "Tôi muốn xin báo giá" trên Messenger.
> 2. Facebook POST tới `/api/webhook`. FastAPI parse, đẩy
>    payload vào Celery, trả về 200 OK kèm task id.
> 3. Worker kéo task, nạp session từ Redis (10 tin nhắn gần
>    nhất), và lấy top-3 tài liệu RAG.
> 4. `ai_service.generate_agent_result` mở trace LangFuse, kích
>    hoạt state machine LangGraph, phân loại intent là
>    `pricing`, và chạy node `pricing_response`.
> 5. Phản hồi được lưu vào Redis (refresh TTL) và PostgreSQL
>    kèm metadata có cấu trúc.
> 6. HubSpot service tạo contact với `lifecyclestage = lead`.
>    Telegram service bắn cảnh báo `hot_lead_captured` vào
>    kênh sales.
> 7. Trace LangFuse nhận điểm faithfulness và relevance, sau
>    đó trace được flush.
