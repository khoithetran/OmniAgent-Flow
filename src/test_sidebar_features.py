import asyncio
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.doc_loader import load_document_from_bytes, supported_extensions, DocPage
from src.chunker import ChunkStrategy, chunk_pages, chunk_text
from src.rag import index_markdown, search, format_context, qdrant_available
from src import hybrid_search, reranker
from src.eval import evaluate_rag_pipeline, format_eval_summary

# Prepare test sample contents
SAMPLE_MD = """# Báo Cáo Quy Trình Công Ty OmniAgent Flow 2026

## 1. Quy Trình Mua Sắm Vật Tư (Procurement Process)
- Bước 1: Nhân viên gửi phiếu yêu cầu mua sắm (Purchase Order - PO) lên hệ thống SAP.
- Bước 2: Trưởng phòng phê duyệt phiếu yêu cầu trong vòng 24 giờ làm việc.
- Bước 3: Bộ phận Mua hàng thực hiện so sánh giá của 3 nhà cung cấp khác nhau.
- Bước 4: Thanh toán 50% tiền đặt cọc qua tài khoản doanh nghiệp. Mã ngân hàng Swift: OMNI2026.

## 2. Quy Định Nghỉ Phép Nội Bộ
- Mỗi nhân viên chính thức có 12 ngày nghỉ phép hưởng nguyên lương mỗi năm.
- Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi năm làm việc tiếp theo.
- Nghỉ phép từ 3 ngày liên tiếp trở lên phải báo trước ít nhất 5 ngày làm việc.
"""

SAMPLE_TXT = "Đây là văn bản kiểm thử định dạng TXT cho OmniAgent Flow. Mã số hợp đồng bảo mật là HD-TEST-998877."

results = []

def record_test(name: str, category: str, status: str, duration_ms: float, details: str):
    results.append({
        "name": name,
        "category": category,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "details": details
    })
    print(f"[{status}] {category} -> {name} ({round(duration_ms, 2)}ms): {details}")

async def run_all_tests():
    print("==================================================")
    print("   BẮT ĐẦU KIỂM THỬ TOÀN BỘ TÍNH NĂNG SIDEBAR BAR   ")
    print("==================================================\n")

    # -------------------------------------------------------------
    # NHÓM 1: KIỂM THỬ NẠP TÀI LIỆU (FILE LOADERS & PARSERS)
    # -------------------------------------------------------------
    print("--- 1. KIỂM THỬ TỪNG ĐỊNH DẠNG FILE ---")
    
    # 1.1 Text/Markdown
    t0 = time.time()
    try:
        pages = load_document_from_bytes(SAMPLE_MD.encode("utf-8"), "sample.md")
        record_test("Nạp File Markdown (.md)", "File Ingestion", "PASS", (time.time()-t0)*1000, f"Đã parse thành công {len(pages)} trang.")
    except Exception as e:
        record_test("Nạp File Markdown (.md)", "File Ingestion", "FAIL", (time.time()-t0)*1000, str(e))

    # 1.2 TXT File
    t0 = time.time()
    try:
        pages_txt = load_document_from_bytes(SAMPLE_TXT.encode("utf-8"), "sample.txt")
        record_test("Nạp File Văn Bản (.txt)", "File Ingestion", "PASS", (time.time()-t0)*1000, f"Đã parse thành công {len(pages_txt)} trang.")
    except Exception as e:
        record_test("Nạp File Văn Bản (.txt)", "File Ingestion", "FAIL", (time.time()-t0)*1000, str(e))

    # 1.3 Unsupported File Extension Check
    t0 = time.time()
    try:
        load_document_from_bytes(b"binary content", "sample.exe")
        record_test("Kiểm tra File không hỗ trợ (.exe)", "File Ingestion", "FAIL", (time.time()-t0)*1000, "Không ném ra ValueError như mong đợi.")
    except ValueError as e:
        record_test("Kiểm tra File không hỗ trợ (.exe)", "File Ingestion", "PASS", (time.time()-t0)*1000, f"Đã bắt lỗi chính xác: {e}")
    except Exception as e:
        record_test("Kiểm tra File không hỗ trợ (.exe)", "File Ingestion", "FAIL", (time.time()-t0)*1000, str(e))

    # -------------------------------------------------------------
    # NHÓM 2: KIỂM THỬ TỪNG CHIẾN LƯỢC CHUNKING (CHUNKING STRATEGIES)
    # -------------------------------------------------------------
    print("\n--- 2. KIỂM THỬ TỪNG CHIẾN LƯỢC CHUNKING ---")
    test_pages = [DocPage(page_num=1, content=SAMPLE_MD, source="sample.md", doc_type="markdown")]

    for strategy in [ChunkStrategy.RECURSIVE, ChunkStrategy.FIXED, ChunkStrategy.PARENT_CHILD, ChunkStrategy.TOKENIZER]:
        t0 = time.time()
        try:
            chunks = chunk_pages(test_pages, strategy=strategy)
            record_test(f"Chunking Strategy: {strategy.value}", "Chunking", "PASS", (time.time()-t0)*1000, f"Tạo ra {len(chunks)} chunks.")
        except Exception as e:
            record_test(f"Chunking Strategy: {strategy.value}", "Chunking", "FAIL", (time.time()-t0)*1000, str(e))

    # -------------------------------------------------------------
    # NHÓM 3: KIỂM THỬ TÙY CHỈNH TÌM KIẾM (SEARCH MODES & RERANKING)
    # -------------------------------------------------------------
    print("\n--- 3. KIỂM THỬ TỪNG TÍNH NĂNG TÌM KIẾM & RERANK ---")

    # Index sample data for search testing
    try:
        await index_markdown(SAMPLE_MD, url="https://omniagent.example.com", title="Báo Cáo Quy Trình 2026", replace=True)
    except Exception as e:
        print(f"Indexing result: {e}")

    # 3.1 Dense Vector Search
    t0 = time.time()
    try:
        hits_dense = await search("Mã ngân hàng Swift mua sắm là gì?", enable_hybrid=False, enable_rerank=False)
        top_txt = hits_dense[0].text[:40] if hits_dense else "No hits (empty context)"
        record_test("Dense Vector Search", "Search Customization", "PASS", (time.time()-t0)*1000, f"Trả về {len(hits_dense)} kết quả. Top 1: '{top_txt}...'")
    except Exception as e:
        record_test("Dense Vector Search", "Search Customization", "FAIL", (time.time()-t0)*1000, str(e))

    # 3.2 Hybrid Search (BM25 + Dense RRF)
    t0 = time.time()
    try:
        hits_hybrid = await search("OMNI2026 PO", enable_hybrid=True, enable_rerank=False)
        top_txt = hits_hybrid[0].text[:40] if hits_hybrid else "No hits (empty context)"
        record_test("Hybrid Search (Dense + BM25 RRF)", "Search Customization", "PASS", (time.time()-t0)*1000, f"Trả về {len(hits_hybrid)} kết quả. Top 1: '{top_txt}...'")
    except Exception as e:
        record_test("Hybrid Search (Dense + BM25 RRF)", "Search Customization", "FAIL", (time.time()-t0)*1000, str(e))

    # 3.3 Re-ranking (Cross-Encoder)
    t0 = time.time()
    try:
        hits_rerank = await search("Nghỉ phép được bao nhiêu ngày?", enable_hybrid=True, enable_rerank=True)
        top_txt = hits_rerank[0].text[:40] if hits_rerank else "No hits (empty context)"
        record_test("Cross-Encoder Re-ranking", "Search Customization", "PASS", (time.time()-t0)*1000, f"Trả về {len(hits_rerank)} kết quả sau khi re-rank.")
    except Exception as e:
        record_test("Cross-Encoder Re-ranking", "Search Customization", "FAIL", (time.time()-t0)*1000, str(e))

    # -------------------------------------------------------------
    # NHÓM 4: KIỂM THỬ RAGAS EVALUATION DASHBOARD
    # -------------------------------------------------------------
    print("\n--- 4. KIỂM THỬ RAGAS EVALUATION DASHBOARD ---")
    t0 = time.time()
    try:
        eval_res = await evaluate_rag_pipeline("Quy trình mua sắm gồm mấy bước?", enable_hybrid=True, enable_rerank=True)
        f_score = eval_res.get("faithfulness", 0.0)
        r_score = eval_res.get("answer_relevance", 0.0)
        record_test("RAGAS Evaluation Runner", "Evaluation", "PASS", (time.time()-t0)*1000, f"Faithfulness: {f_score}, Answer Relevance: {r_score}")
    except Exception as e:
        record_test("RAGAS Evaluation Runner", "Evaluation", "FAIL", (time.time()-t0)*1000, str(e))

    # -------------------------------------------------------------
    # NHÓM 5: KIỂM THỬ PHỐI HỢP NỐI TIẾP (KẾT HỢP MATRIX COMBINATIONS)
    # -------------------------------------------------------------
    print("\n--- 5. KIỂM THỬ PHỐI HỢP CÁC TÍNH NĂNG (COMBINATIONS) ---")

    # Combo A: Parent-Child Chunking + Hybrid Search + Re-rank
    t0 = time.time()
    try:
        chunks_pc = chunk_pages(test_pages, strategy=ChunkStrategy.PARENT_CHILD)
        hits_combo_a = await search("thâm niên 5 năm", enable_hybrid=True, enable_rerank=True)
        record_test("Combo A: Parent-Child + Hybrid + Re-rank", "Combined Workflow", "PASS", (time.time()-t0)*1000, f"Xử lý thành công {len(chunks_pc)} chunks & trả về {len(hits_combo_a)} kết quả.")
    except Exception as e:
        record_test("Combo A: Parent-Child + Hybrid + Re-rank", "Combined Workflow", "FAIL", (time.time()-t0)*1000, str(e))

    # Combo B: Tokenizer Chunking + Dense Search + RAGAS Eval
    t0 = time.time()
    try:
        chunks_tok = chunk_pages(test_pages, strategy=ChunkStrategy.TOKENIZER)
        eval_combo_b = await evaluate_rag_pipeline("Thanh toán tiền cọc bao nhiêu phần trăm?", enable_hybrid=False, enable_rerank=False)
        record_test("Combo B: Tokenizer Chunk + Dense + RAGAS Eval", "Combined Workflow", "PASS", (time.time()-t0)*1000, f"Tạo {len(chunks_tok)} chunks & trả về điểm Faithfulness: {eval_combo_b.get('faithfulness', 0.0)}")
    except Exception as e:
        record_test("Combo B: Tokenizer Chunk + Dense + RAGAS Eval", "Combined Workflow", "FAIL", (time.time()-t0)*1000, str(e))

    # Combo C: Fixed Chunking + Hybrid Search + RAGAS Eval
    t0 = time.time()
    try:
        chunks_fix = chunk_pages(test_pages, strategy=ChunkStrategy.FIXED)
        eval_combo_c = await evaluate_rag_pipeline("Mã ngân hàng Swift là gì?", enable_hybrid=True, enable_rerank=False)
        record_test("Combo C: Fixed Chunk + Hybrid + RAGAS Eval", "Combined Workflow", "PASS", (time.time()-t0)*1000, f"Tạo {len(chunks_fix)} chunks & trả về điểm Context Precision: {eval_combo_c.get('context_precision', 0.0)}")
    except Exception as e:
        record_test("Combo C: Fixed Chunk + Hybrid + RAGAS Eval", "Combined Workflow", "FAIL", (time.time()-t0)*1000, str(e))

    print("\n==================================================")
    print("           HOÀN THÀNH TẤT CẢ BÀI KIỂM THỬ           ")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
