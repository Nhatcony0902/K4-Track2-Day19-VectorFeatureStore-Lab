# K4-Track2-Day19 — Vector Store + Feature Store Lab

| Mục | Giá trị |
|---|---|
| Họ tên | Phạm Long Nhật (Pham Long Nhat) |
| MSSV | 2A202602844 |
| Mã bài | K4-Track2-Day19 |
| Repo | https://github.com/Nhatcony0902/K4-Track2-Day19-VectorFeatureStore-Lab |
| Đường chạy | **Lite**: fastembed `BAAI/bge-small-en-v1.5` (384d) + Qdrant in-memory + Feast (SQLite online / file offline) + FastAPI |
| Hệ điều hành | Windows 11 Pro 10.0.26200, CPU 20 luồng logic |
| Python | 3.11.9 (venv `.venv`, cài bằng `uv pip install -r requirements.txt`) |
| Thư viện | fastembed 0.8.1 · qdrant-client 1.19.1 · onnxruntime 1.30.0 · rank-bm25 0.2.2 · feast 0.66.0 · fastapi 0.142.2 · uvicorn 0.54.0 · polars 1.44.2 · pyarrow 25.0.1 · numpy 2.4.6 |

## Kết quả kiểm tra (chạy từ thư mục gốc repo)

| Lệnh | Kết quả |
|---|---|
| `.venv/Scripts/python scripts/verify_lite.py` | All checks passed |
| `.venv/Scripts/python -m pytest -q` | 41 passed |
| `jupyter nbconvert --execute` cả 8 notebook (`make notebooks`) | 8/8 PASS |
| `EMBED_THREADS=4 .venv/Scripts/python scripts/benchmark.py` (`make benchmark`) | PASS: hybrid hơn keyword 0,8 điểm %, hơn semantic 5,4 điểm %; hybrid P99 27,3 ms |

Notebook đã chạy (giữ output) nằm ở `notebooks/*.ipynb`. Cuối mỗi notebook có cell **"📝 Phân tích kết quả"**.
Riêng NB6 và NB7 có thêm phần trả lời các câu hỏi rubric chấm (vì sao `agentic (+filter)` thấp hơn; chọn
ngưỡng cache nào và vì sao 0,75 chưa đủ). Screenshots trong `submission/screenshots/` được render từ chính
output đã lưu trong các notebook này.

## Số liệu chính

| NB | Tiêu chí | Kết quả |
|---|---|---|
| 1 | count / top-5 / paraphrase | 1000 vector; paraphrase (không có chữ "cloud") → 5/5 `cloud` |
| 2 | P@10 kw / sem / hyb | 77,8% / 73,2% / **78,6%**; `mixed`: hybrid 100% |
| 3 | Hybrid P99 phía server | **36,2 ms** < 50 ms (P50 19,8 ms) |
| 4 | apply / materialize / online / PIT | 3 FV; online P99 **3,43 ms**; PIT 3 rows × 2 features |
| 5 | recall theo độ chọn lọc | post-filter 0,00 ở filter 3,8%; fANN 1,00; cần fetch_k ≈ 50% corpus |
| 6 | agentic so với single-shot (16 doc) | recall 0,526 → 0,906; balance 0,08 → 0,93 |
| 7 | sweep ngưỡng / tenant leak | chọn 0,85 (100% tiết kiệm, 0% sai); 0,75 cho 36% sai; leak → MISS khi namespaced |
| 8 | leakage / PIT / ODFV | gap 0,477 trên `session_id`; +0,120 AUC ảo; ratio 0,03 và 4,21 cho cùng một user |

## Thay đổi so với đề bài (không hạ ngưỡng nào)

Các TODO trong NB1–NB4 đã có sẵn code trong template, mình chạy và kiểm tra lại. Các chỗ sửa đều để
notebook chạy đúng trên Windows và Feast 0.66:

- **NB3:** đổi `localhost` thành `127.0.0.1`. Trên Windows, `localhost` thử IPv6 trước nên mỗi request mất
  khoảng 2 s, và lần chạy đầu bị timeout 900 s.
- **NB3:** dùng chung một `httpx.Client` và chạy 10 query warm-up cho mỗi mode (README có gợi ý). Trước đó
  `httpx.get()` mất khoảng 1,1 s mỗi lần chỉ để tạo client. Thời gian chờ server tăng từ 60 s lên 180 s,
  vì index 1000 doc trên máy này mất khoảng 70 s.
- **`app/embeddings.py` + NB3:** thêm biến môi trường tuỳ chọn `EMBED_THREADS`; không đặt thì hành vi giữ
  nguyên. NB3 khởi động uvicorn với `EMBED_THREADS=4`. Trên CPU 20 luồng, ONNX Runtime dùng cấu hình mặc
  định làm embed 1 query mất 55–70 ms sau khi index, nên hybrid P99 lên tới 236 ms (WARN). Với 4 luồng,
  P99 còn 36 ms.
- **`app/feast_repo_ondemand/definitions.py`:** thêm `value_type=ValueType.STRING` cho entity `user`.
  Feast 0.66 suy ra join key thành kiểu JSON nên NB8 báo lỗi `Invalid JSON string for JSON type` khi
  materialize.
- **NB4:** Feast 0.66 bỏ entity row không có feature hợp lệ tại thời điểm event (`u_001` hỏi lúc NOW−2h,
  nhưng profile mãi tới NOW−1h mới ghi). Mình in kết quả gốc của Feast (2 dòng), sau đó left-join lại với
  `entity_df` để hiện đủ 3 dòng, trong đó dòng `u_001` là NaN, đúng ngữ nghĩa point-in-time.
