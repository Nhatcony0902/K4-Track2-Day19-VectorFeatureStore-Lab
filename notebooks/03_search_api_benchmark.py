# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import os
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
# EMBED_THREADS=4: giới hạn số luồng ONNX Runtime (xem app/embeddings.py). Mặc định
# ORT dùng 1 luồng/logical core; trên CPU lai P/E-core (máy chạy lab: 20 luồng) điều
# đó làm query embedding chậm ~10× sau khi server vừa index 1000 doc.
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--port", "8000", "--log-level", "warning"],
    cwd=str(ROOT),
    env={**os.environ, "EMBED_THREADS": os.getenv("EMBED_THREADS", "4")},
)

# Đợi server up + warm (Searcher.from_corpus loads embeddings + indexes 1000 docs)
# 127.0.0.1 thay vì localhost: trên Windows, localhost thử IPv6 (::1) trước,
# uvicorn chỉ bind IPv4 → mỗi request mất thêm ~2 s chờ fallback, làm sai cột P99(wall).
URL = "http://127.0.0.1:8000"
# 180 s: trên laptop Windows của mình, index 1000 doc lúc startup mất ~70 s.
for _ in range(180):
    try:
        r = httpx.get(f"{URL}/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            break
    except httpx.HTTPError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("API didn't become ready within 180s")

print(httpx.get(f"{URL}/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (100 queries × 3 modes)
#
# Dùng 50 golden queries × 2 reps = 100 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


# Một Client dùng chung (keep-alive) như client production thật. `httpx.get()` tạo
# client mới mỗi lần → mở lại kết nối + nạp lại SSL context (~1 s/lần trên Windows),
# khiến cột P99(wall) đo chi phí khởi tạo client chứ không phải request.
client = httpx.Client(base_url=URL, timeout=10.0)


def benchmark_mode(mode: str, reps: int = 2, warmup: int = 10) -> dict[str, float]:
    # Warm-up (README troubleshooting: "Chạy 10 query warmup trước rồi đo lại").
    for q in golden[:warmup]:
        client.get("/search", params={"q": q["query"], "mode": mode})
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = client.get("/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
client.close()
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.

# %% [markdown]
# ## 📝 Phân tích kết quả (Phạm Long Nhật)
#
# - `/search` trả `SearchResponse` hợp lệ có trường `latency_ms`, đo phía server bằng `perf_counter`.
# - **Hybrid P99 phía server dưới 50 ms → PASS.** Keyword nhanh nhất vì chỉ chạy BM25. Semantic và
#   hybrid tốn thời gian chủ yếu ở bước embed query (ONNX trên CPU); RRF và BM25 chỉ thêm vài ms.
# - **Ba chỉnh sửa cho môi trường chạy (không hạ ngưỡng nào), phát hiện khi chạy trên Windows 11, CPU 20 luồng:**
#   1. Đổi `localhost` thành `127.0.0.1`. Windows thử IPv6 `::1` trước, mà uvicorn chỉ bind IPv4, nên mỗi
#      request mất khoảng 2 s chờ quay về IPv4. Lần chạy đầu vì thế bị timeout 900 s.
#   2. Dùng chung một `httpx.Client` và chạy 10 query warm-up cho mỗi mode. `httpx.get()` tạo client mới
#      mỗi lần (khoảng 1,1 s để nạp SSL context), nên cột P99(wall) khi đó đo chi phí tạo client chứ
#      không phải thời gian request.
#   3. Đặt `EMBED_THREADS=4` cho uvicorn. ONNX Runtime mặc định dùng 1 luồng cho mỗi logical core. Mình
#      đo được: embed 1 query mất khoảng 4 ms ở process mới, nhưng 55–70 ms sau khi process vừa index
#      1000 doc (với 20 luồng). Giới hạn 4 luồng giữ được mức 5–15 ms. Trước khi sửa, hybrid P99 là 236 ms (WARN).
# - Bài học production: trong lab này, P99 phụ thuộc vào cấu hình thread pool của runtime suy luận nhiều
#   hơn là vào thuật toán fusion. Luôn đo trên đúng phần cứng sẽ deploy.
