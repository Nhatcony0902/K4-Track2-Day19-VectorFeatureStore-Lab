# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Filtered Search: cái bẫy recall
#
# **Stack:** `app.filters.FilteredIndex` (Qdrant payload filters) + brute-force
# cosine làm ground truth. Maps to deck §3 "Filtered Search: Cái Bẫy Recall".
#
# > Bản năng đầu tiên của mọi người là *"lọc trước cho nhanh"* hoặc *"lấy top-K
# > rồi lọc sau"*. Cả hai đều sai theo hai cách khác nhau. Notebook này **đo**
# > cả ba chiến lược trên cùng một corpus, cùng một query, để bạn thấy con số
# > chứ không phải nghe kể.
#
# Ba chiến lược:
#
# | | Cách làm | Hỏng ở đâu |
# |---|---|---|
# | **post-filter** | ANN trước → bỏ doc không khớp | recall **sập** khi filter chặt |
# | **pre-filter** | lọc trước → quét chính xác subset | luôn đúng nhưng **mất index** |
# | **filtered-ANN** | đưa filter *vào trong* index | cái bạn thực sự muốn |

# %%
import _setup  # noqa: F401
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # local Qdrant warns that payload indexes are no-ops

from app.filters import (FilteredIndex, access_filter, combo_filter,
                         recent_filter, tenant_filter)
from app.metadata import selectivity
from app.search import Searcher

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Xây index có metadata
#
# `FilteredIndex` **tái sử dụng** vector đã tính trong `Searcher` (kéo ngược ra
# khỏi Qdrant bằng `with_vectors=True`) thay vì embed lại 1000 doc. Embedding là
# phần chậm nhất của lab — trả tiền hai lần không dạy ta điều gì.
#
# Metadata (`tenant`, `access`, `published`) được **suy ra** từ `doc_id` bằng
# hash ổn định (`app/metadata.py`), nên corpus gốc và toàn bộ ngưỡng rubric của
# NB1–NB4 không hề thay đổi.

# %%
searcher = Searcher.from_corpus(DATA / "corpus_vn.jsonl")
index = FilteredIndex.from_searcher(searcher)
print(f"docs: {len(index.docs)}   vectors: {index.vectors.shape}")
print("payload mẫu:", {k: index.docs[0][k] for k in
                       ("doc_id", "topic", "tenant", "access", "published")})

# %% [markdown]
# ## 2. Recall cliff theo độ chọn lọc của filter
#
# Ground truth = brute-force cosine **trên đúng subset khớp filter** — tức là
# đáp án không thể chối cãi. Ta đo recall của post-filter và filtered-ANN so với
# nó, ở bốn mức độ chọn lọc khác nhau.

# %%
QUERY = "tự động mở rộng hệ thống theo lưu lượng"

cases = [
    ("không filter",   lambda d: True, None),
    ("access=internal", *access_filter("internal")),
    ("tenant=acme",     *tenant_filter("acme")),
    ("published ≥ 2026", *recent_filter(20260101)),
    ("acme AND ≥2026",  *combo_filter("acme", 20260101)),
]

print(f"{'filter':<18}{'sel%':>7}{'post':>8}{'fANN':>8}{'post_ms':>9}{'fann_ms':>9}")
rows = []
for name, pred, qf in cases:
    sel = selectivity(index.docs, pred) * 100
    truth = index.pre_filter(QUERY, pred, k=10).doc_ids
    post = index.post_filter(QUERY, pred, k=10, fetch_k=10)
    if qf is None:
        fann_r, fann_ms = 1.0, float("nan")
    else:
        f = index.filtered_ann(QUERY, qf, k=10)
        fann_r, fann_ms = f.recall_against(truth), f.latency_ms
    rows.append((name, sel, post.recall_against(truth), fann_r))
    print(f"{name:<18}{sel:7.1f}{post.recall_against(truth):8.2f}{fann_r:8.2f}"
          f"{post.latency_ms:9.1f}{fann_ms:9.1f}")

# %% [markdown]
# **Đọc bảng:** filter càng chặt (`sel%` càng nhỏ), post-filter càng sập. Ở
# `acme AND ≥2026` (~4% corpus) post-filter thường về **0.00** — nó hỏi index
# 10 doc gần nhất *toàn corpus*, rồi vứt gần hết. Không có exception, không có
# log lỗi: chỉ là câu trả lời tệ đi một cách im lặng.
#
# filtered-ANN giữ **1.00** ở mọi mức, vì filter nằm *bên trong* vòng duyệt.

# %% [markdown]
# ## 3. "Cứ lấy nhiều hơn thì sao?" — mua lại recall bằng over-fetch
#
# Cách sửa của người lười: post-filter nhưng `fetch_k` thật lớn. Nó *có* hiệu
# quả — câu hỏi là bạn phải quét bao nhiêu corpus để đạt được điều đó.

# %%
pred, qf = combo_filter("acme", 20260101)
QUERIES = [QUERY, "bảo mật xác thực người dùng", "mô hình ngôn ngữ lớn"]
truths = {q: index.pre_filter(q, pred, k=10).doc_ids for q in QUERIES}

print(f"selectivity = {selectivity(index.docs, pred)*100:.1f}%  của 1000 doc\n")
print(f"{'fetch_k':>9}{'recall':>9}{'% corpus quét':>16}")
for fk in (10, 50, 200, 500, 1000):
    r = sum(index.post_filter(q, pred, k=10, fetch_k=fk).recall_against(truths[q])
            for q in QUERIES) / len(QUERIES)
    print(f"{fk:>9}{r:9.2f}{fk/len(index.docs)*100:15.0f}%")

r = sum(index.filtered_ann(q, qf, k=10).recall_against(truths[q]) for q in QUERIES) / len(QUERIES)
print(f"{'fANN':>9}{r:9.2f}{10/len(index.docs)*100:15.0f}%")

# %% [markdown]
# Recall quay lại 1.00 — nhưng chỉ khi `fetch_k` ≈ **một nửa corpus**. Lúc đó
# bạn đã bỏ index và đang làm brute-force với các bước thừa. filtered-ANN đạt
# đúng kết quả đó khi chỉ lấy 10.
#
# > **Lưu ý về môi trường lab:** Qdrant chạy in-memory (local mode) *lọc đúng*
# > nhưng bỏ qua payload index, nên cột latency ở đây chỉ mang tính minh hoạ.
# > Trên Qdrant server (path Docker), payload index là thứ giữ filtered-ANN
# > nhanh khi corpus lớn. Bài học về **recall** thì đúng ở cả hai mode.

# %% [markdown]
# ## 4. Bài test bắt buộc trước khi lên production
#
# Đừng chỉ test với query trống filter. Chạy golden set với filter **chọn lọc
# mạnh** — đó là lúc hệ thống gãy.

# %%
for tenant in ("acme", "globex", "initech"):
    pred_t, qf_t = tenant_filter(tenant)
    truth = index.pre_filter(QUERY, pred_t, k=10).doc_ids
    post = index.post_filter(QUERY, pred_t, k=10, fetch_k=10)
    fann = index.filtered_ann(QUERY, qf_t, k=10)
    print(f"tenant={tenant:<9} sel={selectivity(index.docs, pred_t)*100:5.1f}%  "
          f"post={post.recall_against(truth):.2f}  fANN={fann.recall_against(truth):.2f}")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Bảng §2: recall theo độ chọn lọc — post-filter sập, filtered-ANN giữ 1.00.
# 2. Bảng §3: over-fetch ladder — `fetch_k` cần ~50% corpus mới cứu được recall.
# 3. Bảng §4: cả ba tenant, post-filter thua ở mọi tenant.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** vòng lặp in bảng, format `%`, việc dựng `models.Filter`
# từ dict điều kiện. Đây là boilerplate, AI viết đúng ngay.
#
# **Think hard yourself:** *định nghĩa ground truth*. Rất nhiều người đo recall
# của post-filter so với **top-K không filter** — và kết luận sai rằng post-filter
# ổn. Ground truth đúng phải là "top-K chính xác **trong subset khớp filter**".
# Nếu bạn để AI tự chọn baseline, nó thường chọn cái tiện chứ không phải cái đúng,
# và cả bài đo trở thành vô nghĩa. Tự viết `exact_top_k()` và tự kiểm tra nó.

# %% [markdown]
# ## 📝 Phân tích kết quả (Phạm Long Nhật)
#
# - **Post-filter sập khi filter chặt:** recall là 1,00 khi không filter, rơi xuống 0,20 khi filter giữ
#   13–32% corpus, và **0,00 với `acme AND ≥2026` (3,8%)**. Lấy top-10 trước rồi mới lọc thì gần như
#   không doc nào lọt qua.
# - **Filtered-ANN giữ recall 1,00** ở mọi độ chọn lọc, vì filter được áp ngay *trong* lúc tìm kiếm.
# - **Over-fetch ladder** (filter 3,8%): fetch_k=200 (20% corpus) mới đạt 0,80, phải tới **fetch_k=500,
#   khoảng 50% corpus**, mới cứu được recall 1,00. Như vậy gần như là quét toàn bộ, mất hết lợi ích của ANN.
# - Về cột ms: ở đây fANN chậm hơn post-filter, có lẽ do Qdrant chạy in-memory (chế độ lite). Trên Qdrant
#   server có HNSW và payload index thì tương quan này có thể khác, nhưng lab không đo trường hợp đó.
#   Kết luận: chọn chiến lược filter theo **độ chọn lọc**, và phải đo recall chứ đừng chỉ đo latency.
