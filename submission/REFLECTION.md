# Reflection — Lab 19

**Tên:** Phạm Long Nhật (2A202602844)
**Cohort:** A20-K4
**Path đã chạy:** lite (fastembed bge-small + Qdrant in-memory + Feast SQLite), Windows 11

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 trung bình: hybrid 78,6% > BM25 77,8% > vector 73,2%.

- **`exact`:** BM25 và hybrid hoà nhau (96,7%), vector thấp hơn (88,7%). Từ kỹ thuật xuất hiện nguyên văn
  trong doc, nên khớp từ khoá là đủ.
- **`paraphrase`:** cả ba mode đều thấp, và BM25 (33%) lại cao hơn vector (24%). `bge-small-en` là model
  tiếng Anh nên hiểu kém câu tiếng Việt được diễn đạt lại. Muốn vector thắng ở đây phải đổi sang model
  đa ngữ như bge-m3.
- **`mixed`:** hybrid thắng tuyệt đối (100%), vì RRF cộng điểm cho doc được cả hai retriever cùng xếp cao.

Nhìn chung, hybrid thắng nhờ ổn định trên mọi loại query chứ không phải nhờ tốt nhất ở từng loại.

Không nên dùng hybrid khi:
- Query là mã, ID hoặc tên riêng (mã lỗi, SKU, tên hàm): BM25 thuần chính xác hơn và rẻ hơn.
- Ngân sách latency rất chặt: ở lab này bước embed chiếm phần lớn P99, nên dùng BM25 thuần.
- Dữ liệu đa ngôn ngữ, đa phương thức, hoặc người dùng hỏi bằng ngôn ngữ tự nhiên khác hẳn từ ngữ trong
  tài liệu: vector thuần với một model tốt là đủ, vì BM25 chỉ thêm nhiễu.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid P99 lúc đầu là 236 ms, nguyên nhân không nằm ở thuật toán. ONNX Runtime mặc định dùng 20 luồng trên
CPU lai P/E-core, làm việc embed 1 query chậm khoảng 10 lần. Giới hạn còn 4 luồng thì P99 giảm xuống 36 ms.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(không)_
