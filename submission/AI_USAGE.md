# Khai báo sử dụng AI

**Công cụ:** Claude Code (Anthropic, model Claude Opus 5.5).

**Phạm vi hỗ trợ:**

- Cài môi trường và chạy headless 8 notebook, `verify_lite`, `pytest`, `benchmark`.
- Debug các lỗi khi chạy trên Windows và Feast 0.66: `localhost` bị trễ do IPv6, chi phí tạo `httpx`
  client cho mỗi request, entity Feast thiếu `value_type`. Đo và khoanh vùng nguyên nhân hybrid P99 cao
  (thread pool mặc định của ONNX Runtime trên CPU 20 luồng), sau đó thêm biến tuỳ chọn `EMBED_THREADS`.
- Soạn nháp phần "📝 Phân tích kết quả" ở cuối mỗi notebook, `INFO.md`, `REFLECTION.md`, và script render
  screenshots từ output đã lưu. Tất cả dựa trên số liệu của lần chạy thật.

**Không dùng AI để:** tạo số liệu hay output giả, sửa output bằng tay, hay hạ ngưỡng rubric để báo PASS.
Mọi con số trong notebook và screenshots là output thật từ máy này.
