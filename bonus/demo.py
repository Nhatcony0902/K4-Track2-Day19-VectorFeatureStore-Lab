"""5-query demo for HybridMemoryAgent.  Run from the repo root:  python bonus/demo.py"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("EMBED_THREADS", "4")      # see NB3 / app/embeddings.py
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402
from feast import FeatureStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import REPO, HybridMemoryAgent  # noqa: E402

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def setup_feature_store() -> None:
    """Offline parquet -> feast apply -> materialize the batch view (like NB4, via Python API)."""
    data = REPO / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir(parents=True)
    pd.DataFrame([
        {"user_id": "u_001", "preferred_language": "vi", "reading_speed_wpm": 230,
         "topic_affinity": "cloud", "active_hour": 22, "event_timestamp": NOW - timedelta(days=1)},
        {"user_id": "u_002", "preferred_language": "mix", "reading_speed_wpm": 310,
         "topic_affinity": "security", "active_hour": 9, "event_timestamp": NOW - timedelta(days=1)},
    ]).to_parquet(data / "user_profile.parquet")
    pd.DataFrame([{"user_id": "u_000", "event_timestamp": NOW - timedelta(days=2),
                   "queries_last_hour": 0, "top_topic_1h": "other", "last_query": "",
                   "avg_query_words_1h": 0.0}]).to_parquet(data / "recent_activity.parquet")

    spec = importlib.util.spec_from_file_location("definitions", REPO / "definitions.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    fs = FeatureStore(repo_path=str(REPO))
    fs.apply([d.user, d.profile_source, d.activity_push, d.user_profile, d.recent_activity])
    fs.materialize(start_date=NOW - timedelta(days=31), end_date=NOW, feature_views=["user_profile"])


MEMORIES = [  # (user, source, days_ago, text)
    ("u_001", "chat", 1, "Hôm qua mình hỏi trợ lý cách debug pod Kubernetes bị CrashLoopBackOff. "
     "Trợ lý gợi ý xem kubectl logs --previous và kiểm tra liveness probe. "
     "Mình đã sửa được bằng cách tăng initialDelaySeconds lên 30 giây."),
    ("u_001", "doc", 3, "Horizontal Pod Autoscaler co giãn số replica theo CPU và request per second. "
     "Cluster Autoscaler thêm node khi pod bị pending vì thiếu tài nguyên. "
     "Kết hợp hai cơ chế giúp hạ tầng chịu được lưu lượng tăng đột biến mà không phải can thiệp tay. "
     "Nên đặt min replica đủ cao cho giờ cao điểm để tránh cold start."),
    ("u_001", "note", 2, "Checklist cloud security: bật MFA cho root account, IAM theo nguyên tắc least privilege, "
     "mã hoá S3 bằng KMS, bật CloudTrail và cảnh báo khi có thay đổi security group."),
    ("u_001", "doc", 5, "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân yêu cầu có sự đồng ý của chủ thể dữ liệu, "
     "và phải đánh giá tác động khi chuyển dữ liệu cá nhân ra nước ngoài."),
    ("u_001", "chat", 4, "Mình hỏi nên dùng embedding model nào cho RAG tiếng Việt. "
     "Kết luận: bge-m3 tốt hơn bge-small vì đa ngữ, nhưng nặng hơn và phải index lại toàn bộ."),
    ("u_001", "doc", 20, "Tối ưu chi phí cloud: dùng spot instance cho batch job, rightsizing máy ảo, "
     "đặt budget alert và tắt môi trường dev ngoài giờ làm việc."),
    ("u_002", "note", 1, "Kubernetes RBAC: tạo Role riêng cho team security, không dùng cluster-admin cho CI. "
     "Đây là ghi chú bí mật của u_002."),
]

QUERIES = [
    ("1. Simple (vector hit)", "Tôi đã đọc gì về Kubernetes?"),
    ("2. Needs profile", "Recommend đọc gì tiếp"),
    ("3. Needs fresh activity", "Tôi đang quan tâm gì gần đây?"),
    ("4. Paraphrase", "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. Mixed (hybrid + profile)", "Cho tôi summary cloud security"),
]


def main() -> None:
    setup_feature_store()
    agent = HybridMemoryAgent()
    for user, source, days, text in MEMORIES:
        agent.remember(text, user_id=user, source=source, ts=NOW - timedelta(days=days))
    print(f"remembered {len(MEMORIES)} memories for 2 users\n")

    for label, q in QUERIES:
        ctx = agent.recall(q, user_id="u_001")
        print(f"=== {label}: {q!r}\n{ctx}\n")
        assert "u_002" not in ctx, "privacy leak: u_002 memory surfaced for u_001"

    # Vietnamese typing without diacritics still hits via accent-folded BM25 tokens.
    print("=== bonus check: no-diacritic query 'nghi dinh du lieu ca nhan'")
    print(agent.recall("nghi dinh du lieu ca nhan", user_id="u_001").split("[memories]")[1])
    # Freshness of the Push path (Decision 3): push one event, read it back from the online store.
    import time
    t0 = time.perf_counter()
    agent._log_query("u_001", "giá GPU hiện tại")
    got = agent.fs.get_online_features(features=["recent_activity:last_query"],
                                       entity_rows=[{"user_id": "u_001"}]).to_dict()["last_query"][0]
    print(f"\npush -> online read: {(time.perf_counter() - t0) * 1000:.0f} ms, last_query={got!r}")
    print("all 5 queries done, no cross-user leak")


if __name__ == "__main__":
    main()
