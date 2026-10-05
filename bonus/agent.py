"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile & recent activity (Feast).

Design rationale lives in bonus/ARCHITECTURE.md; comments here only point at it.
No LLM is called: recall() returns the context string an LLM would receive.
"""
from __future__ import annotations

import re
import sys
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.embeddings import Embedder  # noqa: E402  -- reuses EMBEDDING_BACKEND / EMBED_THREADS

REPO = Path(__file__).resolve().parent / "feature_repo"
RRF_K = 10   # not 60: per-user memory lists are ~10-100 long (ARCHITECTURE.md §Decision 2)
WEIGHTS = {"bm25": 1.0, "vector": 1.0}
PROFILE_WEIGHT = {"open": 1.0, "topical": 0.1}   # Decision 2: trust the query when it names a topic
TOPICS = {  # tiny keyword tagger; a real system would use a classifier
    "cloud": ["cloud", "đám mây", "kubernetes", "k8s", "pod", "autoscal", "co giãn", "mở rộng",
              "replica", "node", "aws", "serverless", "hạ tầng"],
    "security": ["security", "bảo mật", "iam", "mã hoá", "mã hóa", "xác thực", "lỗ hổng"],
    "ai_ml": ["llm", "embedding", "model", "mô hình", "rag", "fine-tune"],
    "legal": ["nghị định", "pháp luật", "luật", "decree", "hợp đồng"],
}
PROFILE_FEATURES = ["user_profile:preferred_language", "user_profile:reading_speed_wpm",
                    "user_profile:topic_affinity", "user_profile:active_hour",
                    "recent_activity:queries_last_hour", "recent_activity:top_topic_1h",
                    "recent_activity:last_query"]


# Function words that carry no topic; whitespace BM25 would otherwise match "về" / "gì" across
# every memory. Listed with and without diacritics because tokenize() emits both.
STOPWORDS = {"tôi", "toi", "mình", "minh", "đã", "da", "gì", "gi", "về", "ve", "là", "la", "và", "va",
             "của", "cua", "cho", "có", "co", "không", "khong", "đang", "dang", "nào", "nao",
             "được", "duoc", "các", "cac", "những", "nhung", "một", "mot", "này", "nay"}


def fold(s: str) -> str:
    """Strip Vietnamese diacritics: 'mở rộng' -> 'mo rong' (users often type without them)."""
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokenize(s: str, twins: bool = False) -> list[str]:
    """Whitespace syllables; documents also get accent-folded twins (ARCHITECTURE.md §Vietnamese).

    Only the document side is folded: an accented query ("dữ liệu") matches exactly, while an
    unaccented one ("du lieu") is already folded and hits the twins. Folding the query too would
    make "dữ" collide with "đủ"/"dư".
    """
    syl = [t for t in re.findall(r"\w+", unicodedata.normalize("NFC", s.lower())) if t not in STOPWORDS]
    # Syllable bigrams approximate Vietnamese compound words ("tài_liệu" != "dữ_liệu",
    # "hạ_tầng") without pulling in pyvi/underthesea.
    toks = syl + [f"{x}_{y}" for x, y in zip(syl, syl[1:])]
    if twins:
        toks = toks + [f for f in map(fold, toks) if f not in toks]
    return toks


def tag_topic(s: str) -> str:
    low, f = s.lower(), fold(s)
    scores = {t: sum((k in low) or (fold(k) in f) for k in kws) for t, kws in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "other"


def chunk(text: str, max_words: int = 60) -> list[str]:
    """Decision 1: sentence-packed chunks of <= max_words, 1-sentence overlap."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-1:]                      # overlap: carry the last sentence over
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class HybridMemoryAgent:
    def __init__(self, repo: Path = REPO) -> None:
        from feast import FeatureStore
        self.embedder = Embedder()
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection("memories", vectors_config=VectorParams(
            size=self.embedder.dim, distance=Distance.COSINE))
        self.fs = FeatureStore(repo_path=str(repo))
        self._chunks: dict[str, list[dict]] = {}         # user_id -> payloads (for BM25)
        self._queries: dict[str, list[tuple[datetime, str]]] = {}

    # ── write path ────────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001", source: str = "note",
                 ts: datetime | None = None) -> None:
        """Add a new piece of episodic memory for this user."""
        ts = ts or datetime.now(timezone.utc)
        pieces = chunk(text)
        vectors = list(self.embedder.embed(pieces))
        points = []
        for piece, vec in zip(pieces, vectors):
            payload = {"user_id": user_id, "text": piece, "source": source,
                       "topic": tag_topic(piece), "ts": ts.isoformat()}
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload))
            self._chunks.setdefault(user_id, []).append(payload)
        self.qdrant.upsert("memories", points=points)

    # ── read path ─────────────────────────────────────────────────────────
    def recall(self, query: str, user_id: str = "u_001", k: int = 3) -> str:
        """Retrieve top-K memories + user profile features -> return assembled context."""
        f = {n: v[0] for n, v in self.fs.get_online_features(
            features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]).to_dict().items()}
        ranked = self._hybrid(query, user_id, f.get("topic_affinity"))[:k]
        self._log_query(user_id, query)        # Decision 3: push AFTER reading -> no self-echo

        recent = (f"{f['queries_last_hour']} queries in the last hour, mostly about "
                  f"{f['top_topic_1h']} (last: \"{f['last_query']}\")"
                  if f.get("queries_last_hour") else "no queries in the last hour")
        lines = [f"[profile] language={f['preferred_language']}, reads ~{f['reading_speed_wpm']} wpm, "
                 f"interested in {f['topic_affinity']}, usually active around {f['active_hour']}h",
                 f"[recent] {recent}",
                 "[memories]"]
        for i, (p, score, why) in enumerate(ranked, 1):
            lines.append(f"  {i}. ({p['source']}, {p['ts'][:10]}, {p['topic']}, "
                         f"rrf={score:.4f} via {'+'.join(why)}) {p['text'][:110]}")
        return "\n".join(lines)

    def _hybrid(self, query: str, user_id: str, affinity: str | None):
        mine = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        qv = next(self.embedder.embed([query])).tolist()
        vec = [p.payload for p in self.qdrant.query_points(
            "memories", query=qv, query_filter=mine, limit=20).points]   # isolation: filter ALWAYS on

        docs = self._chunks.get(user_id, [])
        bm25 = []
        if docs:
            scores = BM25Okapi([tokenize(d["text"], twins=True) for d in docs]).get_scores(tokenize(query))
            bm25 = [docs[i] for i in sorted(range(len(docs)), key=lambda i: -scores[i]) if scores[i] > 0]
        prof = sorted((d for d in docs if d["topic"] == affinity), key=lambda d: d["ts"], reverse=True)

        weights = {**WEIGHTS, "profile": PROFILE_WEIGHT["open" if tag_topic(query) == "other" else "topical"]}
        fused: dict[str, list] = {}
        for name, hits in (("bm25", bm25), ("vector", vec), ("profile", prof)):
            for rank, p in enumerate(hits, start=1):          # weighted RRF, rank is 1-based
                entry = fused.setdefault(p["text"], [p, 0.0, []])
                entry[1] += weights[name] / (RRF_K + rank)
                entry[2].append(f"{name}#{rank}")
        return sorted(fused.values(), key=lambda e: -e[1])

    def _log_query(self, user_id: str, query: str) -> None:
        """Stand-in for a stream processor: roll a 1-hour window, push to the online store."""
        now = datetime.now(timezone.utc)
        log = [(t, q) for t, q in self._queries.get(user_id, []) if now - t < timedelta(hours=1)]
        log.append((now, query))
        self._queries[user_id] = log
        topics = Counter(t for t in (tag_topic(q) for _, q in log) if t != "other") or Counter(other=1)
        self.fs.push("activity_push", pd.DataFrame([{
            "user_id": user_id, "event_timestamp": now,
            "queries_last_hour": len(log), "top_topic_1h": topics.most_common(1)[0][0],
            "last_query": query, "avg_query_words_1h": sum(len(q.split()) for _, q in log) / len(log),
        }]))
