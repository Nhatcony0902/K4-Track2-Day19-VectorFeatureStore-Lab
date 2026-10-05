"""Feature definitions for the HybridMemoryAgent (see bonus/ARCHITECTURE.md §Decision 2).

Two views, two freshness contracts:

* user_profile    — stable, slow-moving. Batch source, TTL 30 days.
* recent_activity — fast-moving. PushSource (streaming Push API), TTL 1 hour:
                    a "what did I ask in the last hour" value older than an hour
                    is wrong, not just stale, so Feast must return NULL instead.
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, PushSource, ValueType
from feast.types import Float64, Int64, String

DATA = Path(__file__).resolve().parent / "data"

user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)

profile_source = FileSource(
    name="user_profile_source",
    path=str(DATA / "user_profile.parquet"),
    timestamp_field="event_timestamp",
)

user_profile = FeatureView(
    name="user_profile",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="preferred_language", dtype=String),   # vi | en | mix
        Field(name="reading_speed_wpm", dtype=Int64),
        Field(name="topic_affinity", dtype=String),       # dominant topic, 30-day window
        Field(name="active_hour", dtype=Int64),           # modal local hour of activity
    ],
    source=profile_source,
    online=True,
)

activity_batch = FileSource(
    name="recent_activity_batch",
    path=str(DATA / "recent_activity.parquet"),
    timestamp_field="event_timestamp",
)

activity_push = PushSource(name="activity_push", batch_source=activity_batch)

recent_activity = FeatureView(
    name="recent_activity",
    entities=[user],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="queries_last_hour", dtype=Int64),
        Field(name="top_topic_1h", dtype=String),
        Field(name="last_query", dtype=String),
        Field(name="avg_query_words_1h", dtype=Float64),
    ],
    source=activity_push,
    online=True,
)
