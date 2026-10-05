from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Run:
    id: int | None = None
    config_hash: str = ""
    status: str = "pending"
    user_id: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_messages: int = 0
    processed_messages: int = 0


@dataclass
class Message:
    id: int | None = None
    channel: str = ""
    message_id: int = 0
    date: datetime = field(default_factory=datetime.utcnow)
    text: str = ""
    url: str = ""
    media_type: str | None = None
    views: int = 0
    forwards: int = 0


@dataclass
class KeywordMatch:
    id: int | None = None
    message_db_id: int = 0
    keyword: str = ""
    match_count: int = 0
    matched_patterns_json: str = "[]"


@dataclass
class Classification:
    id: int | None = None
    message_db_id: int = 0
    mentions_disability: bool = False
    categories_json: str = "[]"
    tones_json: str = "[]"
    problem_scope: str = ""
    disability_model: str = ""
    agency: str = ""
    portrayal: str = ""
    target_audiences_json: str = "[]"
    barriers_json: str = "[]"
    stigma: str = ""
    raw_response: str = ""


@dataclass
class QueryScope:
    channels: list[str]
    start_date: date
    end_date: date
