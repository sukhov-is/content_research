import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.keywords import _compile_pattern, _compile_group, analyze_keywords
from config.models import KeywordGroupConfig
from parser.fetcher import compute_gaps
from storage.database import Database
from storage.models import Message, QueryScope


def _scope() -> QueryScope:
    return QueryScope(
        channels=["@test", "@ch"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_compile_plain_keyword():
    pattern = _compile_pattern("инвалидность")
    assert pattern.search("Тема инвалидность в СМИ")
    assert not pattern.search("Тема валидности")


def test_compile_regex_keyword():
    pattern = _compile_pattern(r"инвалид\w*")
    assert pattern.search("инвалидность")
    assert pattern.search("инвалидам")
    assert pattern.search("инвалид")


def test_compile_group():
    group = KeywordGroupConfig(label="Инвалидность", patterns=[r"инвалид\w*", "ОВЗ"])
    pattern = _compile_group(group)
    assert pattern.search("инвалидность")
    assert pattern.search("инвалидам")
    assert pattern.search("ОВЗ")
    assert not pattern.search("обычный текст")


@pytest.mark.asyncio
async def test_keyword_analysis(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    await db.connect()

    try:
        scope = _scope()
        messages = [
            Message(
                channel="@test", message_id=1,
                date="2026-06-15T10:00:00+00:00",
                text="Тема инвалидности обсуждается", url="https://t.me/test/1",
            ),
            Message(
                channel="@test", message_id=2,
                date="2026-06-15T10:00:00+00:00",
                text="Обычное сообщение без ключевых слов", url="https://t.me/test/2",
            ),
            Message(
                channel="@test", message_id=3,
                date="2026-06-15T10:00:00+00:00",
                text="ОВЗ и инвалидность -- актуальные темы", url="https://t.me/test/3",
            ),
        ]
        await db.insert_messages_batch(messages)

        groups = [
            KeywordGroupConfig(label="Инвалидность", patterns=[r"инвалид\w*"]),
            KeywordGroupConfig(label="ОВЗ", patterns=["ОВЗ"]),
        ]
        counts = await analyze_keywords(groups, scope, db)

        assert counts["Инвалидность"] == 2
        assert counts["ОВЗ"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_keyword_analysis_stores_matched_patterns(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()

    try:
        scope = _scope()
        await db.insert_messages_batch([
            Message(
                channel="@ch", message_id=1,
                date="2026-06-15T10:00:00+00:00",
                text="ОВЗ и инвалидность -- актуальные темы", url="u/1",
            ),
            Message(
                channel="@ch", message_id=2,
                date="2026-06-15T10:00:00+00:00",
                text="Обычный текст", url="u/2",
            ),
        ])

        groups = [
            KeywordGroupConfig(
                label="Инвалидность (общее)",
                patterns=[r"инвалид\w*", "ОВЗ"],
            ),
        ]
        counts = await analyze_keywords(groups, scope, db)
        assert counts["Инвалидность (общее)"] == 1

        patterns_map = await db.get_matched_patterns_by_message(scope)
        all_msgs = await db.get_messages(scope)
        msg1_id = all_msgs[0].id

        assert msg1_id in patterns_map
        assert set(patterns_map[msg1_id]) == {r"инвалид\w*", "ОВЗ"}
    finally:
        await db.close()


# ── compute_gaps tests ──

def test_compute_gaps_no_fetched():
    gaps = compute_gaps(date(2026, 1, 1), date(2026, 3, 31), [])
    assert gaps == [(date(2026, 1, 1), date(2026, 3, 31))]


def test_compute_gaps_fully_covered():
    gaps = compute_gaps(
        date(2026, 2, 1),
        date(2026, 2, 28),
        [(date(2026, 1, 1), date(2026, 3, 31))],
    )
    assert gaps == []


def test_compute_gaps_partial_overlap():
    gaps = compute_gaps(
        date(2026, 1, 1),
        date(2026, 3, 31),
        [(date(2026, 2, 1), date(2026, 2, 28))],
    )
    assert gaps == [
        (date(2026, 1, 1), date(2026, 1, 31)),
        (date(2026, 3, 1), date(2026, 3, 31)),
    ]


def test_compute_gaps_overlapping_fetched():
    gaps = compute_gaps(
        date(2026, 1, 1),
        date(2026, 6, 30),
        [
            (date(2026, 1, 15), date(2026, 2, 15)),
            (date(2026, 2, 10), date(2026, 3, 10)),
        ],
    )
    assert gaps == [
        (date(2026, 1, 1), date(2026, 1, 14)),
        (date(2026, 3, 11), date(2026, 6, 30)),
    ]
