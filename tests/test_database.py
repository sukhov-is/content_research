import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database
from storage.models import Classification, KeywordMatch, Message, QueryScope, Run


def _scope(channels: list[str] | None = None) -> QueryScope:
    return QueryScope(
        channels=channels or ["@ch"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
    )


@pytest.mark.asyncio
async def test_create_and_get_run(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        run = Run(config_hash="abc123", status="pending", user_id=42)
        run_id = await db.create_run(run)
        assert run_id is not None

        fetched = await db.get_run(run_id)
        assert fetched is not None
        assert fetched.config_hash == "abc123"
        assert fetched.user_id == 42
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_insert_and_count_messages(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        msgs = [
            Message(
                channel="@ch1",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"url/{i}",
            )
            for i in range(5)
        ]
        await db.insert_messages_batch(msgs)
        scope = QueryScope(
            channels=["@ch1"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )
        count = await db.count_messages(scope)
        assert count == 5
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_message_deduplication(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        msg = Message(
            channel="@ch",
            message_id=42,
            date="2026-03-01T12:00:00+00:00",
            text="hello",
            url="u/42",
        )
        await db.insert_message(msg)
        await db.insert_message(msg)  # duplicate — should be ignored
        scope = _scope()
        count = await db.count_messages(scope)
        assert count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unclassified_messages(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(3)
        ]
        await db.insert_messages_batch(msgs)

        unclassified = await db.get_unclassified_messages(scope)
        assert len(unclassified) == 3

        all_msgs = await db.get_messages(scope)
        first = all_msgs[0]
        cls_obj = Classification(
            message_db_id=first.id,
            mentions_disability=True,
            categories_json=json.dumps(["medical"]),
            tones_json=json.dumps(["neutral"]),
        )
        await db.insert_classification(cls_obj)

        unclassified = await db.get_unclassified_messages(scope)
        assert len(unclassified) == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_classification_stats(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(4)
        ]
        await db.insert_messages_batch(msgs)
        all_msgs = await db.get_messages(scope)

        for i, m in enumerate(all_msgs):
            cls_obj = Classification(
                message_db_id=m.id,
                mentions_disability=(i % 2 == 0),
                categories_json=json.dumps(["medical"] if i < 2 else ["fundraising"]),
                tones_json=json.dumps(["neutral"]),
            )
            await db.insert_classification(cls_obj)

        stats = await db.get_classification_stats(scope)
        assert stats["total_classified"] == 4
        assert stats["mentions_disability"] == 2

        medical_msgs = await db.get_messages_by_category(scope, "medical")
        assert len(medical_msgs) == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_classifications_for_keyword(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(3)
        ]
        await db.insert_messages_batch(msgs)
        all_msgs = await db.get_messages(scope)

        await db.insert_keyword_matches([
            KeywordMatch(message_db_id=all_msgs[0].id, keyword="инвалидность", match_count=1),
            KeywordMatch(message_db_id=all_msgs[1].id, keyword="инвалидность", match_count=2),
            KeywordMatch(message_db_id=all_msgs[2].id, keyword="ОВЗ", match_count=1),
        ])

        for m in all_msgs:
            cls_obj = Classification(
                message_db_id=m.id,
                mentions_disability=True,
                categories_json=json.dumps(["medical"]),
                tones_json=json.dumps(["neutral"]),
            )
            await db.insert_classification(cls_obj)

        rows = await db.get_classifications_for_keyword(scope, "инвалидность")
        assert len(rows) == 2
        assert all(r["mentions_disability"] for r in rows)
        assert rows[0]["categories"] == ["medical"]

        rows_ovz = await db.get_classifications_for_keyword(scope, "ОВЗ")
        assert len(rows_ovz) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_keywords_by_message(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(3)
        ]
        await db.insert_messages_batch(msgs)
        all_msgs = await db.get_messages(scope)

        await db.insert_keyword_matches([
            KeywordMatch(message_db_id=all_msgs[0].id, keyword="инвалидность", match_count=1),
            KeywordMatch(message_db_id=all_msgs[0].id, keyword="ОВЗ", match_count=1),
            KeywordMatch(message_db_id=all_msgs[2].id, keyword="ОВЗ", match_count=1),
        ])

        mapping = await db.get_keywords_by_message(scope)
        assert set(mapping[all_msgs[0].id]) == {"инвалидность", "ОВЗ"}
        assert all_msgs[1].id not in mapping
        assert mapping[all_msgs[2].id] == ["ОВЗ"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_matched_patterns_by_message(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(3)
        ]
        await db.insert_messages_batch(msgs)
        all_msgs = await db.get_messages(scope)

        await db.insert_keyword_matches([
            KeywordMatch(
                message_db_id=all_msgs[0].id, keyword="Группа1", match_count=2,
                matched_patterns_json=json.dumps(["инвалид\\w*", "ОВЗ"]),
            ),
            KeywordMatch(
                message_db_id=all_msgs[0].id, keyword="Группа2", match_count=1,
                matched_patterns_json=json.dumps(["доступная среда"]),
            ),
            KeywordMatch(
                message_db_id=all_msgs[2].id, keyword="Группа1", match_count=1,
                matched_patterns_json=json.dumps(["ОВЗ"]),
            ),
        ])

        mapping = await db.get_matched_patterns_by_message(scope)
        assert set(mapping[all_msgs[0].id]) == {"инвалид\\w*", "ОВЗ", "доступная среда"}
        assert all_msgs[1].id not in mapping
        assert mapping[all_msgs[2].id] == ["ОВЗ"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_pattern_stats(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        msgs = [
            Message(
                channel="@ch",
                message_id=i,
                date="2026-06-15T10:00:00+00:00",
                text=f"msg {i}",
                url=f"u/{i}",
            )
            for i in range(3)
        ]
        await db.insert_messages_batch(msgs)
        all_msgs = await db.get_messages(scope)

        await db.insert_keyword_matches([
            KeywordMatch(
                message_db_id=all_msgs[0].id, keyword="Группа1", match_count=2,
                matched_patterns_json=json.dumps(["инвалид\\w*", "ОВЗ"]),
            ),
            KeywordMatch(
                message_db_id=all_msgs[1].id, keyword="Группа1", match_count=1,
                matched_patterns_json=json.dumps(["инвалид\\w*"]),
            ),
            KeywordMatch(
                message_db_id=all_msgs[2].id, keyword="Группа2", match_count=1,
                matched_patterns_json=json.dumps(["доступная среда"]),
            ),
        ])

        result = await db.get_pattern_stats(scope)
        assert result["Группа1"]["инвалид\\w*"] == 2
        assert result["Группа1"]["ОВЗ"] == 1
        assert result["Группа2"]["доступная среда"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_data_persists_across_connections(tmp_path):
    """Data survives a close+reconnect cycle (no more DROP on connect)."""
    db_path = tmp_path / "persist.db"

    db = Database(db_path)
    await db.connect()
    await db.insert_message(
        Message(
            channel="@ch",
            message_id=1,
            date="2026-05-01T00:00:00+00:00",
            text="persisted",
            url="u/1",
        )
    )
    await db.close()

    db2 = Database(db_path)
    await db2.connect()
    try:
        scope = _scope()
        count = await db2.count_messages(scope)
        assert count == 1
        msgs = await db2.get_messages(scope)
        assert msgs[0].text == "persisted"
    finally:
        await db2.close()


@pytest.mark.asyncio
async def test_scope_date_filtering(tmp_path):
    """Only messages within scope dates are returned."""
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        await db.insert_messages_batch([
            Message(channel="@ch", message_id=1, date="2026-01-15T00:00:00+00:00", text="jan", url="u/1"),
            Message(channel="@ch", message_id=2, date="2026-02-15T00:00:00+00:00", text="feb", url="u/2"),
            Message(channel="@ch", message_id=3, date="2026-03-15T00:00:00+00:00", text="mar", url="u/3"),
        ])

        feb_scope = QueryScope(
            channels=["@ch"],
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        )
        msgs = await db.get_messages(feb_scope)
        assert len(msgs) == 1
        assert msgs[0].text == "feb"

        full_scope = QueryScope(
            channels=["@ch"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 31),
        )
        msgs = await db.get_messages(full_scope)
        assert len(msgs) == 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fetch_log_and_metadata(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        assert await db.get_fetched_ranges("@ch") == []
        await db.record_fetch("@ch", "2026-01-01", "2026-01-31")
        await db.record_fetch("@ch", "2026-03-01", "2026-03-31")
        ranges = await db.get_fetched_ranges("@ch")
        assert len(ranges) == 2
        assert ranges[0] == ("2026-01-01", "2026-01-31")

        assert await db.get_metadata("key1") is None
        await db.set_metadata("key1", "val1")
        assert await db.get_metadata("key1") == "val1"
        await db.set_metadata("key1", "val2")
        assert await db.get_metadata("key1") == "val2"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_delete_all_keyword_matches(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        await db.insert_message(
            Message(channel="@ch", message_id=1, date="2026-06-01T00:00:00+00:00", text="t", url="u/1")
        )
        msgs = await db.get_messages(scope)
        await db.insert_keyword_matches([
            KeywordMatch(message_db_id=msgs[0].id, keyword="kw", match_count=1),
        ])
        stats = await db.get_keyword_stats(scope)
        assert len(stats) == 1

        await db.delete_all_keyword_matches()
        stats = await db.get_keyword_stats(scope)
        assert len(stats) == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_messages_without_keyword_matches(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    try:
        scope = _scope()
        await db.insert_messages_batch([
            Message(channel="@ch", message_id=1, date="2026-06-01T00:00:00+00:00", text="matched", url="u/1"),
            Message(channel="@ch", message_id=2, date="2026-06-01T00:00:00+00:00", text="not matched", url="u/2"),
        ])
        msgs = await db.get_messages(scope)
        await db.insert_keyword_matches([
            KeywordMatch(message_db_id=msgs[0].id, keyword="kw", match_count=1),
        ])

        unmatched = await db.get_messages_without_keyword_matches(scope)
        assert len(unmatched) == 1
        assert unmatched[0].text == "not matched"
    finally:
        await db.close()
