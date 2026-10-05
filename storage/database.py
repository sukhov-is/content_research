from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

from storage.models import Classification, KeywordMatch, Message, QueryScope, Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    total_messages INTEGER NOT NULL DEFAULT 0,
    processed_messages INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    media_type TEXT,
    views INTEGER NOT NULL DEFAULT 0,
    forwards INTEGER NOT NULL DEFAULT 0,
    UNIQUE(channel, message_id)
);

CREATE TABLE IF NOT EXISTS keyword_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_db_id INTEGER NOT NULL REFERENCES messages(id),
    keyword TEXT NOT NULL,
    match_count INTEGER NOT NULL DEFAULT 0,
    matched_patterns_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_db_id INTEGER NOT NULL UNIQUE REFERENCES messages(id),
    mentions_disability INTEGER NOT NULL DEFAULT 0,
    categories_json TEXT NOT NULL DEFAULT '[]',
    tones_json TEXT NOT NULL DEFAULT '[]',
    problem_scope TEXT NOT NULL DEFAULT '',
    disability_model TEXT NOT NULL DEFAULT '',
    agency TEXT NOT NULL DEFAULT '',
    portrayal TEXT NOT NULL DEFAULT '',
    target_audiences_json TEXT NOT NULL DEFAULT '[]',
    barriers_json TEXT NOT NULL DEFAULT '[]',
    stigma TEXT NOT NULL DEFAULT '',
    raw_response TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_date ON messages(channel, date);
CREATE INDEX IF NOT EXISTS idx_keyword_matches_msg ON keyword_matches(message_db_id);
CREATE INDEX IF NOT EXISTS idx_classifications_msg ON classifications(message_db_id);
CREATE INDEX IF NOT EXISTS idx_fetch_log_channel ON fetch_log(channel);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        self._conn.row_factory = aiosqlite.Row
        await self._migrate_if_needed()
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def _migrate_if_needed(self) -> None:
        """Drop old-schema tables that still carry a run_id column."""
        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        if not await cur.fetchone():
            return
        cur = await self.conn.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in await cur.fetchall()]
        if "run_id" in columns:
            await self.conn.executescript(
                "DROP TABLE IF EXISTS classifications;"
                "DROP TABLE IF EXISTS keyword_matches;"
                "DROP TABLE IF EXISTS messages;"
            )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "Database not connected"
        return self._conn

    # ── Helpers ──

    @staticmethod
    def _scope_where(scope: QueryScope, alias: str = "m") -> tuple[str, list]:
        ph = ",".join("?" * len(scope.channels))
        end_exclusive = (scope.end_date + timedelta(days=1)).isoformat()
        clause = (
            f"{alias}.channel IN ({ph}) "
            f"AND {alias}.date >= ? AND {alias}.date < ?"
        )
        params: list = list(scope.channels) + [
            scope.start_date.isoformat(),
            end_exclusive,
        ]
        return clause, params

    @staticmethod
    def _msg_from_row(r: aiosqlite.Row) -> Message:
        return Message(
            id=r["id"],
            channel=r["channel"],
            message_id=r["message_id"],
            date=r["date"],
            text=r["text"],
            url=r["url"],
            media_type=r["media_type"],
            views=r["views"],
            forwards=r["forwards"],
        )

    # ── Runs (audit / progress tracking) ──

    async def create_run(self, run: Run) -> int:
        cur = await self.conn.execute(
            "INSERT INTO runs (config_hash, status, user_id, created_at) VALUES (?, ?, ?, ?)",
            (run.config_hash, run.status, run.user_id, run.created_at.isoformat()),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def update_run_status(self, run_id: int, status: str) -> None:
        await self.conn.execute(
            "UPDATE runs SET status = ? WHERE id = ?", (status, run_id)
        )
        await self.conn.commit()

    async def update_run_counts(
        self, run_id: int, total: int, processed: int
    ) -> None:
        await self.conn.execute(
            "UPDATE runs SET total_messages = ?, processed_messages = ? WHERE id = ?",
            (total, processed, run_id),
        )
        await self.conn.commit()

    async def get_run(self, run_id: int) -> Run | None:
        cur = await self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return Run(
            id=row["id"],
            config_hash=row["config_hash"],
            status=row["status"],
            user_id=row["user_id"],
            total_messages=row["total_messages"],
            processed_messages=row["processed_messages"],
        )

    # ── Messages ──

    async def insert_message(self, msg: Message) -> int:
        date_val = msg.date.isoformat() if hasattr(msg.date, "isoformat") else msg.date
        cur = await self.conn.execute(
            """INSERT OR IGNORE INTO messages
               (channel, message_id, date, text, url, media_type, views, forwards)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg.channel,
                msg.message_id,
                date_val,
                msg.text,
                msg.url,
                msg.media_type,
                msg.views,
                msg.forwards,
            ),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def insert_messages_batch(self, messages: list[Message]) -> None:
        await self.conn.executemany(
            """INSERT OR IGNORE INTO messages
               (channel, message_id, date, text, url, media_type, views, forwards)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    m.channel,
                    m.message_id,
                    m.date.isoformat() if hasattr(m.date, "isoformat") else m.date,
                    m.text,
                    m.url,
                    m.media_type,
                    m.views,
                    m.forwards,
                )
                for m in messages
            ],
        )
        await self.conn.commit()

    async def get_messages(self, scope: QueryScope) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"SELECT * FROM messages m WHERE {where} ORDER BY m.date",
            params,
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    async def count_messages(self, scope: QueryScope) -> int:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM messages m WHERE {where}", params
        )
        row = await cur.fetchone()
        return row["cnt"]  # type: ignore[index]

    async def get_unclassified_messages(self, scope: QueryScope) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.* FROM messages m
               LEFT JOIN classifications c ON c.message_db_id = m.id
               WHERE {where} AND c.id IS NULL
               ORDER BY m.date""",
            params,
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    async def get_keyword_matched_unclassified(
        self, scope: QueryScope
    ) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT DISTINCT m.* FROM messages m
               JOIN keyword_matches km ON km.message_db_id = m.id
               LEFT JOIN classifications c ON c.message_db_id = m.id
               WHERE {where} AND c.id IS NULL
               ORDER BY m.date""",
            params,
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    async def get_messages_without_keyword_matches(
        self, scope: QueryScope
    ) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.* FROM messages m
               LEFT JOIN keyword_matches km ON km.message_db_id = m.id
               WHERE {where} AND km.id IS NULL
               ORDER BY m.date""",
            params,
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    # ── Keywords ──

    async def insert_keyword_matches(self, matches: list[KeywordMatch]) -> None:
        await self.conn.executemany(
            """INSERT INTO keyword_matches
               (message_db_id, keyword, match_count, matched_patterns_json)
               VALUES (?, ?, ?, ?)""",
            [
                (m.message_db_id, m.keyword, m.match_count, m.matched_patterns_json)
                for m in matches
            ],
        )
        await self.conn.commit()

    async def delete_all_keyword_matches(self) -> None:
        await self.conn.execute("DELETE FROM keyword_matches")
        await self.conn.commit()

    async def get_keyword_stats(self, scope: QueryScope) -> list[dict]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT km.keyword,
                      COUNT(DISTINCT km.message_db_id) as message_count,
                      SUM(km.match_count) as total_matches
               FROM keyword_matches km
               JOIN messages m ON m.id = km.message_db_id
               WHERE {where}
               GROUP BY km.keyword
               ORDER BY total_matches DESC""",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_messages_for_keyword(
        self, scope: QueryScope, keyword: str
    ) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.* FROM messages m
               JOIN keyword_matches km ON km.message_db_id = m.id
               WHERE {where} AND km.keyword = ?
               ORDER BY m.date""",
            params + [keyword],
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    # ── Classifications ──

    async def insert_classification(self, cls_obj: Classification) -> int:
        cur = await self.conn.execute(
            """INSERT OR REPLACE INTO classifications
               (message_db_id, mentions_disability, categories_json,
                tones_json,
                problem_scope, disability_model, agency,
                portrayal, target_audiences_json, barriers_json, stigma,
                raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cls_obj.message_db_id,
                int(cls_obj.mentions_disability),
                cls_obj.categories_json,
                cls_obj.tones_json,
                cls_obj.problem_scope,
                cls_obj.disability_model,
                cls_obj.agency,
                cls_obj.portrayal,
                cls_obj.target_audiences_json,
                cls_obj.barriers_json,
                cls_obj.stigma,
                cls_obj.raw_response,
            ),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def insert_classifications_batch(
        self, items: list[Classification]
    ) -> None:
        await self.conn.executemany(
            """INSERT OR REPLACE INTO classifications
               (message_db_id, mentions_disability, categories_json,
                tones_json,
                problem_scope, disability_model, agency,
                portrayal, target_audiences_json, barriers_json, stigma,
                raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.message_db_id,
                    int(c.mentions_disability),
                    c.categories_json,
                    c.tones_json,
                    c.problem_scope,
                    c.disability_model,
                    c.agency,
                    c.portrayal,
                    c.target_audiences_json,
                    c.barriers_json,
                    c.stigma,
                    c.raw_response,
                )
                for c in items
            ],
        )
        await self.conn.commit()

    async def get_classification_stats(self, scope: QueryScope) -> dict:
        where, params = self._scope_where(scope)
        total_cur = await self.conn.execute(
            f"""SELECT COUNT(*) as cnt FROM classifications c
               JOIN messages m ON m.id = c.message_db_id WHERE {where}""",
            params,
        )
        total_row = await total_cur.fetchone()
        total = total_row["cnt"]  # type: ignore[index]

        disability_cur = await self.conn.execute(
            f"""SELECT COUNT(*) as cnt FROM classifications c
               JOIN messages m ON m.id = c.message_db_id
               WHERE {where} AND c.mentions_disability = 1""",
            params,
        )
        disability_row = await disability_cur.fetchone()
        disability_count = disability_row["cnt"]  # type: ignore[index]

        return {
            "total_classified": total,
            "mentions_disability": disability_count,
        }

    async def get_messages_by_category(
        self, scope: QueryScope, category: str
    ) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.* FROM messages m
               JOIN classifications c ON c.message_db_id = m.id
               WHERE {where}
                 AND c.categories_json LIKE ?
               ORDER BY m.date""",
            params + [f'%"{category}"%'],
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    async def get_messages_by_tone(
        self, scope: QueryScope, tone: str
    ) -> list[Message]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.* FROM messages m
               JOIN classifications c ON c.message_db_id = m.id
               WHERE {where}
                 AND c.tones_json LIKE ?
               ORDER BY m.date""",
            params + [f'%"{tone}"%'],
        )
        return [self._msg_from_row(r) for r in await cur.fetchall()]

    async def get_classifications_for_keyword(
        self, scope: QueryScope, keyword: str
    ) -> list[dict]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.url, c.mentions_disability, c.categories_json,
                      c.tones_json, c.problem_scope,
                      c.disability_model, c.agency,
                      c.portrayal, c.target_audiences_json, c.barriers_json, c.stigma
               FROM classifications c
               JOIN messages m ON m.id = c.message_db_id
               JOIN keyword_matches km ON km.message_db_id = m.id
               WHERE {where} AND km.keyword = ?
               ORDER BY m.date""",
            params + [keyword],
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            row_dict["categories"] = json.loads(row_dict.pop("categories_json", "[]"))
            row_dict["tones"] = json.loads(row_dict.pop("tones_json", "[]"))
            row_dict["target_audiences"] = json.loads(
                row_dict.pop("target_audiences_json", "[]")
            )
            row_dict["barriers"] = json.loads(row_dict.pop("barriers_json", "[]"))
            row_dict["mentions_disability"] = bool(row_dict["mentions_disability"])
            result.append(row_dict)
        return result

    async def get_keywords_by_message(
        self, scope: QueryScope
    ) -> dict[int, list[str]]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT km.message_db_id, km.keyword
               FROM keyword_matches km
               JOIN messages m ON m.id = km.message_db_id
               WHERE {where}
               ORDER BY km.message_db_id""",
            params,
        )
        rows = await cur.fetchall()
        mapping: dict[int, list[str]] = {}
        for r in rows:
            mapping.setdefault(r["message_db_id"], []).append(r["keyword"])
        return mapping

    async def get_matched_patterns_by_message(
        self, scope: QueryScope
    ) -> dict[int, list[str]]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT km.message_db_id, km.matched_patterns_json
               FROM keyword_matches km
               JOIN messages m ON m.id = km.message_db_id
               WHERE {where}
               ORDER BY km.message_db_id""",
            params,
        )
        rows = await cur.fetchall()
        mapping: dict[int, list[str]] = {}
        for r in rows:
            patterns = json.loads(r["matched_patterns_json"])
            existing = mapping.setdefault(r["message_db_id"], [])
            for p in patterns:
                if p not in existing:
                    existing.append(p)
        return mapping

    async def get_pattern_stats(
        self, scope: QueryScope
    ) -> dict[str, dict[str, int]]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT km.keyword, km.matched_patterns_json
               FROM keyword_matches km
               JOIN messages m ON m.id = km.message_db_id
               WHERE {where}""",
            params,
        )
        rows = await cur.fetchall()
        result: dict[str, dict[str, int]] = {}
        for r in rows:
            group = r["keyword"]
            patterns = json.loads(r["matched_patterns_json"])
            group_dict = result.setdefault(group, {})
            for p in patterns:
                group_dict[p] = group_dict.get(p, 0) + 1
        return result

    async def get_all_classifications(self, scope: QueryScope) -> list[dict]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.channel, m.message_id, m.date, m.text, m.url,
                      m.views, m.forwards,
                      c.mentions_disability, c.categories_json,
                      c.tones_json,
                      c.problem_scope, c.disability_model, c.agency,
                      c.portrayal, c.target_audiences_json, c.barriers_json,
                      c.stigma
               FROM classifications c
               JOIN messages m ON m.id = c.message_db_id
               WHERE {where}
               ORDER BY m.date""",
            params,
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            row_dict["categories"] = json.loads(row_dict.pop("categories_json"))
            row_dict["tones"] = json.loads(row_dict.pop("tones_json"))
            row_dict["target_audiences"] = json.loads(
                row_dict.pop("target_audiences_json")
            )
            row_dict["barriers"] = json.loads(row_dict.pop("barriers_json"))
            row_dict["mentions_disability"] = bool(row_dict["mentions_disability"])
            result.append(row_dict)
        return result

    # ── Fetch log ──

    async def get_fetched_ranges(self, channel: str) -> list[tuple[str, str]]:
        cur = await self.conn.execute(
            "SELECT start_date, end_date FROM fetch_log "
            "WHERE channel = ? ORDER BY start_date",
            (channel,),
        )
        return [(r["start_date"], r["end_date"]) for r in await cur.fetchall()]

    async def record_fetch(
        self, channel: str, start_date: str, end_date: str
    ) -> None:
        await self.conn.execute(
            "INSERT INTO fetch_log (channel, start_date, end_date, fetched_at) "
            "VALUES (?, ?, ?, ?)",
            (channel, start_date, end_date, datetime.utcnow().isoformat()),
        )
        await self.conn.commit()

    # ── Metadata ──

    async def get_metadata(self, key: str) -> str | None:
        cur = await self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        )
        row = await cur.fetchone()
        return row["value"] if row else None

    async def set_metadata(self, key: str, value: str) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self.conn.commit()

    # ── Advanced queries for stats/charts ──

    async def get_temporal_message_counts(
        self, scope: QueryScope, granularity: str = "day"
    ) -> list[dict]:
        """Message counts grouped by date period.

        granularity: 'day' or 'week'.
        """
        where, params = self._scope_where(scope)
        if granularity == "week":
            date_expr = "strftime('%Y-W%W', m.date)"
        else:
            date_expr = "date(m.date)"
        cur = await self.conn.execute(
            f"""SELECT {date_expr} as period, COUNT(*) as cnt
               FROM messages m
               WHERE {where}
               GROUP BY period ORDER BY period""",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_temporal_keyword_counts(
        self, scope: QueryScope, granularity: str = "day"
    ) -> list[dict]:
        """Keyword-matched message counts grouped by date period and keyword."""
        where, params = self._scope_where(scope)
        if granularity == "week":
            date_expr = "strftime('%Y-W%W', m.date)"
        else:
            date_expr = "date(m.date)"
        cur = await self.conn.execute(
            f"""SELECT {date_expr} as period, km.keyword,
                      COUNT(DISTINCT km.message_db_id) as cnt
               FROM keyword_matches km
               JOIN messages m ON m.id = km.message_db_id
               WHERE {where}
               GROUP BY period, km.keyword ORDER BY period""",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_channel_message_counts(
        self, scope: QueryScope
    ) -> list[dict]:
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.channel, COUNT(*) as total,
                      SUM(CASE WHEN c.id IS NOT NULL THEN 1 ELSE 0 END) as classified,
                      SUM(CASE WHEN c.mentions_disability = 1 THEN 1 ELSE 0 END) as mentions_disability
               FROM messages m
               LEFT JOIN classifications c ON c.message_db_id = m.id
               WHERE {where}
               GROUP BY m.channel ORDER BY total DESC""",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_all_classifications_full(self, scope: QueryScope) -> list[dict]:
        """All classifications with channel, views, forwards for cross-tabs and engagement."""
        where, params = self._scope_where(scope)
        cur = await self.conn.execute(
            f"""SELECT m.channel, m.message_id, m.date, m.text, m.url,
                      m.views, m.forwards,
                      c.mentions_disability, c.categories_json,
                      c.tones_json,
                      c.problem_scope, c.disability_model, c.agency,
                      c.portrayal, c.target_audiences_json, c.barriers_json,
                      c.stigma
               FROM classifications c
               JOIN messages m ON m.id = c.message_db_id
               WHERE {where}
               ORDER BY m.date""",
            params,
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            row_dict["categories"] = json.loads(row_dict.pop("categories_json"))
            row_dict["tones"] = json.loads(row_dict.pop("tones_json"))
            row_dict["target_audiences"] = json.loads(
                row_dict.pop("target_audiences_json")
            )
            row_dict["barriers"] = json.loads(row_dict.pop("barriers_json"))
            row_dict["mentions_disability"] = bool(row_dict["mentions_disability"])
            result.append(row_dict)
        return result
