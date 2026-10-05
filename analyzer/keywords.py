from __future__ import annotations

import hashlib
import json
import logging
import re

from config.models import KeywordGroupConfig
from storage.database import Database
from storage.models import KeywordMatch, QueryScope

logger = logging.getLogger(__name__)


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    r"""Compile a pattern string into a regex.

    If the pattern contains regex metacharacters (\w, \d, etc.),
    it is used as-is. Otherwise it is escaped and wrapped in word boundaries.
    """
    has_regex = bool(re.search(r"\\[wWdDsS+*?]|\[|\]|\(|\)|\{|\}", pattern))
    if has_regex:
        return re.compile(pattern, re.IGNORECASE)
    escaped = re.escape(pattern)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _compile_group(group: KeywordGroupConfig) -> re.Pattern[str]:
    """Compile all patterns of a group into a single alternation regex."""
    parts = []
    for p in group.patterns:
        has_regex = bool(re.search(r"\\[wWdDsS+*?]|\[|\]|\(|\)|\{|\}", p))
        if has_regex:
            parts.append(f"(?:{p})")
        else:
            parts.append(rf"\b{re.escape(p)}\b")
    combined = "|".join(parts)
    return re.compile(combined, re.IGNORECASE)


def compute_keyword_hash(keyword_groups: list[KeywordGroupConfig]) -> str:
    """Deterministic SHA-256 hash of keyword_groups configuration."""
    data = [
        {"label": g.label, "patterns": sorted(g.patterns)}
        for g in sorted(keyword_groups, key=lambda g: g.label)
    ]
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


async def analyze_keywords(
    keyword_groups: list[KeywordGroupConfig],
    scope: QueryScope,
    db: Database,
) -> dict[str, int]:
    """Search messages for keyword group matches and store results.

    If keyword_groups config changed since last run, all existing
    keyword_matches are cleared and every message in the scope is
    re-analyzed. Otherwise only new (unmatched) messages are processed.

    Returns a dict of group_label -> total message count (including cached).
    """
    kw_hash = compute_keyword_hash(keyword_groups)
    stored_hash = await db.get_metadata("keyword_groups_hash")

    if stored_hash != kw_hash:
        logger.info("Keyword config changed — clearing old keyword_matches")
        await db.delete_all_keyword_matches()
        messages = await db.get_messages(scope)
        await db.set_metadata("keyword_groups_hash", kw_hash)
    else:
        messages = await db.get_messages_without_keyword_matches(scope)

    compiled: list[tuple[str, list[tuple[str, re.Pattern[str]]]]] = []
    for g in keyword_groups:
        patterns = [(p, _compile_pattern(p)) for p in g.patterns]
        compiled.append((g.label, patterns))

    all_matches: list[KeywordMatch] = []

    for msg in messages:
        if not msg.text or not msg.id:
            continue
        for label, patterns in compiled:
            matched_pats: list[str] = []
            total = 0
            for pat_str, pat_re in patterns:
                found = pat_re.findall(msg.text)
                if found:
                    matched_pats.append(pat_str)
                    total += len(found)
            if matched_pats:
                all_matches.append(
                    KeywordMatch(
                        message_db_id=msg.id,
                        keyword=label,
                        match_count=total,
                        matched_patterns_json=json.dumps(
                            matched_pats, ensure_ascii=False,
                        ),
                    )
                )

    if all_matches:
        await db.insert_keyword_matches(all_matches)

    logger.info(
        "Keyword analysis complete: %d group matches across %d messages",
        len(all_matches),
        len(messages),
    )

    kw_stats = await db.get_keyword_stats(scope)
    total_counts: dict[str, int] = {g.label: 0 for g in keyword_groups}
    for stat in kw_stats:
        total_counts[stat["keyword"]] = stat["message_count"]
    return total_counts
