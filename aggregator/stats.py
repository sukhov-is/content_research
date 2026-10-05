from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from config.models import ResearchConfig
from storage.database import Database
from storage.models import QueryScope


@dataclass
class KeywordClassificationStats:
    """Classification breakdown for messages matching a single keyword."""
    keyword_message_count: int = 0
    total_classified: int = 0
    mentions_disability_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    tone_counts: dict[str, int] = field(default_factory=dict)
    problem_scope_counts: dict[str, int] = field(default_factory=dict)
    disability_model_counts: dict[str, int] = field(default_factory=dict)
    agency_counts: dict[str, int] = field(default_factory=dict)
    portrayal_counts: dict[str, int] = field(default_factory=dict)
    target_audience_counts: dict[str, int] = field(default_factory=dict)
    barrier_counts: dict[str, int] = field(default_factory=dict)
    stigma_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AggregatedStats:
    total_messages: int = 0
    channels_count: int = 0
    period_start: str = ""
    period_end: str = ""

    # Keyword stats: keyword -> {message_count, total_matches}
    keyword_stats: list[dict] = field(default_factory=list)

    # LLM classification stats
    total_classified: int = 0
    mentions_disability_count: int = 0

    # category_id -> count
    category_counts: dict[str, int] = field(default_factory=dict)
    # tone_id -> count
    tone_counts: dict[str, int] = field(default_factory=dict)
    # problem_scope -> count
    problem_scope_counts: dict[str, int] = field(default_factory=dict)
    # disability_model -> count
    disability_model_counts: dict[str, int] = field(default_factory=dict)
    # agency -> count
    agency_counts: dict[str, int] = field(default_factory=dict)
    # portrayal -> count
    portrayal_counts: dict[str, int] = field(default_factory=dict)
    # target_audience -> count
    target_audience_counts: dict[str, int] = field(default_factory=dict)
    # barrier -> count
    barrier_counts: dict[str, int] = field(default_factory=dict)
    # stigma -> count
    stigma_counts: dict[str, int] = field(default_factory=dict)

    # category_id -> list of message URLs
    category_urls: dict[str, list[str]] = field(default_factory=dict)
    # tone_id -> list of message URLs
    tone_urls: dict[str, list[str]] = field(default_factory=dict)
    # keyword -> list of message URLs
    keyword_urls: dict[str, list[str]] = field(default_factory=dict)
    # URLs of messages that mention disability
    disability_urls: list[str] = field(default_factory=list)
    # problem_scope -> list of message URLs
    problem_scope_urls: dict[str, list[str]] = field(default_factory=dict)
    # disability_model -> list of message URLs
    disability_model_urls: dict[str, list[str]] = field(default_factory=dict)
    # agency -> list of message URLs
    agency_urls: dict[str, list[str]] = field(default_factory=dict)
    # portrayal -> list of message URLs
    portrayal_urls: dict[str, list[str]] = field(default_factory=dict)
    # target_audience -> list of message URLs
    target_audience_urls: dict[str, list[str]] = field(default_factory=dict)
    # barrier -> list of message URLs
    barrier_urls: dict[str, list[str]] = field(default_factory=dict)
    # stigma -> list of message URLs
    stigma_urls: dict[str, list[str]] = field(default_factory=dict)

    # keyword -> classification breakdown
    keyword_cls_stats: dict[str, KeywordClassificationStats] = field(default_factory=dict)

    # group_label -> {pattern: message_count}
    pattern_stats: dict[str, dict[str, int]] = field(default_factory=dict)


def _compute_cls_counters(rows: list[dict]) -> KeywordClassificationStats:
    """Build KeywordClassificationStats from a list of classification dicts."""
    kcs = KeywordClassificationStats(total_classified=len(rows))
    cat_c: Counter[str] = Counter()
    tone_c: Counter[str] = Counter()
    ps_c: Counter[str] = Counter()
    dm_c: Counter[str] = Counter()
    ag_c: Counter[str] = Counter()
    port_c: Counter[str] = Counter()
    ta_c: Counter[str] = Counter()
    bar_c: Counter[str] = Counter()
    stig_c: Counter[str] = Counter()

    for row in rows:
        for cat in row.get("categories", []):
            cat_c[cat] += 1
        for tone in row.get("tones", []):
            tone_c[tone] += 1
        if row.get("mentions_disability"):
            kcs.mentions_disability_count += 1
        ps = row.get("problem_scope", "")
        if ps:
            ps_c[ps] += 1
        dm = row.get("disability_model", "")
        if dm:
            dm_c[dm] += 1
        ag = row.get("agency", "")
        if ag:
            ag_c[ag] += 1
        port = row.get("portrayal", "")
        if port:
            port_c[port] += 1
        for ta in row.get("target_audiences", []):
            ta_c[ta] += 1
        for bar in row.get("barriers", []):
            bar_c[bar] += 1
        stig = row.get("stigma", "")
        if stig:
            stig_c[stig] += 1

    kcs.category_counts = dict(cat_c.most_common())
    kcs.tone_counts = dict(tone_c.most_common())
    kcs.problem_scope_counts = dict(ps_c.most_common())
    kcs.disability_model_counts = dict(dm_c.most_common())
    kcs.agency_counts = dict(ag_c.most_common())
    kcs.portrayal_counts = dict(port_c.most_common())
    kcs.target_audience_counts = dict(ta_c.most_common())
    kcs.barrier_counts = dict(bar_c.most_common())
    kcs.stigma_counts = dict(stig_c.most_common())
    return kcs


async def compute_stats(
    config: ResearchConfig,
    scope: QueryScope,
    db: Database,
) -> AggregatedStats:
    stats = AggregatedStats(
        period_start=str(config.period.start),
        period_end=str(config.period.end),
        channels_count=len(config.channels),
    )

    stats.total_messages = await db.count_messages(scope)

    # Keyword statistics (group-level and per-pattern)
    stats.keyword_stats = await db.get_keyword_stats(scope)
    stats.pattern_stats = await db.get_pattern_stats(scope)
    for g in config.keyword_groups:
        msgs = await db.get_messages_for_keyword(scope, g.label)
        stats.keyword_urls[g.label] = [m.url for m in msgs]

    # Classification statistics
    cls_stats = await db.get_classification_stats(scope)
    stats.total_classified = cls_stats["total_classified"]
    stats.mentions_disability_count = cls_stats["mentions_disability"]

    all_cls = await db.get_all_classifications(scope)

    category_counter: Counter[str] = Counter()
    tone_counter: Counter[str] = Counter()
    problem_scope_counter: Counter[str] = Counter()
    disability_model_counter: Counter[str] = Counter()
    agency_counter: Counter[str] = Counter()
    portrayal_counter: Counter[str] = Counter()
    target_audience_counter: Counter[str] = Counter()
    barrier_counter: Counter[str] = Counter()
    stigma_counter: Counter[str] = Counter()

    for row in all_cls:
        cats = row.get("categories", [])
        for cat in cats:
            category_counter[cat] += 1
            stats.category_urls.setdefault(cat, []).append(row["url"])

        tones = row.get("tones", [])
        for tone in tones:
            tone_counter[tone] += 1
            stats.tone_urls.setdefault(tone, []).append(row["url"])

        if row.get("mentions_disability"):
            stats.disability_urls.append(row["url"])

        ps = row.get("problem_scope", "")
        if ps:
            problem_scope_counter[ps] += 1
            stats.problem_scope_urls.setdefault(ps, []).append(row["url"])

        dm = row.get("disability_model", "")
        if dm:
            disability_model_counter[dm] += 1
            stats.disability_model_urls.setdefault(dm, []).append(row["url"])

        ag = row.get("agency", "")
        if ag:
            agency_counter[ag] += 1
            stats.agency_urls.setdefault(ag, []).append(row["url"])

        port = row.get("portrayal", "")
        if port:
            portrayal_counter[port] += 1
            stats.portrayal_urls.setdefault(port, []).append(row["url"])

        for ta in row.get("target_audiences", []):
            target_audience_counter[ta] += 1
            stats.target_audience_urls.setdefault(ta, []).append(row["url"])

        for bar in row.get("barriers", []):
            barrier_counter[bar] += 1
            stats.barrier_urls.setdefault(bar, []).append(row["url"])

        stig = row.get("stigma", "")
        if stig:
            stigma_counter[stig] += 1
            stats.stigma_urls.setdefault(stig, []).append(row["url"])

    stats.category_counts = dict(category_counter.most_common())
    stats.tone_counts = dict(tone_counter.most_common())
    stats.problem_scope_counts = dict(problem_scope_counter.most_common())
    stats.disability_model_counts = dict(disability_model_counter.most_common())
    stats.agency_counts = dict(agency_counter.most_common())
    stats.portrayal_counts = dict(portrayal_counter.most_common())
    stats.target_audience_counts = dict(target_audience_counter.most_common())
    stats.barrier_counts = dict(barrier_counter.most_common())
    stats.stigma_counts = dict(stigma_counter.most_common())

    # Per-keyword classification breakdown
    for g in config.keyword_groups:
        kw = g.label
        kw_cls_rows = await db.get_classifications_for_keyword(scope, kw)
        kcs = _compute_cls_counters(kw_cls_rows)
        kw_msg_count = next(
            (s["message_count"] for s in stats.keyword_stats if s["keyword"] == kw),
            0,
        )
        kcs.keyword_message_count = kw_msg_count
        stats.keyword_cls_stats[kw] = kcs

    return stats
