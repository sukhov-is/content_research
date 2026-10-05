import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aggregator.report import format_telegram_report
from aggregator.stats import AggregatedStats, KeywordClassificationStats
from config.models import (
    CategoryConfig,
    ClassificationConfig,
    KeywordGroupConfig,
    LLMConfig,
    PeriodConfig,
    ResearchConfig,
    ToneConfig,
)


def _make_config() -> ResearchConfig:
    return ResearchConfig(
        channels=["@ch1", "@ch2"],
        period=PeriodConfig(start="2026-01-01", end="2026-01-31"),
        keyword_groups=[KeywordGroupConfig(label="Инвалидность", patterns=["инвалидность"])],
        classification=ClassificationConfig(
            categories=[CategoryConfig(id="medical", label="Мед. контекст", description="")],
            tones=[ToneConfig(id="neutral", label="Нейтральный")],
        ),
    )


def test_report_contains_general_stats():
    config = _make_config()
    stats = AggregatedStats(
        total_messages=100,
        channels_count=2,
        period_start="2026-01-01",
        period_end="2026-01-31",
        total_classified=90,
        mentions_disability_count=30,
        keyword_stats=[{"keyword": "инвалидность", "message_count": 25, "total_matches": 40}],
        category_counts={"medical": 30},
        tone_counts={"neutral": 60},
    )

    messages = format_telegram_report(stats, config)
    full_text = "\n".join(messages)
    assert "100" in full_text
    assert "инвалидность" in full_text
    assert "Мед. контекст" in full_text
    assert "Нейтральный" in full_text


def test_report_contains_keyword_classification():
    config = _make_config()
    kcs = KeywordClassificationStats(
        keyword_message_count=25,
        total_classified=20,
        mentions_disability_count=15,
        category_counts={"medical": 18},
        tone_counts={"neutral": 12},
    )
    stats = AggregatedStats(
        total_messages=100,
        channels_count=2,
        period_start="2026-01-01",
        period_end="2026-01-31",
        total_classified=90,
        mentions_disability_count=30,
        keyword_stats=[{"keyword": "инвалидность", "message_count": 25, "total_matches": 40}],
        category_counts={"medical": 30},
        tone_counts={"neutral": 60},
        keyword_cls_stats={"инвалидность": kcs},
    )

    messages = format_telegram_report(stats, config)
    full_text = "\n".join(messages)
    assert 'Анализ по ключевому слову: "инвалидность"' in full_text
    assert "Сообщений с ключевым словом: 25" in full_text
    assert "Из них классифицировано: 20" in full_text
    assert "Мед. контекст: 18" in full_text


def test_report_contains_pattern_breakdown():
    config = _make_config()
    stats = AggregatedStats(
        total_messages=100,
        channels_count=2,
        period_start="2026-01-01",
        period_end="2026-01-31",
        total_classified=90,
        mentions_disability_count=30,
        keyword_stats=[{"keyword": "Инвалидность", "message_count": 50, "total_matches": 120}],
        category_counts={"medical": 30},
        tone_counts={"neutral": 60},
        pattern_stats={
            "Инвалидность": {
                r"инвалид\w*": 45,
                "ОВЗ": 8,
            },
        },
    )

    messages = format_telegram_report(stats, config)
    full_text = "\n".join(messages)
    assert r"инвалид\w*: 45 сообщ." in full_text
    assert "ОВЗ: 8 сообщ." in full_text


def test_report_skips_empty_keyword_cls():
    config = _make_config()
    kcs = KeywordClassificationStats(keyword_message_count=0)
    stats = AggregatedStats(
        total_messages=100,
        channels_count=2,
        period_start="2026-01-01",
        period_end="2026-01-31",
        keyword_cls_stats={"инвалидность": kcs},
    )

    messages = format_telegram_report(stats, config)
    full_text = "\n".join(messages)
    assert "Анализ по ключевому слову" not in full_text


def test_report_respects_message_limit():
    config = _make_config()
    stats = AggregatedStats(
        total_messages=100,
        channels_count=2,
        period_start="2026-01-01",
        period_end="2026-01-31",
        total_classified=90,
        mentions_disability_count=30,
        category_urls={"medical": [f"https://t.me/ch/{i}" for i in range(50)]},
    )

    messages = format_telegram_report(stats, config)
    for msg in messages:
        assert len(msg) <= 4096
