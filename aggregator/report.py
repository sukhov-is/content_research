from __future__ import annotations

from config.models import ResearchConfig
from aggregator.stats import AggregatedStats, KeywordClassificationStats


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{part / total * 100:.1f}%"


def format_telegram_report(stats: AggregatedStats, config: ResearchConfig) -> list[str]:
    """Format the report as a list of Telegram messages (split to fit 4096 char limit).

    Only summary statistics are included; full message lists are in CSV.
    """
    category_label = {c.id: c.label for c in config.classification.categories}
    tone_label = {t.id: t.label for t in config.classification.tones}
    problem_scope_label = {s.id: s.label for s in config.classification.problem_scopes}
    disability_model_label = {d.id: d.label for d in config.classification.disability_models}
    agency_label = {a.id: a.label for a in config.classification.agency_types}
    portrayal_label = {p.id: p.label for p in config.classification.portrayal_types}
    target_audience_label = {t.id: t.label for t in config.classification.target_audiences}
    barrier_label = {b.id: b.label for b in config.classification.barriers}
    stigma_label = {s.id: s.label for s in config.classification.stigma_types}

    sections: list[str] = []

    # Section 1: General stats
    header = (
        f"== Общая статистика ==\n\n"
        f"Каналов: {stats.channels_count}\n"
        f"Период: {stats.period_start} -- {stats.period_end}\n"
        f"Всего сообщений: {stats.total_messages}\n"
        f"Классифицировано LLM: {stats.total_classified}\n"
        f"Упоминание инвалидности (LLM): "
        f"{stats.mentions_disability_count} "
        f"({_pct(stats.mentions_disability_count, stats.total_classified)})"
    )
    sections.append(header)

    # Section 2: Keyword stats (with per-pattern breakdown)
    kw_lines = ["== Статистика по ключевым словам ==\n"]
    for kw_stat in stats.keyword_stats:
        kw = kw_stat["keyword"]
        msg_count = kw_stat["message_count"]
        total_matches = kw_stat["total_matches"]
        kw_lines.append(f'"{kw}": {msg_count} сообщ., {total_matches} совпад.')
        pattern_breakdown = stats.pattern_stats.get(kw, {})
        for pattern, pat_count in pattern_breakdown.items():
            kw_lines.append(f"  - {pattern}: {pat_count} сообщ.")
    if not stats.keyword_stats:
        kw_lines.append("Совпадений не найдено.")
    sections.append("\n".join(kw_lines))

    # Per-keyword classification breakdown
    for kw, kcs in stats.keyword_cls_stats.items():
        if kcs.keyword_message_count == 0:
            continue
        lines: list[str] = [f'== Анализ по ключевому слову: "{kw}" ==\n']
        lines.append(
            f"Сообщений с ключевым словом: {kcs.keyword_message_count}\n"
            f"Из них классифицировано: {kcs.total_classified}\n"
            f"Упоминание инвалидности (LLM): "
            f"{kcs.mentions_disability_count} "
            f"({_pct(kcs.mentions_disability_count, kcs.total_classified)})"
        )
        if kcs.category_counts:
            lines.append("\nКатегории:")
            for cat_id, count in kcs.category_counts.items():
                label = category_label.get(cat_id, cat_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.tone_counts:
            lines.append("\nТон:")
            for tone_id, count in kcs.tone_counts.items():
                label = tone_label.get(tone_id, tone_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.problem_scope_counts:
            lines.append("\nМасштаб проблемы:")
            for ps_id, count in kcs.problem_scope_counts.items():
                label = problem_scope_label.get(ps_id, ps_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.disability_model_counts:
            lines.append("\nМодель инвалидности:")
            for dm_id, count in kcs.disability_model_counts.items():
                label = disability_model_label.get(dm_id, dm_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.agency_counts:
            lines.append("\nАгентность:")
            for ag_id, count in kcs.agency_counts.items():
                label = agency_label.get(ag_id, ag_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.portrayal_counts:
            lines.append("\nРепрезентация:")
            for p_id, count in kcs.portrayal_counts.items():
                label = portrayal_label.get(p_id, p_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.target_audience_counts:
            lines.append("\nАдресат сообщения:")
            for ta_id, count in kcs.target_audience_counts.items():
                label = target_audience_label.get(ta_id, ta_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.barrier_counts:
            lines.append("\nБарьеры:")
            for b_id, count in kcs.barrier_counts.items():
                label = barrier_label.get(b_id, b_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        if kcs.stigma_counts:
            lines.append("\nСтигма:")
            for s_id, count in kcs.stigma_counts.items():
                label = stigma_label.get(s_id, s_id)
                lines.append(f"  {label}: {count} ({_pct(count, kcs.total_classified)})")
        sections.append("\n".join(lines))

    # Section 3: Category stats
    cat_lines = ["== Категории (LLM) ==\n"]
    for cat_id, count in stats.category_counts.items():
        label = category_label.get(cat_id, cat_id)
        cat_lines.append(
            f"{label}: {count} ({_pct(count, stats.total_classified)})"
        )
    if not stats.category_counts:
        cat_lines.append("Категории не определены.")
    sections.append("\n".join(cat_lines))

    # Section 4: Tone stats
    tone_lines = ["== Тон сообщений (LLM) ==\n"]
    for tone_id, count in stats.tone_counts.items():
        label = tone_label.get(tone_id, tone_id)
        tone_lines.append(
            f"{label}: {count} ({_pct(count, stats.total_classified)})"
        )
    if not stats.tone_counts:
        tone_lines.append("Тона не определены.")
    sections.append("\n".join(tone_lines))

    # Section 5: Problem scope stats
    if stats.problem_scope_counts:
        ps_lines = ["== Масштаб проблемы (LLM) ==\n"]
        for ps_id, count in stats.problem_scope_counts.items():
            label = problem_scope_label.get(ps_id, ps_id)
            ps_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(ps_lines))

    # Section 7: Disability model stats
    if stats.disability_model_counts:
        dm_lines = ["== Модель инвалидности (LLM) ==\n"]
        for dm_id, count in stats.disability_model_counts.items():
            label = disability_model_label.get(dm_id, dm_id)
            dm_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(dm_lines))

    # Section 8: Agency stats
    if stats.agency_counts:
        ag_lines = ["== Агентность (LLM) ==\n"]
        for ag_id, count in stats.agency_counts.items():
            label = agency_label.get(ag_id, ag_id)
            ag_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(ag_lines))

    # Section 9: Portrayal stats
    if stats.portrayal_counts:
        p_lines = ["== Репрезентация (LLM) ==\n"]
        for p_id, count in stats.portrayal_counts.items():
            label = portrayal_label.get(p_id, p_id)
            p_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(p_lines))

    # Section 10: Target audience stats
    if stats.target_audience_counts:
        ta_lines = ["== Адресат сообщения (LLM) ==\n"]
        for ta_id, count in stats.target_audience_counts.items():
            label = target_audience_label.get(ta_id, ta_id)
            ta_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(ta_lines))

    # Section 11: Barrier stats
    if stats.barrier_counts:
        b_lines = ["== Барьеры (LLM) ==\n"]
        for b_id, count in stats.barrier_counts.items():
            label = barrier_label.get(b_id, b_id)
            b_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(b_lines))

    # Section 12: Stigma stats
    if stats.stigma_counts:
        s_lines = ["== Стигма (LLM) ==\n"]
        for s_id, count in stats.stigma_counts.items():
            label = stigma_label.get(s_id, s_id)
            s_lines.append(f"{label}: {count} ({_pct(count, stats.total_classified)})")
        sections.append("\n".join(s_lines))

    sections.append("Подробные списки сообщений по категориям -- в Excel-файлах.")

    return _split_to_telegram_messages(sections)


_TG_LIMIT = 4000  # leave margin below Telegram's 4096


def _split_to_telegram_messages(sections: list[str]) -> list[str]:
    """Split sections into messages respecting Telegram's 4096 char limit.

    Oversized individual sections are further split by line breaks.
    """
    telegram_messages: list[str] = []
    current = ""

    for section in sections:
        if len(section) > _TG_LIMIT:
            if current:
                telegram_messages.append(current)
                current = ""
            telegram_messages.extend(_split_long_section(section))
            continue

        if len(current) + len(section) + 2 > _TG_LIMIT:
            if current:
                telegram_messages.append(current)
            current = section
        else:
            current = f"{current}\n\n{section}" if current else section

    if current:
        telegram_messages.append(current)

    return telegram_messages


def _split_long_section(text: str) -> list[str]:
    """Split a single oversized section by line breaks."""
    lines = text.split("\n")
    parts: list[str] = []
    chunk = ""
    for line in lines:
        candidate = f"{chunk}\n{line}" if chunk else line
        if len(candidate) > _TG_LIMIT:
            if chunk:
                parts.append(chunk)
            chunk = line[:_TG_LIMIT] if len(line) > _TG_LIMIT else line
        else:
            chunk = candidate
    if chunk:
        parts.append(chunk)
    return parts
