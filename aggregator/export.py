from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

from aggregator.stats import AggregatedStats
from config.models import ResearchConfig
from storage.database import Database
from storage.models import QueryScope


async def export_messages_xlsx(
    scope: QueryScope,
    db: Database,
    config: ResearchConfig,
    output_path: str | Path,
) -> Path:
    """Export detailed message data with classifications to Excel (.xlsx)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_cls = await db.get_all_classifications(scope)
    kw_by_msg = await db.get_keywords_by_message(scope)
    patterns_by_msg = await db.get_matched_patterns_by_message(scope)

    messages = await db.get_messages(scope)
    msg_id_map = {(m.channel, m.message_id): m.id for m in messages}

    rows = []
    for row in all_cls:
        categories = row.get("categories", [])
        tones = row.get("tones", [])
        db_id = msg_id_map.get((row["channel"], row["message_id"]))
        matched_kws = kw_by_msg.get(db_id, []) if db_id else []
        matched_pats = patterns_by_msg.get(db_id, []) if db_id else []
        rows.append({
            "channel": row["channel"],
            "message_id": row["message_id"],
            "date": row["date"],
            "url": row["url"],
            "text": (row.get("text") or "")[:500],
            "views": row.get("views", 0),
            "forwards": row.get("forwards", 0),
            "matched_keywords": ", ".join(matched_kws),
            "matched_patterns": ", ".join(matched_pats),
            "mentions_disability": row.get("mentions_disability", False),
            "categories": ", ".join(categories),
            "tones": ", ".join(tones),
            "problem_scope": row.get("problem_scope", ""),
            "disability_model": row.get("disability_model", ""),
            "agency": row.get("agency", ""),
            "portrayal": row.get("portrayal", ""),
            "target_audiences": ", ".join(row.get("target_audiences", [])),
            "barriers": ", ".join(row.get("barriers", [])),
            "stigma": row.get("stigma", ""),
        })

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Сообщения")
        _autofit_columns(writer, df, "Сообщения")
    return output_path


async def export_summary_xlsx(
    stats: AggregatedStats,
    config: ResearchConfig,
    output_path: str | Path,
) -> Path:
    """Export summary statistics to Excel (.xlsx)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_label = {c.id: c.label for c in config.classification.categories}
    tone_label = {t.id: t.label for t in config.classification.tones}
    problem_scope_label = {s.id: s.label for s in config.classification.problem_scopes}
    disability_model_label = {d.id: d.label for d in config.classification.disability_models}
    agency_label = {a.id: a.label for a in config.classification.agency_types}
    portrayal_label = {p.id: p.label for p in config.classification.portrayal_types}
    target_audience_label = {t.id: t.label for t in config.classification.target_audiences}
    barrier_label = {b.id: b.label for b in config.classification.barriers}
    stigma_label = {s.id: s.label for s in config.classification.stigma_types}

    rows = []

    rows.append({"metric_type": "general", "metric": "total_messages", "value": stats.total_messages})
    rows.append({"metric_type": "general", "metric": "channels_count", "value": stats.channels_count})
    rows.append({"metric_type": "general", "metric": "total_classified", "value": stats.total_classified})
    rows.append({"metric_type": "general", "metric": "mentions_disability", "value": stats.mentions_disability_count})

    for kw_stat in stats.keyword_stats:
        rows.append({
            "metric_type": "keyword",
            "metric": kw_stat["keyword"],
            "value": kw_stat["message_count"],
            "detail": f"total_matches={kw_stat['total_matches']}",
        })
        pattern_breakdown = stats.pattern_stats.get(kw_stat["keyword"], {})
        for pattern, pat_msg_count in pattern_breakdown.items():
            rows.append({
                "metric_type": "keyword_pattern",
                "metric": f"{kw_stat['keyword']}|{pattern}",
                "value": pat_msg_count,
            })

    for cat_id, count in stats.category_counts.items():
        rows.append({
            "metric_type": "category",
            "metric": category_label.get(cat_id, cat_id),
            "value": count,
        })

    for tone_id, count in stats.tone_counts.items():
        rows.append({
            "metric_type": "tone",
            "metric": tone_label.get(tone_id, tone_id),
            "value": count,
        })

    for ps_id, count in stats.problem_scope_counts.items():
        rows.append({
            "metric_type": "problem_scope",
            "metric": problem_scope_label.get(ps_id, ps_id),
            "value": count,
        })

    for dm_id, count in stats.disability_model_counts.items():
        rows.append({
            "metric_type": "disability_model",
            "metric": disability_model_label.get(dm_id, dm_id),
            "value": count,
        })

    for ag_id, count in stats.agency_counts.items():
        rows.append({
            "metric_type": "agency",
            "metric": agency_label.get(ag_id, ag_id),
            "value": count,
        })

    for p_id, count in stats.portrayal_counts.items():
        rows.append({
            "metric_type": "portrayal",
            "metric": portrayal_label.get(p_id, p_id),
            "value": count,
        })

    for ta_id, count in stats.target_audience_counts.items():
        rows.append({
            "metric_type": "target_audience",
            "metric": target_audience_label.get(ta_id, ta_id),
            "value": count,
        })

    for b_id, count in stats.barrier_counts.items():
        rows.append({
            "metric_type": "barrier",
            "metric": barrier_label.get(b_id, b_id),
            "value": count,
        })

    for s_id, count in stats.stigma_counts.items():
        rows.append({
            "metric_type": "stigma",
            "metric": stigma_label.get(s_id, s_id),
            "value": count,
        })

    for kw, kcs in stats.keyword_cls_stats.items():
        if kcs.total_classified == 0:
            continue
        prefix = f"kw:{kw}"
        rows.append({
            "metric_type": "keyword_general",
            "metric": f"{prefix}|classified",
            "value": kcs.total_classified,
            "detail": f"keyword_messages={kcs.keyword_message_count}",
        })
        rows.append({
            "metric_type": "keyword_general",
            "metric": f"{prefix}|mentions_disability",
            "value": kcs.mentions_disability_count,
        })
        for cat_id, count in kcs.category_counts.items():
            rows.append({
                "metric_type": "keyword_category",
                "metric": f"{prefix}|{category_label.get(cat_id, cat_id)}",
                "value": count,
            })
        for tone_id, count in kcs.tone_counts.items():
            rows.append({
                "metric_type": "keyword_tone",
                "metric": f"{prefix}|{tone_label.get(tone_id, tone_id)}",
                "value": count,
            })
        for ps_id, count in kcs.problem_scope_counts.items():
            rows.append({
                "metric_type": "keyword_problem_scope",
                "metric": f"{prefix}|{problem_scope_label.get(ps_id, ps_id)}",
                "value": count,
            })
        for dm_id, count in kcs.disability_model_counts.items():
            rows.append({
                "metric_type": "keyword_disability_model",
                "metric": f"{prefix}|{disability_model_label.get(dm_id, dm_id)}",
                "value": count,
            })
        for ag_id, count in kcs.agency_counts.items():
            rows.append({
                "metric_type": "keyword_agency",
                "metric": f"{prefix}|{agency_label.get(ag_id, ag_id)}",
                "value": count,
            })
        for p_id, count in kcs.portrayal_counts.items():
            rows.append({
                "metric_type": "keyword_portrayal",
                "metric": f"{prefix}|{portrayal_label.get(p_id, p_id)}",
                "value": count,
            })
        for ta_id, count in kcs.target_audience_counts.items():
            rows.append({
                "metric_type": "keyword_target_audience",
                "metric": f"{prefix}|{target_audience_label.get(ta_id, ta_id)}",
                "value": count,
            })
        for b_id, count in kcs.barrier_counts.items():
            rows.append({
                "metric_type": "keyword_barrier",
                "metric": f"{prefix}|{barrier_label.get(b_id, b_id)}",
                "value": count,
            })
        for s_id, count in kcs.stigma_counts.items():
            rows.append({
                "metric_type": "keyword_stigma",
                "metric": f"{prefix}|{stigma_label.get(s_id, s_id)}",
                "value": count,
            })

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Статистика")
        _autofit_columns(writer, df, "Статистика")
    return output_path


def _autofit_columns(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
) -> None:
    """Adjust column widths so data is readable without manual resizing."""
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        max_len = max(
            df[col].astype(str).str.len().max(),
            len(str(col)),
        )
        col_letter = get_column_letter(i + 1)
        ws.column_dimensions[col_letter].width = min(max_len + 2, 60)
