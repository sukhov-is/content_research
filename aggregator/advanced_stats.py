from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config.models import ResearchConfig
from storage.database import Database
from storage.models import QueryScope

DIMENSION_FIELDS: dict[str, dict] = {
    "categories": {"json_list": True},
    "tones": {"json_list": True},
    "problem_scope": {"json_list": False},
    "disability_model": {"json_list": False},
    "agency": {"json_list": False},
    "portrayal": {"json_list": False},
    "target_audiences": {"json_list": True},
    "barriers": {"json_list": True},
    "stigma": {"json_list": False},
}


def _get_values(row: dict, field: str) -> list[str]:
    """Extract classification values from a row, always returning a list."""
    meta = DIMENSION_FIELDS.get(field, {})
    val = row.get(field, "" if not meta.get("json_list") else [])
    if isinstance(val, list):
        return [v for v in val if v]
    if val:
        return [val]
    return []


def compute_crosstab(
    all_cls: list[dict],
    dim_row: str,
    dim_col: str,
) -> pd.DataFrame:
    """Build a cross-tabulation DataFrame: rows=dim_row values, cols=dim_col values."""
    pairs: list[tuple[str, str]] = []
    for row in all_cls:
        row_vals = _get_values(row, dim_row)
        col_vals = _get_values(row, dim_col)
        for rv in row_vals:
            for cv in col_vals:
                pairs.append((rv, cv))

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs, columns=[dim_row, dim_col])
    return pd.crosstab(df[dim_row], df[dim_col])


def compute_keyword_crosstab(
    all_cls: list[dict],
    keyword_map: dict[int, list[str]],
    msg_id_map: dict[tuple[str, int], int],
    dimension: str,
) -> pd.DataFrame:
    """Cross-tab of keyword groups vs a classification dimension."""
    pairs: list[tuple[str, str]] = []
    for row in all_cls:
        db_id = msg_id_map.get((row["channel"], row["message_id"]))
        if db_id is None:
            continue
        keywords = keyword_map.get(db_id, [])
        dim_vals = _get_values(row, dimension)
        for kw in keywords:
            for dv in dim_vals:
                pairs.append((kw, dv))

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs, columns=["keyword", dimension])
    return pd.crosstab(df["keyword"], df[dimension])


def compute_channel_dimension(
    all_cls: list[dict],
    dimension: str,
) -> pd.DataFrame:
    """Cross-tab of channel vs a classification dimension."""
    pairs: list[tuple[str, str]] = []
    for row in all_cls:
        ch = row.get("channel", "")
        for val in _get_values(row, dimension):
            pairs.append((ch, val))

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs, columns=["channel", dimension])
    return pd.crosstab(df["channel"], df[dimension])


def compute_cooccurrence(
    all_cls: list[dict],
    field: str,
) -> pd.DataFrame:
    """Build a co-occurrence matrix for a list-valued dimension (e.g. categories)."""
    counter: Counter[tuple[str, str]] = Counter()
    all_labels: set[str] = set()

    for row in all_cls:
        vals = sorted(set(_get_values(row, field)))
        all_labels.update(vals)
        for a, b in combinations(vals, 2):
            counter[(a, b)] += 1
            counter[(b, a)] += 1
        for v in vals:
            counter[(v, v)] += 1

    if not all_labels:
        return pd.DataFrame()

    labels = sorted(all_labels)
    data = {l: [counter.get((l, c), 0) for c in labels] for l in labels}
    return pd.DataFrame(data, index=labels)


def compute_engagement_stats(
    all_cls: list[dict],
    dimension: str,
) -> pd.DataFrame:
    """Average views and forwards grouped by dimension values."""
    records: list[dict] = []
    for row in all_cls:
        views = row.get("views", 0) or 0
        forwards = row.get("forwards", 0) or 0
        for val in _get_values(row, dimension):
            records.append({"value": val, "views": views, "forwards": forwards})

    if not records:
        return pd.DataFrame(columns=["value", "views_mean", "forwards_mean", "count"])

    df = pd.DataFrame(records)
    agg = df.groupby("value").agg(
        views_mean=("views", "mean"),
        forwards_mean=("forwards", "mean"),
        count=("views", "size"),
    ).reset_index()
    return agg.sort_values("views_mean", ascending=False)


def compute_temporal_dimension(
    all_cls: list[dict],
    dimension: str,
    granularity: str = "week",
) -> pd.DataFrame:
    """Time series of a dimension: rows=periods, columns=dimension values, values=counts."""
    records: list[dict] = []
    for row in all_cls:
        date_str = row.get("date", "")
        if not date_str:
            continue
        try:
            dt = pd.Timestamp(date_str)
        except Exception:
            continue

        if granularity == "week":
            period = dt.strftime("%Y-W%W")
        else:
            period = dt.strftime("%Y-%m-%d")

        for val in _get_values(row, dimension):
            records.append({"period": period, "value": val})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return pd.crosstab(df["period"], df["value"]).sort_index()


def _cramers_v(ct: pd.DataFrame) -> float:
    """Compute Cramer's V for a contingency table DataFrame."""
    if ct.empty or ct.values.sum() == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct.values)
    n = ct.values.sum()
    r, c = ct.shape
    min_dim = min(r, c) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_cramers_v_matrix(
    all_cls: list[dict],
    dimensions: list[str],
) -> pd.DataFrame:
    """Build a symmetric matrix of Cramer's V values for all dimension pairs."""
    n = len(dimensions)
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            ct = compute_crosstab(all_cls, dimensions[i], dimensions[j])
            v = _cramers_v(ct)
            matrix[i][j] = v
            matrix[j][i] = v

    return pd.DataFrame(matrix, index=dimensions, columns=dimensions)


def get_label_maps(config: ResearchConfig) -> dict[str, dict[str, str]]:
    """Build id->label mappings for all classification dimensions from config."""
    cls_cfg = config.classification
    return {
        "categories": {c.id: c.label for c in cls_cfg.categories},
        "tones": {t.id: t.label for t in cls_cfg.tones},
        "problem_scope": {s.id: s.label for s in cls_cfg.problem_scopes},
        "disability_model": {d.id: d.label for d in cls_cfg.disability_models},
        "agency": {a.id: a.label for a in cls_cfg.agency_types},
        "portrayal": {p.id: p.label for p in cls_cfg.portrayal_types},
        "target_audiences": {t.id: t.label for t in cls_cfg.target_audiences},
        "barriers": {b.id: b.label for b in cls_cfg.barriers},
        "stigma": {s.id: s.label for s in cls_cfg.stigma_types},
    }


def translate_ids(
    counts: dict[str, int],
    label_map: dict[str, str],
) -> dict[str, int]:
    """Replace dimension IDs with human-readable labels."""
    return {label_map.get(k, k): v for k, v in counts.items()}


def translate_df_index(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame index using label_map."""
    return df.rename(index=label_map)


def translate_df_columns(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns using label_map."""
    return df.rename(columns=label_map)
