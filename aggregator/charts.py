from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

_TEMP_DIR = Path(tempfile.gettempdir()) / "tg_research_charts"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

_PALETTE = sns.color_palette("muted", 12)


def _save(fig: plt.Figure, name: str) -> Path:
    path = _TEMP_DIR / f"{name}.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def chart_dimension_bar(
    counts: dict[str, int],
    title: str,
    *,
    total: int | None = None,
    max_bars: int = 15,
) -> Path:
    """Horizontal bar chart for a single dimension distribution."""
    if not counts:
        return _empty_chart(title)

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_bars]
    labels = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig_height = max(3, len(labels) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(labels, values, color=_PALETTE[0])

    for bar, val in zip(bars, values):
        pct = f" ({val / total * 100:.1f}%)" if total and total > 0 else ""
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}{pct}", va="center", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Количество")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    return _save(fig, _slugify(title))


def chart_pie(
    counts: dict[str, int],
    title: str,
) -> Path:
    """Pie / donut chart for scalar dimensions with few values."""
    if not counts:
        return _empty_chart(title)

    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})",
        startangle=140,
        colors=sns.color_palette("muted", len(labels)),
        pctdistance=0.75,
        wedgeprops={"width": 0.5},
    )
    for t in autotexts:
        t.set_fontsize(9)

    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save(fig, _slugify(title))


def chart_crosstab_heatmap(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Heatmap for a cross-tabulation DataFrame."""
    if df.empty:
        return _empty_chart(title)

    fig_w = max(8, len(df.columns) * 0.9 + 3)
    fig_h = max(5, len(df.index) * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        df, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    return _save(fig, _slugify(title))


def chart_crosstab_grouped_bar(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Grouped bar chart from a cross-tab DataFrame (index=groups, columns=values)."""
    if df.empty:
        return _empty_chart(title)

    fig_w = max(10, len(df.index) * 1.5 + 2)
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    df.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(df.columns)))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Количество")
    ax.set_xlabel("")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    return _save(fig, _slugify(title))


def chart_timeline(
    periods: list[str],
    counts: list[int],
    title: str,
    *,
    secondary_series: dict[str, list[int]] | None = None,
) -> Path:
    """Line chart for a time series. Optional secondary_series for overlaid lines."""
    if not periods:
        return _empty_chart(title)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(periods, counts, marker="o", linewidth=2, markersize=4,
            label="Всего сообщений", color=_PALETTE[0])

    if secondary_series:
        for i, (label, vals) in enumerate(secondary_series.items(), 1):
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(periods, vals, marker="s", linewidth=1.5, markersize=3,
                    label=label, color=color, alpha=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Количество")
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    n_ticks = len(periods)
    if n_ticks > 20:
        step = max(1, n_ticks // 15)
        ax.set_xticks(range(0, n_ticks, step))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    if secondary_series:
        ax.legend(fontsize=8, loc="upper left")

    return _save(fig, _slugify(title))


def chart_channel_comparison(
    channel_data: list[dict],
    title: str,
) -> Path:
    """Grouped bar chart comparing channels by total / classified / mentions_disability."""
    if not channel_data:
        return _empty_chart(title)

    channels = [d["channel"] for d in channel_data]
    totals = [d["total"] for d in channel_data]
    classified = [d["classified"] for d in channel_data]
    mentions = [d["mentions_disability"] for d in channel_data]

    x = np.arange(len(channels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 1.5), 6))
    ax.bar(x - width, totals, width, label="Всего", color=_PALETTE[0])
    ax.bar(x, classified, width, label="Классифицировано", color=_PALETTE[1])
    ax.bar(x + width, mentions, width, label="Упоминание инвалидности", color=_PALETTE[2])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Количество")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    return _save(fig, _slugify(title))


def chart_engagement_bar(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Horizontal bar chart of average views by dimension value."""
    if df.empty:
        return _empty_chart(title)

    df_sorted = df.sort_values("views_mean", ascending=True).tail(15)
    labels = df_sorted["value"].tolist()
    views = df_sorted["views_mean"].tolist()
    counts = df_sorted["count"].tolist()

    fig_height = max(3, len(labels) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(labels, views, color=_PALETTE[3])
    for bar, v, c in zip(bars, views, counts):
        ax.text(bar.get_width() + max(views) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.0f} (n={c})", va="center", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Среднее число просмотров")

    return _save(fig, _slugify(title))


def chart_crosstab_normalized_heatmap(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Heatmap for a cross-tabulation DataFrame, normalized by row (percentages)."""
    if df.empty:
        return _empty_chart(title)

    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig_w = max(8, len(df.columns) * 0.9 + 3)
    fig_h = max(5, len(df.index) * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        df_pct, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8, "label": "%"},
        vmin=0, vmax=100,
    )
    ax.set_title(title + " (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    return _save(fig, _slugify(title + "_norm"))


def chart_stacked_bar(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Horizontal stacked 100% bar chart from a cross-tab DataFrame."""
    if df.empty:
        return _empty_chart(title)

    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig_h = max(4, len(df.index) * 0.55 + 2)
    fig_w = max(10, 12)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = sns.color_palette("muted", len(df.columns))
    df_pct.plot(kind="barh", stacked=True, ax=ax, color=colors, width=0.7)

    for container in ax.containers:
        for rect in container:
            w = rect.get_width()
            if w >= 10:
                cx = rect.get_x() + w / 2
                cy = rect.get_y() + rect.get_height() / 2
                ax.text(cx, cy, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("%")
    ax.set_xlim(0, 100)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.yticks(fontsize=9)

    return _save(fig, _slugify(title + "_stacked"))


def chart_cramers_v_heatmap(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Heatmap for a Cramer's V association matrix (values 0..1)."""
    if df.empty:
        return _empty_chart(title)

    size = max(7, len(df) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(size, size))

    mask = np.zeros_like(df.values, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="Greens", linewidths=0.5,
        mask=mask, ax=ax, cbar_kws={"shrink": 0.8, "label": "Cramer's V"},
        vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    return _save(fig, _slugify(title))


def chart_cooccurrence_heatmap(
    df: pd.DataFrame,
    title: str,
) -> Path:
    """Heatmap for a co-occurrence matrix (diagonal = self-count)."""
    if df.empty:
        return _empty_chart(title)

    size = max(6, len(df) * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(size, size))

    mask = np.zeros_like(df.values, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        df, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
        mask=mask, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    return _save(fig, _slugify(title))


def _empty_chart(title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
            fontsize=14, color="gray", transform=ax.transAxes)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    return _save(fig, _slugify(title))


def _slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s]+", "_", slug).strip("_")[:60]


def cleanup_chart(path: Path) -> None:
    """Remove a temporary chart file after sending."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
