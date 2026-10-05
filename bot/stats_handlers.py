from __future__ import annotations

import asyncio
import logging
import zipfile
from collections import Counter
from pathlib import Path
from tempfile import gettempdir

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from aggregator.advanced_stats import (
    DIMENSION_FIELDS,
    compute_channel_dimension,
    compute_cooccurrence,
    compute_cramers_v_matrix,
    compute_crosstab,
    compute_engagement_stats,
    compute_keyword_crosstab,
    compute_temporal_dimension,
    get_label_maps,
    translate_df_columns,
    translate_df_index,
    translate_ids,
)
from aggregator.charts import (
    chart_channel_comparison,
    chart_cooccurrence_heatmap,
    chart_cramers_v_heatmap,
    chart_crosstab_grouped_bar,
    chart_crosstab_heatmap,
    chart_crosstab_normalized_heatmap,
    chart_dimension_bar,
    chart_engagement_bar,
    chart_pie,
    chart_stacked_bar,
    chart_timeline,
    cleanup_chart,
)
from aggregator.stats import compute_stats
from config.models import ResearchConfig
from config.settings import AppSettings
from storage.database import Database
from storage.models import QueryScope

logger = logging.getLogger(__name__)

stats_router = Router()

_SCALAR_DIMS = {"problem_scope", "disability_model", "agency", "portrayal", "stigma"}

DIMENSION_TITLES: dict[str, str] = {
    "categories": "Категории контента",
    "tones": "Тон сообщений",
    "problem_scope": "Масштаб проблемы",
    "disability_model": "Модель инвалидности",
    "agency": "Агентность",
    "portrayal": "Репрезентация",
    "target_audiences": "Адресат сообщения",
    "barriers": "Барьеры",
    "stigma": "Стигма",
}


# ── Helpers ──

async def _load_context() -> tuple[ResearchConfig, QueryScope, Database] | None:
    """Load last analysis config from metadata, return (config, scope, db) or None."""
    settings = AppSettings.from_env()
    db = Database(settings.database_path)
    await db.connect()

    yaml_str = await db.get_metadata("last_config_yaml")
    if not yaml_str:
        await db.close()
        return None

    try:
        config = ResearchConfig.from_yaml(yaml_str)
    except Exception:
        await db.close()
        return None

    scope = QueryScope(
        channels=config.channels,
        start_date=config.period.start,
        end_date=config.period.end,
    )
    return config, scope, db


async def _send_chart(bot: Bot, chat_id: int, chart_path: Path, caption: str = "") -> None:
    try:
        await bot.send_photo(
            chat_id,
            FSInputFile(str(chart_path)),
            caption=caption[:1024] if caption else None,
        )
    finally:
        cleanup_chart(chart_path)


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{part / total * 100:.1f}%"


def _counts_text(counts: dict[str, int], total: int) -> str:
    lines = []
    for label, cnt in counts.items():
        lines.append(f"  {label}: {cnt} ({_pct(cnt, total)})")
    return "\n".join(lines) if lines else "  Нет данных"


NO_DATA_MSG = (
    "Нет данных для анализа. Сначала запустите пайплайн, "
    "отправив YAML-конфигурацию."
)


# ── /stats command ──

def _stats_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Обзор", callback_data="st:overview"),
            InlineKeyboardButton(text="Ключевые слова", callback_data="st:keywords"),
        ],
        [
            InlineKeyboardButton(text="Категории", callback_data="st:categories"),
            InlineKeyboardButton(text="Тон", callback_data="st:tones"),
        ],
        [
            InlineKeyboardButton(text="Модель инвал.", callback_data="st:disability_model"),
            InlineKeyboardButton(text="Агентность", callback_data="st:agency"),
        ],
        [
            InlineKeyboardButton(text="Репрезентация", callback_data="st:portrayal"),
            InlineKeyboardButton(text="Адресат", callback_data="st:target_audiences"),
        ],
        [
            InlineKeyboardButton(text="Барьеры", callback_data="st:barriers"),
            InlineKeyboardButton(text="Стигма", callback_data="st:stigma"),
        ],
        [
            InlineKeyboardButton(text="Масштаб проблемы", callback_data="st:problem_scope"),
            InlineKeyboardButton(text="Вовлечённость", callback_data="st:engagement"),
        ],
    ])


@stats_router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    ctx = await _load_context()
    if ctx is None:
        await message.answer(NO_DATA_MSG)
        return

    _, _, db = ctx
    await db.close()
    await message.answer(
        "Выберите раздел статистики:",
        reply_markup=_stats_keyboard(),
    )


@stats_router.callback_query(F.data == "st:overview")
async def on_stats_overview(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        stats = await compute_stats(config, scope, db)
        text = (
            f"== Общая статистика ==\n\n"
            f"Каналов: {stats.channels_count}\n"
            f"Период: {stats.period_start} -- {stats.period_end}\n"
            f"Всего сообщений: {stats.total_messages}\n"
            f"Классифицировано LLM: {stats.total_classified}\n"
            f"Упоминание инвалидности: "
            f"{stats.mentions_disability_count} "
            f"({_pct(stats.mentions_disability_count, stats.total_classified)})"
        )
        await callback.answer()
        await bot.send_message(callback.message.chat.id, text)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "st:keywords")
async def on_stats_keywords(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        stats = await compute_stats(config, scope, db)
        lines = ["== Ключевые слова ==\n"]
        for kw_stat in stats.keyword_stats:
            kw = kw_stat["keyword"]
            msg_count = kw_stat["message_count"]
            total_matches = kw_stat["total_matches"]
            lines.append(f'"{kw}": {msg_count} сообщ., {total_matches} совпад.')
            for pattern, pat_count in stats.pattern_stats.get(kw, {}).items():
                lines.append(f"  - {pattern}: {pat_count}")
        if not stats.keyword_stats:
            lines.append("Совпадений не найдено.")

        await callback.answer()
        await bot.send_message(callback.message.chat.id, "\n".join(lines))

        kw_counts = {s["keyword"]: s["message_count"] for s in stats.keyword_stats}
        if kw_counts:
            chart_path = chart_dimension_bar(
                kw_counts, "Сообщения по ключевым словам",
                total=stats.total_messages,
            )
            await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data.startswith("st:"))
async def on_stats_dimension(callback: CallbackQuery, bot: Bot) -> None:
    dim = callback.data[3:]

    if dim in ("overview", "keywords", "engagement"):
        return

    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        total = len(all_cls)
        label_maps = get_label_maps(config)
        dim_labels = label_maps.get(dim, {})
        dim_title = DIMENSION_TITLES.get(dim, dim)

        counter: Counter[str] = Counter()
        meta = {"json_list": dim in ("categories", "tones", "target_audiences", "barriers")}
        for row in all_cls:
            val = row.get(dim, [] if meta["json_list"] else "")
            if isinstance(val, list):
                for v in val:
                    if v:
                        counter[v] += 1
            elif val:
                counter[val] += 1

        translated = translate_ids(dict(counter.most_common()), dim_labels)

        text_lines = [f"== {dim_title} ==\n"]
        text_lines.append(f"Всего классифицировано: {total}\n")
        text_lines.append(_counts_text(translated, total))

        await callback.answer()
        await bot.send_message(callback.message.chat.id, "\n".join(text_lines))

        if dim in _SCALAR_DIMS and len(translated) <= 6:
            chart_path = chart_pie(translated, dim_title)
        else:
            chart_path = chart_dimension_bar(translated, dim_title, total=total)
        await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "st:engagement")
async def on_stats_engagement(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        await callback.answer()

        for dim in ("categories", "tones"):
            eng_df = compute_engagement_stats(all_cls, dim)
            if eng_df.empty:
                continue
            dim_labels = label_maps.get(dim, {})
            eng_df["value"] = eng_df["value"].map(lambda x: dim_labels.get(x, x))
            dim_title = DIMENSION_TITLES.get(dim, dim)

            lines = [f"== Вовлечённость: {dim_title} ==\n"]
            for _, r in eng_df.iterrows():
                lines.append(
                    f"  {r['value']}: {r['views_mean']:.0f} просм., "
                    f"{r['forwards_mean']:.1f} пересыл. (n={r['count']})"
                )
            await bot.send_message(callback.message.chat.id, "\n".join(lines))

            chart_path = chart_engagement_bar(eng_df, f"Просмотры: {dim_title}")
            await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


# ── /charts command ──

def _charts_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        # -- Кросс-таблицы --
        [
            InlineKeyboardButton(
                text="Тон x Категория", callback_data="ch:xt:tones:categories"
            ),
            InlineKeyboardButton(
                text="Агентн. x Репрезент.", callback_data="ch:xt:agency:portrayal"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Модель x Тон", callback_data="ch:xt:disability_model:tones"
            ),
            InlineKeyboardButton(
                text="Тон x Репрезент.", callback_data="ch:xt:tones:portrayal"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Репрезент. x Модель", callback_data="ch:xt:portrayal:disability_model"
            ),
            InlineKeyboardButton(
                text="Стигма x Агентн.", callback_data="ch:xt:stigma:agency"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Репрезент. x Категор.", callback_data="ch:xt:portrayal:categories"
            ),
            InlineKeyboardButton(
                text="Адресат x Образ", callback_data="ch:xt:target_audiences:portrayal"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Тон x Барьеры", callback_data="ch:xt:tones:barriers"
            ),
            InlineKeyboardButton(
                text="Масштаб x Барьеры", callback_data="ch:xt:problem_scope:barriers"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Барьеры x Агентн.", callback_data="ch:xt:barriers:agency"
            ),
            InlineKeyboardButton(
                text="Адресат x Барьеры", callback_data="ch:xt:target_audiences:barriers"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Сила связей (V)", callback_data="ch:cramers"
            ),
        ],
        # -- Ключевые слова --
        [
            InlineKeyboardButton(
                text="Ключевые x Категория", callback_data="ch:kw:categories"
            ),
            InlineKeyboardButton(
                text="Ключевые x Тон", callback_data="ch:kw:tones"
            ),
        ],
        # -- Каналы --
        [
            InlineKeyboardButton(
                text="Канал x Категория", callback_data="ch:chan:categories"
            ),
            InlineKeyboardButton(
                text="Канал x Тон", callback_data="ch:chan:tones"
            ),
        ],
        # -- Общие --
        [
            InlineKeyboardButton(
                text="Временной ряд", callback_data="ch:timeline"
            ),
            InlineKeyboardButton(
                text="Сравнение каналов", callback_data="ch:channels"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Вовлечённость", callback_data="ch:engagement"
            ),
        ],
        [
            InlineKeyboardButton(
                text="Ко-окуренция категорий", callback_data="ch:cooc:categories"
            ),
            InlineKeyboardButton(
                text="Ко-окуренция барьеров", callback_data="ch:cooc:barriers"
            ),
        ],
    ])


@stats_router.message(Command("charts"))
async def cmd_charts(message: Message) -> None:
    ctx = await _load_context()
    if ctx is None:
        await message.answer(NO_DATA_MSG)
        return

    _, _, db = ctx
    await db.close()
    await message.answer(
        "Выберите график:",
        reply_markup=_charts_keyboard(),
    )


@stats_router.callback_query(F.data.startswith("ch:xt:"))
async def on_chart_crosstab(callback: CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    dim_row, dim_col = parts[2], parts[3]

    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        ct = compute_crosstab(all_cls, dim_row, dim_col)
        if ct.empty:
            await callback.answer("Нет данных для кросс-таблицы.", show_alert=True)
            return

        ct = translate_df_index(ct, label_maps.get(dim_row, {}))
        ct = translate_df_columns(ct, label_maps.get(dim_col, {}))

        title_row = DIMENSION_TITLES.get(dim_row, dim_row)
        title_col = DIMENSION_TITLES.get(dim_col, dim_col)
        title = f"{title_row} x {title_col}"

        await callback.answer()

        heatmap_path = chart_crosstab_heatmap(ct, title)
        await _send_chart(bot, callback.message.chat.id, heatmap_path)

        norm_path = chart_crosstab_normalized_heatmap(ct, title)
        await _send_chart(bot, callback.message.chat.id, norm_path)

        stacked_path = chart_stacked_bar(ct, title)
        await _send_chart(bot, callback.message.chat.id, stacked_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data.startswith("ch:kw:"))
async def on_chart_keyword_crosstab(callback: CallbackQuery, bot: Bot) -> None:
    dim = callback.data[6:]

    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        messages = await db.get_messages(scope)
        msg_id_map = {(m.channel, m.message_id): m.id for m in messages}
        kw_map = await db.get_keywords_by_message(scope)
        label_maps = get_label_maps(config)

        ct = compute_keyword_crosstab(all_cls, kw_map, msg_id_map, dim)
        if ct.empty:
            await callback.answer("Нет данных.", show_alert=True)
            return

        ct = translate_df_columns(ct, label_maps.get(dim, {}))
        dim_title = DIMENSION_TITLES.get(dim, dim)
        title = f"Ключевые слова x {dim_title}"

        await callback.answer()

        heatmap_path = chart_crosstab_heatmap(ct, title)
        await _send_chart(bot, callback.message.chat.id, heatmap_path)

        bar_path = chart_crosstab_grouped_bar(ct, title)
        await _send_chart(bot, callback.message.chat.id, bar_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data.startswith("ch:chan:"))
async def on_chart_channel_dim(callback: CallbackQuery, bot: Bot) -> None:
    dim = callback.data[7:]

    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        ct = compute_channel_dimension(all_cls, dim)
        if ct.empty:
            await callback.answer("Нет данных.", show_alert=True)
            return

        ct = translate_df_columns(ct, label_maps.get(dim, {}))
        dim_title = DIMENSION_TITLES.get(dim, dim)
        title = f"Каналы x {dim_title}"

        await callback.answer()

        heatmap_path = chart_crosstab_heatmap(ct, title)
        await _send_chart(bot, callback.message.chat.id, heatmap_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "ch:timeline")
async def on_chart_timeline(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        temporal = await db.get_temporal_message_counts(scope, "week")
        if not temporal:
            await callback.answer("Нет данных.", show_alert=True)
            return

        periods = [r["period"] for r in temporal]
        counts = [r["cnt"] for r in temporal]

        kw_temporal = await db.get_temporal_keyword_counts(scope, "week")
        kw_series: dict[str, list[int]] = {}
        if kw_temporal:
            kw_by_period: dict[str, dict[str, int]] = {}
            for r in kw_temporal:
                kw_by_period.setdefault(r["keyword"], {})[r["period"]] = r["cnt"]
            for kw, period_map in kw_by_period.items():
                kw_series[kw] = [period_map.get(p, 0) for p in periods]

        await callback.answer()

        chart_path = chart_timeline(
            periods, counts,
            "Динамика сообщений по неделям",
            secondary_series=kw_series if kw_series else None,
        )
        await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "ch:channels")
async def on_chart_channels(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        channel_data = await db.get_channel_message_counts(scope)
        if not channel_data:
            await callback.answer("Нет данных.", show_alert=True)
            return

        await callback.answer()

        chart_path = chart_channel_comparison(channel_data, "Сравнение каналов")
        await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "ch:engagement")
async def on_chart_engagement(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        await callback.answer()

        for dim in ("categories", "tones"):
            eng_df = compute_engagement_stats(all_cls, dim)
            if eng_df.empty:
                continue
            dim_labels = label_maps.get(dim, {})
            eng_df["value"] = eng_df["value"].map(lambda x, dl=dim_labels: dl.get(x, x))
            dim_title = DIMENSION_TITLES.get(dim, dim)

            chart_path = chart_engagement_bar(eng_df, f"Просмотры: {dim_title}")
            await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data == "ch:cramers")
async def on_chart_cramers_v(callback: CallbackQuery, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        dims = list(DIMENSION_FIELDS.keys())
        v_matrix = compute_cramers_v_matrix(all_cls, dims)

        dim_label_map = {d: DIMENSION_TITLES.get(d, d) for d in dims}
        v_matrix = v_matrix.rename(index=dim_label_map, columns=dim_label_map)

        await callback.answer()

        chart_path = chart_cramers_v_heatmap(v_matrix, "Сила связей (Cramer's V)")
        await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


@stats_router.callback_query(F.data.startswith("ch:cooc:"))
async def on_chart_cooccurrence(callback: CallbackQuery, bot: Bot) -> None:
    dim = callback.data[8:]

    ctx = await _load_context()
    if ctx is None:
        await callback.answer(NO_DATA_MSG, show_alert=True)
        return

    config, scope, db = ctx
    try:
        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)

        cooc = compute_cooccurrence(all_cls, dim)
        if cooc.empty:
            await callback.answer("Нет данных.", show_alert=True)
            return

        dim_labels = label_maps.get(dim, {})
        cooc = translate_df_index(cooc, dim_labels)
        cooc = translate_df_columns(cooc, dim_labels)

        dim_title = DIMENSION_TITLES.get(dim, dim)
        title = f"Совместная встречаемость: {dim_title}"

        await callback.answer()

        chart_path = chart_cooccurrence_heatmap(cooc, title)
        await _send_chart(bot, callback.message.chat.id, chart_path)
    finally:
        await db.close()


# ── /export_charts command ──

@stats_router.message(Command("export_charts"))
async def cmd_export_charts(message: Message, bot: Bot) -> None:
    ctx = await _load_context()
    if ctx is None:
        await message.answer(NO_DATA_MSG)
        return

    config, scope, db = ctx
    try:
        await message.answer("Генерирую все графики, это может занять некоторое время...")

        all_cls = await db.get_all_classifications_full(scope)
        label_maps = get_label_maps(config)
        stats = await compute_stats(config, scope, db)
        chart_paths: list[Path] = []

        # 1. Keyword bar
        kw_counts = {s["keyword"]: s["message_count"] for s in stats.keyword_stats}
        if kw_counts:
            chart_paths.append(
                chart_dimension_bar(kw_counts, "Ключевые слова", total=stats.total_messages)
            )

        # 2. Dimension bars / pies
        for dim, dim_title in DIMENSION_TITLES.items():
            counter: Counter[str] = Counter()
            is_list = dim in ("categories", "tones", "target_audiences", "barriers")
            for row in all_cls:
                val = row.get(dim, [] if is_list else "")
                if isinstance(val, list):
                    for v in val:
                        if v:
                            counter[v] += 1
                elif val:
                    counter[val] += 1

            if not counter:
                continue

            translated = translate_ids(dict(counter.most_common()), label_maps.get(dim, {}))
            if dim in _SCALAR_DIMS and len(translated) <= 6:
                chart_paths.append(chart_pie(translated, dim_title))
            else:
                chart_paths.append(
                    chart_dimension_bar(translated, dim_title, total=len(all_cls))
                )

        # 3. Cross-tabs (all correlation pairs)
        cross_pairs = [
            ("tones", "categories"),
            ("agency", "portrayal"),
            ("disability_model", "tones"),
            ("tones", "portrayal"),
            ("portrayal", "disability_model"),
            ("stigma", "agency"),
            ("portrayal", "categories"),
            ("target_audiences", "portrayal"),
            ("tones", "barriers"),
            ("problem_scope", "barriers"),
            ("barriers", "agency"),
            ("target_audiences", "barriers"),
        ]
        for dim_row, dim_col in cross_pairs:
            ct = compute_crosstab(all_cls, dim_row, dim_col)
            if ct.empty:
                continue
            ct = translate_df_index(ct, label_maps.get(dim_row, {}))
            ct = translate_df_columns(ct, label_maps.get(dim_col, {}))
            title = f"{DIMENSION_TITLES.get(dim_row, dim_row)} x {DIMENSION_TITLES.get(dim_col, dim_col)}"
            chart_paths.append(chart_crosstab_heatmap(ct, title))
            chart_paths.append(chart_crosstab_normalized_heatmap(ct, title))
            chart_paths.append(chart_stacked_bar(ct, title))

        # 4. Keyword cross-tabs
        messages = await db.get_messages(scope)
        msg_id_map = {(m.channel, m.message_id): m.id for m in messages}
        kw_map = await db.get_keywords_by_message(scope)
        for dim in ("categories", "tones"):
            ct = compute_keyword_crosstab(all_cls, kw_map, msg_id_map, dim)
            if ct.empty:
                continue
            ct = translate_df_columns(ct, label_maps.get(dim, {}))
            title = f"Ключевые слова x {DIMENSION_TITLES.get(dim, dim)}"
            chart_paths.append(chart_crosstab_heatmap(ct, title))

        # 5. Channel comparison
        channel_data = await db.get_channel_message_counts(scope)
        if channel_data:
            chart_paths.append(chart_channel_comparison(channel_data, "Сравнение каналов"))

        # 6. Channel x dimension
        for dim in ("categories", "tones"):
            ct = compute_channel_dimension(all_cls, dim)
            if ct.empty:
                continue
            ct = translate_df_columns(ct, label_maps.get(dim, {}))
            title = f"Каналы x {DIMENSION_TITLES.get(dim, dim)}"
            chart_paths.append(chart_crosstab_heatmap(ct, title))

        # 7. Timeline
        temporal = await db.get_temporal_message_counts(scope, "week")
        if temporal:
            periods = [r["period"] for r in temporal]
            counts = [r["cnt"] for r in temporal]
            kw_temporal = await db.get_temporal_keyword_counts(scope, "week")
            kw_series: dict[str, list[int]] = {}
            if kw_temporal:
                kw_by_period: dict[str, dict[str, int]] = {}
                for r in kw_temporal:
                    kw_by_period.setdefault(r["keyword"], {})[r["period"]] = r["cnt"]
                for kw, period_map in kw_by_period.items():
                    kw_series[kw] = [period_map.get(p, 0) for p in periods]
            chart_paths.append(chart_timeline(
                periods, counts, "Динамика по неделям",
                secondary_series=kw_series if kw_series else None,
            ))

        # 8. Engagement
        for dim in ("categories", "tones"):
            eng_df = compute_engagement_stats(all_cls, dim)
            if eng_df.empty:
                continue
            dim_labels = label_maps.get(dim, {})
            eng_df["value"] = eng_df["value"].map(lambda x, dl=dim_labels: dl.get(x, x))
            chart_paths.append(
                chart_engagement_bar(eng_df, f"Просмотры: {DIMENSION_TITLES.get(dim, dim)}")
            )

        # 9. Co-occurrence
        for dim in ("categories", "barriers"):
            cooc = compute_cooccurrence(all_cls, dim)
            if cooc.empty:
                continue
            dim_labels = label_maps.get(dim, {})
            cooc = translate_df_index(cooc, dim_labels)
            cooc = translate_df_columns(cooc, dim_labels)
            chart_paths.append(
                chart_cooccurrence_heatmap(
                    cooc, f"Ко-окуренция: {DIMENSION_TITLES.get(dim, dim)}"
                )
            )

        # 10. Cramer's V matrix
        dims = list(DIMENSION_FIELDS.keys())
        v_matrix = compute_cramers_v_matrix(all_cls, dims)
        dim_label_map = {d: DIMENSION_TITLES.get(d, d) for d in dims}
        v_matrix = v_matrix.rename(index=dim_label_map, columns=dim_label_map)
        chart_paths.append(
            chart_cramers_v_heatmap(v_matrix, "Сила связей (Cramer's V)")
        )

        if not chart_paths:
            await message.answer("Не удалось сгенерировать ни одного графика.")
            return

        zip_path = Path(gettempdir()) / "tg_research_charts" / "all_charts.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for p in chart_paths:
                zf.write(str(p), p.name)

        await bot.send_document(
            message.chat.id,
            FSInputFile(str(zip_path), filename="charts.zip"),
            caption=f"Все графики ({len(chart_paths)} шт.)",
        )

        for p in chart_paths:
            cleanup_chart(p)
        cleanup_chart(zip_path)

    finally:
        await db.close()
