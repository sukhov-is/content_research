from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from aiogram import Bot
from aiogram.types import FSInputFile

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from aggregator.export import export_messages_xlsx, export_summary_xlsx
from aggregator.report import format_telegram_report
from aggregator.stats import compute_stats
from analyzer.keywords import analyze_keywords
from analyzer.llm import run_llm_analysis
from bot.code_input import cancel_input
from bot.confirmations import create_confirmation, get_confirmation_result
from bot.progress import ProgressReporter
from config.models import ResearchConfig
from config.settings import AppSettings
from parser.client import create_telethon_client
from parser.fetcher import fetch_all_channels
from storage.database import Database
from storage.models import QueryScope, Run

logger = logging.getLogger(__name__)


async def run_pipeline(
    config: ResearchConfig,
    user_id: int,
    bot: Bot,
    chat_id: int,
    system_prompt_override: str | None = None,
    config_yaml: str = "",
) -> None:
    settings = AppSettings.from_env()
    db = Database(settings.database_path)
    progress = ProgressReporter(bot, chat_id)
    telethon_client = None

    scope = QueryScope(
        channels=config.channels,
        start_date=config.period.start,
        end_date=config.period.end,
    )

    try:
        await db.connect()

        run = Run(
            config_hash="",
            status="collecting",
            user_id=user_id,
            created_at=datetime.utcnow(),
        )
        run_id = await db.create_run(run)

        # ── Phase 1: Collect messages ──
        await progress.send_new("Фаза 1/4: Сбор сообщений из каналов...")

        loop = asyncio.get_running_loop()

        async def _code_callback():
            await bot.send_message(
                chat_id,
                "Telegram отправил код подтверждения.\n"
                "Введите его в консоль (терминал), где запущен бот.\n\n"
                "Код нельзя отправлять через Telegram -- он заблокирует вход.",
            )
            code = await loop.run_in_executor(
                None, lambda: input("Введите код подтверждения: ")
            )
            return code.strip()

        async def _password_callback():
            await bot.send_message(
                chat_id,
                "Требуется пароль двухфакторной аутентификации (2FA).\n"
                "Введите его в консоль (терминал), где запущен бот.",
            )
            code = await loop.run_in_executor(
                None, lambda: input("Введите пароль 2FA: ")
            )
            return code.strip()

        telethon_client = await create_telethon_client(
            settings,
            code_callback=_code_callback,
            password_callback=_password_callback,
        )

        def on_fetch_progress(channel: str, count: int, _: int) -> None:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(
                progress.send(f"Сбор: {channel} -- {count} сообщений...")
            )

        new_counts = await fetch_all_channels(
            telethon_client, config, db, on_fetch_progress
        )

        total_new = sum(new_counts.values())
        total_msgs = await db.count_messages(scope)
        total_cached = total_msgs - total_new

        await db.update_run_counts(run_id, total=total_msgs, processed=0)
        await db.update_run_status(run_id, "keyword_analysis")

        channels_info = "\n".join(
            f"  {ch}: +{cnt} новых" for ch, cnt in new_counts.items()
        )
        cache_info = (
            f"Из кэша: {total_cached}" if total_cached > 0 else "Кэш пуст"
        )
        await progress.send_new(
            f"Фаза 1 завершена.\n"
            f"Всего сообщений в периоде: {total_msgs}\n"
            f"{cache_info}, загружено: {total_new}\n{channels_info}"
        )

        if total_msgs == 0:
            await db.update_run_status(run_id, "done")
            await progress.send_new("Сообщений не найдено. Анализ завершён.")
            return

        # ── Phase 2: Keyword analysis ──
        await progress.send_new("Фаза 2/4: Поиск по ключевым словам...")
        kw_counts = await analyze_keywords(config.keyword_groups, scope, db)

        kw_info = "\n".join(f'  "{kw}": {cnt} сообщ.' for kw, cnt in kw_counts.items())
        await progress.send_new(
            f"Фаза 2 завершена.\nКлючевые слова:\n{kw_info}"
        )
        await db.update_run_status(run_id, "llm_analysis")

        # ── Confirmation before LLM ──
        unclassified = await db.get_keyword_matched_unclassified(scope)
        already_classified = await db.get_classification_stats(scope)
        cached_cls = already_classified["total_classified"]

        if not unclassified and cached_cls > 0:
            await progress.send_new(
                f"Все {cached_cls} сообщений уже классифицированы (из кэша). "
                "Пропускаю LLM-фазу."
            )
        elif unclassified:
            est_requests = max(1, len(unclassified) // config.llm.batch_size)

            confirm_kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="Подтвердить", callback_data="llm_approve"),
                InlineKeyboardButton(text="Отменить", callback_data="llm_cancel"),
            ]])
            cls_cache_note = (
                f"\nУже классифицировано (кэш): {cached_cls}"
                if cached_cls > 0
                else ""
            )
            await bot.send_message(
                chat_id,
                f"Готово к LLM-классификации.\n\n"
                f"Сообщений для анализа: {len(unclassified)}{cls_cache_note}\n"
                f"Размер батча: {config.llm.batch_size}\n"
                f"Запросов к API: ~{est_requests}\n"
                f"Модель: {config.llm.model}\n\n"
                f"Подтвердите отправку запросов к LLM.",
                reply_markup=confirm_kb,
            )

            confirm_event = create_confirmation(user_id)
            await confirm_event.wait()

            if not get_confirmation_result(user_id):
                await db.update_run_status(run_id, "cancelled")
                await progress.send_new("LLM-классификация отменена пользователем.")
                return

            # ── Phase 3: LLM classification ──
            await progress.send_new("Фаза 3/4: LLM-классификация сообщений...")

            def on_llm_progress(processed: int, total: int) -> None:
                import asyncio
                loop = asyncio.get_event_loop()
                pct = processed / total * 100 if total else 0
                loop.create_task(
                    progress.send(
                        f"Классификация: {processed}/{total} ({pct:.0f}%)"
                    )
                )

            classified = await run_llm_analysis(
                openai_api_key=settings.openai_api_key,
                llm_config=config.llm,
                cls_config=config.classification,
                scope=scope,
                db=db,
                on_progress=on_llm_progress,
                system_prompt_override=system_prompt_override,
            )

            await db.update_run_counts(run_id, total=total_msgs, processed=classified)
            await db.update_run_status(run_id, "aggregating")
            await progress.send_new(
                f"Фаза 3 завершена. Новых классификаций: {classified}"
            )
        else:
            await progress.send_new(
                "Нет сообщений для LLM-классификации. Пропускаю фазу 3."
            )

        # ── Phase 4: Aggregation & report ──
        await progress.send_new("Фаза 4/4: Агрегация и формирование отчёта...")

        stats = await compute_stats(config, scope, db)
        telegram_messages = format_telegram_report(stats, config)

        for msg_text in telegram_messages:
            try:
                await bot.send_message(chat_id, msg_text)
            except TelegramBadRequest:
                truncated = msg_text[:4000] + "\n\n[...обрезано]"
                await bot.send_message(chat_id, truncated)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        detail_path = await export_messages_xlsx(
            scope, db, config, f"data/reports/messages_detail_{ts}.xlsx"
        )
        summary_path = await export_summary_xlsx(
            stats, config, f"data/reports/statistics_summary_{ts}.xlsx"
        )

        await bot.send_document(chat_id, FSInputFile(str(detail_path)))
        await bot.send_document(chat_id, FSInputFile(str(summary_path)))

        if config_yaml:
            await db.set_metadata("last_config_yaml", config_yaml)

        await db.update_run_status(run_id, "done")
        await progress.send_new("Анализ завершён.")

    except Exception:
        logger.exception("Pipeline failed for user %d", user_id)
        await bot.send_message(chat_id, "Произошла ошибка при анализе. Проверьте логи.")
        raise
    finally:
        cancel_input(user_id)
        await db.close()
        if telethon_client:
            await telethon_client.disconnect()
