from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from aiogram.filters import BaseFilter

from analyzer.prompts import build_system_prompt
from bot.code_input import cancel_input, has_pending_input, submit_input
from bot.confirmations import has_pending_confirmation, resolve_confirmation
from config.models import ResearchConfig

logger = logging.getLogger(__name__)

router = Router()

_CONFIG_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "config_example.yaml"

_active_tasks: dict[int, asyncio.Task] = {}


class AnalysisStates(StatesGroup):
    waiting_prompt_approval = State()


def get_orchestrator():
    """Lazy import to avoid circular dependencies."""
    from orchestrator import run_pipeline
    return run_pipeline


def _prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Подтвердить", callback_data="prompt_approve"),
            InlineKeyboardButton(text="Отменить", callback_data="prompt_cancel"),
        ]
    ])


# ── Commands ──

@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Бот для исследования контента Telegram-каналов.\n\n"
        "Заполните прилагаемый шаблон конфигурации и отправьте его обратно "
        "в формате .yaml для начала анализа.\n\n"
        "Команды:\n"
        "  /start -- приветствие + шаблон конфига\n"
        "  /status -- статус текущего анализа\n"
        "  /cancel -- отмена текущего анализа\n"
        "  /stats -- статистика по результатам анализа\n"
        "  /charts -- графики и визуализации\n"
        "  /export_charts -- выгрузить все графики (ZIP)"
    )
    await message.answer_document(
        FSInputFile(_CONFIG_EXAMPLE_PATH, filename="config_example.yaml"),
        caption="Шаблон конфигурации. Заполните и отправьте обратно.",
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    user_id = message.from_user.id  # type: ignore[union-attr]
    if user_id in _active_tasks and not _active_tasks[user_id].done():
        await message.answer("Анализ в процессе...")
    else:
        await message.answer("Нет активного анализа. Отправьте YAML-файл для начала.")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id  # type: ignore[union-attr]

    current_state = await state.get_state()
    if current_state == AnalysisStates.waiting_prompt_approval:
        await state.clear()
        await message.answer("Согласование промпта отменено.")
        return

    if has_pending_input(user_id):
        cancel_input(user_id)

    if has_pending_confirmation(user_id):
        resolve_confirmation(user_id, approved=False)
        await message.answer("LLM-классификация отменена.")
        return

    task = _active_tasks.get(user_id)
    if task and not task.done():
        task.cancel()
        await message.answer("Анализ отменён.")
    else:
        await message.answer("Нет активного анализа для отмены.")


# ── Config file handler ──

@router.message(F.document)
async def handle_config_file(message: Message, bot: Bot, state: FSMContext) -> None:
    user_id = message.from_user.id  # type: ignore[union-attr]

    if user_id in _active_tasks and not _active_tasks[user_id].done():
        await message.answer("Анализ уже запущен. Используйте /cancel для отмены.")
        return

    doc = message.document
    if not doc:
        return

    file_name = doc.file_name or ""
    if not (file_name.endswith(".yaml") or file_name.endswith(".yml")):
        await message.answer("Пожалуйста, отправьте файл в формате YAML (.yaml / .yml).")
        return

    await message.answer("Получен файл конфигурации. Валидирую...")

    file = await bot.download(doc)
    if file is None:
        await message.answer("Не удалось скачать файл.")
        return

    content = file.read().decode("utf-8")

    try:
        config = ResearchConfig.from_yaml(content)
    except Exception as e:
        await message.answer(f"Ошибка валидации конфигурации:\n{e}")
        return

    system_prompt = build_system_prompt(config.classification)

    await state.set_state(AnalysisStates.waiting_prompt_approval)
    await state.update_data(config_yaml=content, system_prompt=system_prompt)

    await message.answer(
        f"Конфигурация валидна.\n"
        f"Каналов: {len(config.channels)}\n"
        f"Период: {config.period.start} -- {config.period.end}\n"
        f"Групп ключевых слов: {len(config.keyword_groups)}\n"
        f"Категорий: {len(config.classification.categories)}"
    )

    prompt_file = Path(tempfile.gettempdir()) / "system_prompt.txt"
    prompt_file.write_text(system_prompt, encoding="utf-8")

    await message.answer_document(
        FSInputFile(str(prompt_file), filename="system_prompt.txt"),
        caption=(
            "Системный промпт для LLM (файл выше).\n\n"
            "Нажмите «Подтвердить» для запуска анализа "
            "или отправьте текстом свой вариант промпта."
        ),
        reply_markup=_prompt_keyboard(),
    )


# ── Prompt approval: button ──

@router.callback_query(F.data == "prompt_approve")
async def on_prompt_approve(callback: CallbackQuery, bot: Bot, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state != AnalysisStates.waiting_prompt_approval:
        await callback.answer("Нет ожидающего подтверждения.")
        return

    data = await state.get_data()
    await state.clear()
    await callback.answer("Промпт подтверждён.")

    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)

    await _launch_analysis(
        config_yaml=data["config_yaml"],
        system_prompt=data["system_prompt"],
        bot=bot,
        chat_id=callback.message.chat.id if callback.message else 0,
        user_id=callback.from_user.id,
    )


@router.callback_query(F.data == "prompt_cancel")
async def on_prompt_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Отменено.")
    if callback.message:
        await callback.message.edit_text("Согласование промпта отменено.")


# ── Prompt approval: custom text or file ──

@router.message(AnalysisStates.waiting_prompt_approval, F.document)
async def on_custom_prompt_file(message: Message, bot: Bot, state: FSMContext) -> None:
    doc = message.document
    if not doc:
        return

    file_name = doc.file_name or ""
    if not file_name.endswith(".txt"):
        await message.answer("Для промпта принимаются только .txt файлы.")
        return

    file = await bot.download(doc)
    if file is None:
        await message.answer("Не удалось скачать файл.")
        return

    custom_prompt = file.read().decode("utf-8").strip()
    if len(custom_prompt) < 20:
        await message.answer(
            "Промпт слишком короткий (минимум 20 символов). "
            "Отправьте исправленный файл или нажмите «Подтвердить»."
        )
        return

    data = await state.get_data()
    await state.clear()

    await message.answer("Принят пользовательский промпт из файла. Запускаю анализ...")

    await _launch_analysis(
        config_yaml=data["config_yaml"],
        system_prompt=custom_prompt,
        bot=bot,
        chat_id=message.chat.id,
        user_id=message.from_user.id,  # type: ignore[union-attr]
    )


@router.message(AnalysisStates.waiting_prompt_approval, F.text)
async def on_custom_prompt(message: Message, bot: Bot, state: FSMContext) -> None:
    custom_prompt = message.text or ""
    if len(custom_prompt.strip()) < 20:
        await message.answer(
            "Промпт слишком короткий (минимум 20 символов). "
            "Отправьте полный текст промпта или нажмите «Подтвердить»."
        )
        return

    data = await state.get_data()
    await state.clear()

    await message.answer("Принят пользовательский промпт. Запускаю анализ...")

    await _launch_analysis(
        config_yaml=data["config_yaml"],
        system_prompt=custom_prompt,
        bot=bot,
        chat_id=message.chat.id,
        user_id=message.from_user.id,  # type: ignore[union-attr]
    )


# ── Telethon code / password input ──

class _PendingInputFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return message.from_user is not None and has_pending_input(message.from_user.id)


@router.message(_PendingInputFilter(), F.text)
async def on_telethon_input(message: Message) -> None:
    user_id = message.from_user.id  # type: ignore[union-attr]
    value = (message.text or "").strip()
    if submit_input(user_id, value):
        await message.answer("Принято. Выполняю авторизацию...")


# ── LLM confirmation ──

@router.callback_query(F.data == "llm_approve")
async def on_llm_approve(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    if not has_pending_confirmation(user_id):
        await callback.answer("Нет ожидающего подтверждения.")
        return
    resolve_confirmation(user_id, approved=True)
    await callback.answer("LLM-классификация подтверждена.")
    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)


@router.callback_query(F.data == "llm_cancel")
async def on_llm_cancel(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    if not has_pending_confirmation(user_id):
        await callback.answer("Нет ожидающего подтверждения.")
        return
    resolve_confirmation(user_id, approved=False)
    await callback.answer("LLM-классификация отменена.")
    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)


# ── Launch analysis ──

async def _launch_analysis(
    config_yaml: str,
    system_prompt: str,
    bot: Bot,
    chat_id: int,
    user_id: int,
) -> None:
    config = ResearchConfig.from_yaml(config_yaml)

    run_pipeline = get_orchestrator()
    task = asyncio.create_task(
        run_pipeline(
            config=config,
            user_id=user_id,
            bot=bot,
            chat_id=chat_id,
            system_prompt_override=system_prompt,
            config_yaml=config_yaml,
        )
    )
    _active_tasks[user_id] = task

    def _on_done(t: asyncio.Task) -> None:
        _active_tasks.pop(user_id, None)
        if t.cancelled():
            logger.info("Analysis cancelled for user %d", user_id)
        elif t.exception():
            logger.exception("Analysis failed for user %d", user_id, exc_info=t.exception())

    task.add_done_callback(_on_done)
