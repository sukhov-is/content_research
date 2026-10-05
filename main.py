from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from bot.handlers import router
from bot.stats_handlers import stats_router
from bot.middleware import AuthMiddleware
from config.settings import AppSettings


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    settings = AppSettings.from_env()
    logger.info("Starting bot...")

    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    if settings.allowed_user_ids:
        dp.message.middleware(AuthMiddleware(settings.allowed_user_ids))

    dp.include_router(router)
    dp.include_router(stats_router)

    logger.info("Bot is polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
