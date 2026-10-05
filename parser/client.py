from __future__ import annotations

import asyncio
import logging

from telethon import TelegramClient

from config.settings import AppSettings

logger = logging.getLogger(__name__)

_CLIENT_KWARGS = {
    "device_model": "Desktop",
    "system_version": "Windows 10",
    "app_version": "5.12.1 x64",
    "lang_code": "ru",
    "system_lang_code": "ru",
}

MAX_CONNECT_ATTEMPTS = 3
RETRY_DELAY = 5


async def create_telethon_client(
    settings: AppSettings,
    code_callback=None,
    password_callback=None,
) -> TelegramClient:
    session_file = "data/telethon_session"

    start_kwargs = {"phone": settings.telegram_phone}
    if code_callback is not None:
        start_kwargs["code_callback"] = code_callback
    if password_callback is not None:
        start_kwargs["password"] = password_callback

    for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
        try:
            client = TelegramClient(
                session_file,
                settings.telegram_api_id,
                settings.telegram_api_hash,
                **_CLIENT_KWARGS,
            )
            await client.start(**start_kwargs)
            logger.info("Telethon client started")
            return client
        except Exception:
            logger.exception("Connection attempt %d/%d failed", attempt, MAX_CONNECT_ATTEMPTS)
            if attempt == MAX_CONNECT_ATTEMPTS:
                raise
            await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError("Unreachable")
