from __future__ import annotations

import logging
import time

from aiogram import Bot

logger = logging.getLogger(__name__)

MIN_UPDATE_INTERVAL = 5  # seconds between progress updates


class ProgressReporter:
    """Sends progress updates to a Telegram chat, throttled to avoid flooding."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._last_update: float = 0
        self._message_id: int | None = None

    async def send(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_update) < MIN_UPDATE_INTERVAL:
            return
        self._last_update = now
        try:
            if self._message_id:
                await self._bot.edit_message_text(
                    text=text,
                    chat_id=self._chat_id,
                    message_id=self._message_id,
                )
            else:
                msg = await self._bot.send_message(self._chat_id, text)
                self._message_id = msg.message_id
        except Exception:
            logger.debug("Failed to update progress message", exc_info=True)

    async def send_new(self, text: str) -> None:
        """Always send a new message (don't edit previous)."""
        self._message_id = None
        self._last_update = 0
        await self.send(text, force=True)
