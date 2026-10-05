from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message


class AuthMiddleware(BaseMiddleware):
    def __init__(self, allowed_user_ids: list[int]) -> None:
        self._allowed = set(allowed_user_ids)

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if not self._allowed:
            return await handler(event, data)
        if event.from_user and event.from_user.id in self._allowed:
            return await handler(event, data)
        await event.answer("У вас нет доступа к этому боту.")
        return None
