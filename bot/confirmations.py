from __future__ import annotations

import asyncio

_llm_confirm_events: dict[int, asyncio.Event] = {}
_llm_confirm_results: dict[int, bool] = {}


def create_confirmation(user_id: int) -> asyncio.Event:
    event = asyncio.Event()
    _llm_confirm_events[user_id] = event
    _llm_confirm_results[user_id] = False
    return event


def resolve_confirmation(user_id: int, approved: bool) -> None:
    _llm_confirm_results[user_id] = approved
    event = _llm_confirm_events.pop(user_id, None)
    if event:
        event.set()


def get_confirmation_result(user_id: int) -> bool:
    return _llm_confirm_results.pop(user_id, False)


def has_pending_confirmation(user_id: int) -> bool:
    return user_id in _llm_confirm_events
