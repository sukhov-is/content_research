from __future__ import annotations

import asyncio

_pending: dict[int, asyncio.Future[str]] = {}


def create_input_request(user_id: int) -> asyncio.Future[str]:
    old = _pending.pop(user_id, None)
    if old and not old.done():
        old.cancel()
    fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    _pending[user_id] = fut
    return fut


def has_pending_input(user_id: int) -> bool:
    f = _pending.get(user_id)
    return f is not None and not f.done()


def submit_input(user_id: int, value: str) -> bool:
    f = _pending.get(user_id)
    if f and not f.done():
        f.set_result(value)
        return True
    return False


def cancel_input(user_id: int) -> None:
    f = _pending.pop(user_id, None)
    if f and not f.done():
        f.cancel()
