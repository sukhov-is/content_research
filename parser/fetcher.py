from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from telethon import TelegramClient
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

from config.models import ResearchConfig
from storage.database import Database
from storage.models import Message

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None] | None


def _media_type(media: object) -> str | None:
    if media is None:
        return None
    if isinstance(media, MessageMediaPhoto):
        return "photo"
    if isinstance(media, MessageMediaDocument):
        return "document"
    return "other"


def _message_url(channel: str, message_id: int) -> str:
    username = channel.lstrip("@")
    return f"https://t.me/{username}/{message_id}"


def compute_gaps(
    start: date, end: date, fetched: list[tuple[date, date]]
) -> list[tuple[date, date]]:
    """Subtract already-fetched ranges from [start, end].

    Returns a list of (gap_start, gap_end) date pairs that still need fetching.
    """
    if not fetched:
        return [(start, end)]

    sorted_ranges = sorted(fetched)
    merged: list[tuple[date, date]] = [sorted_ranges[0]]
    for s, e in sorted_ranges[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + timedelta(days=1):
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    gaps: list[tuple[date, date]] = []
    cursor = start
    for s, e in merged:
        if s > end:
            break
        s = max(s, start)
        e = min(e, end)
        if cursor < s:
            gaps.append((cursor, s - timedelta(days=1)))
        cursor = max(cursor, e + timedelta(days=1))
    if cursor <= end:
        gaps.append((cursor, end))
    return gaps


async def _fetch_range(
    client: TelegramClient,
    channel: str,
    start_dt: datetime,
    end_dt: datetime,
    db: Database,
    on_progress: ProgressCallback = None,
) -> int:
    """Fetch messages for a single contiguous date range.

    Returns the number of messages fetched.
    """
    username = channel.lstrip("@")
    entity = await client.get_entity(username)
    count = 0
    batch: list[Message] = []
    batch_size = 100

    async for tg_msg in client.iter_messages(entity, offset_date=end_dt, reverse=False):
        if tg_msg.date < start_dt:
            break
        if tg_msg.date >= end_dt:
            continue

        text = tg_msg.text or ""
        msg = Message(
            channel=channel,
            message_id=tg_msg.id,
            date=tg_msg.date,
            text=text,
            url=_message_url(channel, tg_msg.id),
            media_type=_media_type(tg_msg.media),
            views=tg_msg.views or 0,
            forwards=tg_msg.forwards or 0,
        )
        batch.append(msg)
        count += 1

        if len(batch) >= batch_size:
            await db.insert_messages_batch(batch)
            batch.clear()
            if on_progress:
                on_progress(channel, count, 0)

    if batch:
        await db.insert_messages_batch(batch)

    logger.info("Fetched %d messages from %s", count, channel)
    return count


async def fetch_all_channels(
    client: TelegramClient,
    config: ResearchConfig,
    db: Database,
    on_progress: ProgressCallback = None,
) -> dict[str, int]:
    """Fetch only unfetched date ranges for all configured channels.

    Returns a dict of channel -> newly fetched message count.
    """
    results: dict[str, int] = {}
    for channel in config.channels:
        try:
            fetched_raw = await db.get_fetched_ranges(channel)
            fetched = [
                (date.fromisoformat(s), date.fromisoformat(e))
                for s, e in fetched_raw
            ]
            gaps = compute_gaps(config.period.start, config.period.end, fetched)

            channel_new = 0
            for gap_start, gap_end in gaps:
                start_dt = datetime.combine(
                    gap_start, datetime.min.time(), tzinfo=timezone.utc
                )
                end_dt = datetime.combine(
                    gap_end + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                n = await _fetch_range(
                    client, channel, start_dt, end_dt, db, on_progress
                )
                await db.record_fetch(
                    channel, gap_start.isoformat(), gap_end.isoformat()
                )
                channel_new += n

            results[channel] = channel_new
        except Exception:
            logger.exception("Failed to fetch channel %s", channel)
            results[channel] = 0
    return results
