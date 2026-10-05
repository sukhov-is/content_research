from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable

from openai import AsyncOpenAI, RateLimitError

from analyzer.prompts import build_system_prompt, build_user_prompt
from analyzer.schemas import build_json_schema
from config.models import ClassificationConfig, LLMConfig
from storage.database import Database
from storage.models import Classification, Message, QueryScope

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None] | None

MIN_TEXT_LENGTH = 10
_EMOJI_ONLY_RE = re.compile(
    r"^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0\U000024C2-\U0001F251]+$"
)


def _should_skip(msg: Message) -> bool:
    text = (msg.text or "").strip()
    if len(text) < MIN_TEXT_LENGTH:
        return True
    if _EMOJI_ONLY_RE.match(text):
        return True
    return False


async def _call_openai_with_retry(
    client: AsyncOpenAI,
    llm_config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    max_retries: int = 5,
) -> dict:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            kwargs: dict = {
                "model": llm_config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_schema", "json_schema": json_schema},
            }
            if llm_config.reasoning_effort:
                kwargs["reasoning_effort"] = llm_config.reasoning_effort
            if llm_config.temperature is not None and not llm_config.reasoning_effort:
                kwargs["temperature"] = llm_config.temperature

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            logger.warning("Rate limited, retrying in %.1fs (attempt %d)", backoff, attempt + 1)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def classify_batch(
    client: AsyncOpenAI,
    messages: list[Message],
    llm_config: LLMConfig,
    cls_config: ClassificationConfig,
    system_prompt: str,
    json_schema: dict,
) -> list[Classification]:
    indexed_msgs = [(i + 1, m.text or "") for i, m in enumerate(messages)]
    user_prompt = build_user_prompt(indexed_msgs)

    result = await _call_openai_with_retry(
        client, llm_config, system_prompt, user_prompt, json_schema
    )

    classifications: list[Classification] = []
    results_list = result.get("results", [])

    for i, msg in enumerate(messages):
        if i < len(results_list):
            r = results_list[i]
        else:
            r = {}

        classifications.append(
            Classification(
                message_db_id=msg.id,  # type: ignore[arg-type]
                mentions_disability=r.get("mentions_disability", False),
                categories_json=json.dumps(r.get("categories", []), ensure_ascii=False),
                tones_json=json.dumps(r.get("tones", []), ensure_ascii=False),
                problem_scope=r.get("problem_scope", ""),
                disability_model=r.get("disability_model", ""),
                agency=r.get("agency", ""),
                portrayal=r.get("portrayal", ""),
                target_audiences_json=json.dumps(r.get("target_audiences", []), ensure_ascii=False),
                barriers_json=json.dumps(r.get("barriers", []), ensure_ascii=False),
                stigma=r.get("stigma", ""),
                raw_response=json.dumps(r, ensure_ascii=False),
            )
        )

    return classifications


async def run_llm_analysis(
    openai_api_key: str,
    llm_config: LLMConfig,
    cls_config: ClassificationConfig,
    scope: QueryScope,
    db: Database,
    on_progress: ProgressCallback = None,
    system_prompt_override: str | None = None,
) -> int:
    """Classify keyword-matched but unclassified messages in the scope.

    Returns the number of messages classified.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    system_prompt = system_prompt_override or build_system_prompt(cls_config)
    json_schema = build_json_schema(cls_config)

    messages = await db.get_keyword_matched_unclassified(scope)

    filterable = [m for m in messages if not _should_skip(m)]
    skipped = [m for m in messages if _should_skip(m)]

    # Mark skipped messages with empty classification
    if skipped:
        skip_classifications = [
            Classification(
                message_db_id=m.id,  # type: ignore[arg-type]
                mentions_disability=False,
                categories_json="[]",
                tones_json="[]",
            )
            for m in skipped
        ]
        await db.insert_classifications_batch(skip_classifications)

    total = len(filterable)
    processed = 0
    semaphore = asyncio.Semaphore(llm_config.max_concurrent)

    async def process_batch(batch: list[Message]) -> list[Classification]:
        async with semaphore:
            return await classify_batch(
                client, batch, llm_config, cls_config, system_prompt, json_schema
            )

    batches = [
        filterable[i : i + llm_config.batch_size]
        for i in range(0, total, llm_config.batch_size)
    ]

    tasks = [asyncio.create_task(process_batch(batch)) for batch in batches]

    for task in asyncio.as_completed(tasks):
        try:
            classifications = await task
            await db.insert_classifications_batch(classifications)
            processed += len(classifications)
            if on_progress:
                on_progress(processed, total)
        except Exception:
            logger.exception("Failed to classify a batch")

    logger.info(
        "LLM analysis complete: %d classified, %d skipped out of %d",
        processed,
        len(skipped),
        len(messages),
    )
    return processed
