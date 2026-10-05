from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()


class AppSettings(BaseModel):
    bot_token: str
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone: str
    openai_api_key: str
    allowed_user_ids: list[int]
    database_path: Path = Path("data/research.db")

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_user_ids(cls, v: str | list[int]) -> list[int]:
        if isinstance(v, str):
            return [int(uid.strip()) for uid in v.split(",") if uid.strip()]
        return v

    @classmethod
    def from_env(cls) -> AppSettings:
        return cls(
            bot_token=os.environ["BOT_TOKEN"],
            telegram_api_id=int(os.environ["TELEGRAM_API_ID"]),
            telegram_api_hash=os.environ["TELEGRAM_API_HASH"],
            telegram_phone=os.environ["TELEGRAM_PHONE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            allowed_user_ids=os.environ.get("ALLOWED_USER_IDS", ""),
            database_path=Path(os.environ.get("DATABASE_PATH", "data/research.db")),
        )
