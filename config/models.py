from __future__ import annotations

import re
from datetime import date

import yaml
from pydantic import BaseModel, field_validator, model_validator


class PeriodConfig(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def start_before_end(self) -> PeriodConfig:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        return self


class CategoryConfig(BaseModel):
    id: str
    label: str
    description: str = ""


class ToneConfig(BaseModel):
    id: str
    label: str


class DimensionOptionConfig(BaseModel):
    id: str
    label: str
    description: str = ""


class KeywordGroupConfig(BaseModel):
    label: str
    patterns: list[str]

    @field_validator("patterns")
    @classmethod
    def at_least_one_pattern(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one pattern is required per keyword group")
        return v


class ClassificationConfig(BaseModel):
    categories: list[CategoryConfig]
    tones: list[ToneConfig]
    problem_scopes: list[DimensionOptionConfig] = []
    disability_models: list[DimensionOptionConfig] = []
    agency_types: list[DimensionOptionConfig] = []
    portrayal_types: list[DimensionOptionConfig] = []
    target_audiences: list[DimensionOptionConfig] = []
    barriers: list[DimensionOptionConfig] = []
    stigma_types: list[DimensionOptionConfig] = []

    @field_validator("categories")
    @classmethod
    def at_least_one_category(cls, v: list[CategoryConfig]) -> list[CategoryConfig]:
        if not v:
            raise ValueError("At least one category is required")
        return v

    @field_validator("tones")
    @classmethod
    def at_least_one_tone(cls, v: list[ToneConfig]) -> list[ToneConfig]:
        if not v:
            raise ValueError("At least one tone is required")
        return v


_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class LLMConfig(BaseModel):
    model: str = "gpt-5.4-mini"
    temperature: float | None = 0.0
    reasoning_effort: str | None = None
    batch_size: int = 5
    max_concurrent: int = 10

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("reasoning_effort")
    @classmethod
    def valid_reasoning_effort(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_VALID_REASONING_EFFORTS)}, got '{v}'"
            )
        return v


class ResearchConfig(BaseModel):
    channels: list[str]
    period: PeriodConfig
    keyword_groups: list[KeywordGroupConfig]
    classification: ClassificationConfig
    llm: LLMConfig = LLMConfig()

    @field_validator("channels")
    @classmethod
    def at_least_one_channel(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one channel is required")
        return v

    @field_validator("channels", mode="after")
    @classmethod
    def normalize_channels(cls, v: list[str]) -> list[str]:
        normalized = []
        for ch in v:
            ch = ch.strip()
            m = re.match(r"https?://t\.me/([^/]+)", ch)
            if m:
                ch = f"@{m.group(1)}"
            if not ch.startswith("@"):
                ch = f"@{ch}"
            normalized.append(ch)
        return normalized

    @classmethod
    def from_yaml(cls, content: str) -> ResearchConfig:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("Config must be a YAML mapping")
        if "keywords" in data and "keyword_groups" not in data:
            data["keyword_groups"] = [
                {"label": kw, "patterns": [kw]}
                for kw in data.pop("keywords")
            ]
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> ResearchConfig:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())
