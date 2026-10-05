from __future__ import annotations

from pydantic import BaseModel, Field

from config.models import ClassificationConfig


class MessageClassification(BaseModel):
    mentions_disability: bool = False
    categories: list[str] = Field(default_factory=list)
    tones: list[str] = Field(default_factory=list)
    problem_scope: str = ""
    disability_model: str = ""
    agency: str = ""
    portrayal: str = ""
    target_audiences: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)
    stigma: str = ""


class BatchClassificationResponse(BaseModel):
    results: list[MessageClassification]


def build_json_schema(cls_config: ClassificationConfig) -> dict:
    """Build a JSON Schema for structured outputs based on dynamic config."""
    category_ids = [c.id for c in cls_config.categories]
    tone_ids = [t.id for t in cls_config.tones]
    problem_scope_ids = [p.id for p in cls_config.problem_scopes]
    disability_model_ids = [d.id for d in cls_config.disability_models]
    agency_ids = [a.id for a in cls_config.agency_types]

    properties: dict = {
        "mentions_disability": {"type": "boolean"},
        "categories": {
            "type": "array",
            "items": {"type": "string", "enum": category_ids},
        },
        "tones": {
            "type": "array",
            "items": {"type": "string", "enum": tone_ids},
        },
    }
    required = [
        "mentions_disability",
        "categories",
        "tones",
    ]

    if problem_scope_ids:
        properties["problem_scope"] = {
            "type": "string",
            "enum": problem_scope_ids + [""],
        }
        required.append("problem_scope")

    if disability_model_ids:
        properties["disability_model"] = {
            "type": "string",
            "enum": disability_model_ids + [""],
        }
        required.append("disability_model")

    if agency_ids:
        properties["agency"] = {
            "type": "string",
            "enum": agency_ids + [""],
        }
        required.append("agency")

    portrayal_ids = [p.id for p in cls_config.portrayal_types]
    if portrayal_ids:
        properties["portrayal"] = {
            "type": "string",
            "enum": portrayal_ids + [""],
        }
        required.append("portrayal")

    target_audience_ids = [t.id for t in cls_config.target_audiences]
    if target_audience_ids:
        properties["target_audiences"] = {
            "type": "array",
            "items": {"type": "string", "enum": target_audience_ids},
        }
        required.append("target_audiences")

    barrier_ids = [b.id for b in cls_config.barriers]
    if barrier_ids:
        properties["barriers"] = {
            "type": "array",
            "items": {"type": "string", "enum": barrier_ids},
        }
        required.append("barriers")

    stigma_ids = [s.id for s in cls_config.stigma_types]
    if stigma_ids:
        properties["stigma"] = {
            "type": "string",
            "enum": stigma_ids + [""],
        }
        required.append("stigma")

    single_item = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    return {
        "name": "batch_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": single_item,
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    }
