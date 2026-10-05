import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.schemas import build_json_schema
from config.models import CategoryConfig, ClassificationConfig, ToneConfig


def test_json_schema_structure():
    cls_config = ClassificationConfig(
        categories=[
            CategoryConfig(id="medical", label="Medical", description="Med"),
            CategoryConfig(id="fundraising", label="Fundraising", description="Fund"),
        ],
        tones=[
            ToneConfig(id="neutral", label="Neutral"),
            ToneConfig(id="mocking", label="Mocking"),
        ],
    )

    schema = build_json_schema(cls_config)
    assert schema["name"] == "batch_classification"
    assert schema["strict"] is True

    inner = schema["schema"]
    assert "results" in inner["properties"]
    items = inner["properties"]["results"]["items"]
    assert "mentions_disability" in items["properties"]
    assert "categories" in items["properties"]

    cat_enum = items["properties"]["categories"]["items"]["enum"]
    assert "medical" in cat_enum
    assert "fundraising" in cat_enum

    tone_enum = items["properties"]["tones"]["items"]["enum"]
    assert "neutral" in tone_enum
    assert "mocking" in tone_enum
