import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models import ResearchConfig


VALID_YAML = """\
channels:
  - "@test_channel"
  - "https://t.me/another_channel"

period:
  start: "2026-01-01"
  end: "2026-01-31"

keyword_groups:
  - label: "Инвалидность"
    patterns:
      - "инвалид\\\\w*"
  - label: "ОВЗ"
    patterns:
      - "ОВЗ"

classification:
  categories:
    - id: "medical"
      label: "Medical"
      description: "Medical context"
  tones:
    - id: "neutral"
      label: "Neutral"

llm:
  model: "gpt-5.4-mini"
  batch_size: 3
"""


def test_parse_valid_config():
    config = ResearchConfig.from_yaml(VALID_YAML)
    assert len(config.channels) == 2
    assert config.channels[0] == "@test_channel"
    assert config.channels[1] == "@another_channel"
    assert config.period.start.isoformat() == "2026-01-01"
    assert config.period.end.isoformat() == "2026-01-31"
    assert len(config.keyword_groups) == 2
    assert config.keyword_groups[0].label == "Инвалидность"
    assert config.classification.categories[0].id == "medical"
    assert config.classification.tones[0].id == "neutral"
    assert config.llm.batch_size == 3


def test_channel_normalization():
    yaml_content = """\
channels:
  - "https://t.me/my_channel"
  - "test_channel"
  - "@already_prefixed"
period:
  start: "2026-01-01"
  end: "2026-01-31"
keyword_groups:
  - label: "test"
    patterns: ["test"]
classification:
  categories:
    - id: "cat1"
      label: "Cat"
  tones:
    - id: "tone1"
      label: "Tone"
"""
    config = ResearchConfig.from_yaml(yaml_content)
    assert config.channels == ["@my_channel", "@test_channel", "@already_prefixed"]


def test_invalid_period():
    yaml_content = """\
channels:
  - "@ch"
period:
  start: "2026-02-01"
  end: "2026-01-01"
keyword_groups:
  - label: "test"
    patterns: ["test"]
classification:
  categories:
    - id: "c"
      label: "C"
  tones:
    - id: "t"
      label: "T"
"""
    with pytest.raises(Exception):
        ResearchConfig.from_yaml(yaml_content)


def test_empty_channels_rejected():
    yaml_content = """\
channels: []
period:
  start: "2026-01-01"
  end: "2026-01-31"
keyword_groups:
  - label: "test"
    patterns: ["test"]
classification:
  categories:
    - id: "c"
      label: "C"
  tones:
    - id: "t"
      label: "T"
"""
    with pytest.raises(Exception):
        ResearchConfig.from_yaml(yaml_content)
