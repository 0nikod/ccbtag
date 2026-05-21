from __future__ import annotations

import pytest

from app.models.base import ModelConfig
from app.models.taggers.cl_tagger_onnx import CLTaggerOnnx


def _cl_config() -> ModelConfig:
    return ModelConfig(
        id="cl_tagger_1_02",
        display_name="CL Tagger",
        task="tag",
        source="local",
        backend="onnx",
        model_path="nikoovo/cl-tagger",
        entry="CLTaggerOnnx",
    )


def test_extract_tags_handles_digit_keys_with_nested_dicts() -> None:
    model = CLTaggerOnnx(_cl_config())
    
    data = {
        "0": {"tag": "general", "category": "Rating"},
        "1": {"tag": "1girl", "category": "General"},
        "2": {"name": "cat"},
        "3": "dog"
    }
    
    tags = model._extract_tags(data)
    assert tags == ["general", "1girl", "cat", "dog"]


def test_extract_tags_handles_legacy_digit_keys() -> None:
    model = CLTaggerOnnx(_cl_config())
    data = {
        "0": "1girl",
        "1": "solo",
        "2": "cat"
    }
    
    tags = model._extract_tags(data)
    assert tags == ["1girl", "solo", "cat"]


def test_extract_tags_handles_well_known_keys() -> None:
    model = CLTaggerOnnx(_cl_config())
    
    data = {
        "metadata": "some info",
        "tag_mapping": {
            "0": {"name": "1girl"},
            "1": {"name": "solo"}
        }
    }
    
    tags = model._extract_tags(data)
    assert tags == ["1girl", "solo"]

def test_extract_tags_handles_digit_values_as_indices() -> None:
    model = CLTaggerOnnx(_cl_config())
    
    data = {
        "solo": "1",
        "1girl": 0,
        "cat": "2"
    }
    
    tags = model._extract_tags(data)
    assert tags == ["1girl", "solo", "cat"]

def test_extract_tags_empty_data() -> None:
    model = CLTaggerOnnx(_cl_config())
    assert model._extract_tags({}) == []
    assert model._extract_tags([]) == []
    assert model._extract_tags({"unknown_key": "value"}) == []
