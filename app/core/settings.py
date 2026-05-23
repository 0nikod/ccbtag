from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.tag_utils import TagRuleConfig


VALID_METADATA_LOCATIONS = ("caption_json", "same_folder")


@dataclass(frozen=True)
class CaptionConfig:
    joiner: str = ". "
    save_txt: bool = True
    save_metadata_json: bool = True
    metadata_location: str = "caption_json"


@dataclass(frozen=True)
class NlConfig:
    max_length: int = 128
    language: str = "en"
    use_tags_as_context: bool = True
    shuffle_tags: bool = True


@dataclass(frozen=True)
class UiDefaults:
    metadata_location: str = "caption_json"
    nl_endpoint: str = "http://127.0.0.1:8000/v1"
    nl_model_name: str = "toriigate-0.5"
    nl_api_key: str = ""
    nl_image_resize_mode: str = "None"


@dataclass(frozen=True)
class AppConfig:
    tag: TagRuleConfig
    nl: NlConfig
    caption: CaptionConfig
    ui: UiDefaults


def load_app_config(config_root: str | Path | None = None) -> AppConfig:
    root = (
        Path(config_root)
        if config_root is not None
        else Path(__file__).resolve().parents[1] / "config"
    )
    app_path = root / "app.json"
    data = _read_json(app_path)
    config = AppConfig(
        tag=_tag_config(_mapping(data.get("tag", {}))),
        nl=_nl_config(_mapping(data.get("nl", {}))),
        caption=_caption_config(_mapping(data.get("caption", {}))),
        ui=_ui_config(_mapping(data.get("ui", {}))),
    )
    validate_app_config(config)
    return config


def validate_app_config(config: AppConfig) -> AppConfig:
    if not 0.0 <= config.tag.threshold <= 1.0:
        raise ValueError("tag.threshold 必须在 0 到 1 之间")
    if config.tag.max_tags <= 0:
        raise ValueError("tag.max_tags 必须大于 0")
    if config.nl.max_length <= 0:
        raise ValueError("nl.max_length 必须大于 0")
    if config.caption.metadata_location not in VALID_METADATA_LOCATIONS:
        raise ValueError("caption.metadata_location 非法")
    if config.ui.metadata_location not in VALID_METADATA_LOCATIONS:
        raise ValueError("ui.metadata_location 非法")
    if not config.caption.save_txt and not config.caption.save_metadata_json:
        raise ValueError(
            "caption.save_txt 和 caption.save_metadata_json 不能同时为 false"
        )
    return config


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("app.json 必须是对象")
    return data


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _tag_config(data: dict[str, Any]) -> TagRuleConfig:
    return TagRuleConfig(
        threshold=float(data.get("threshold", 0.35)),
        max_tags=int(data.get("max_tags", 80)),
        replace_underscore=bool(data.get("replace_underscore", True)),
        sort_by_score=bool(data.get("sort_by_score", True)),
        blacklist=tuple(str(item) for item in data.get("blacklist", [])),
        replace_rules=dict(data.get("replace_rules", {})),
        separator=str(data.get("separator", ", ")),
    )


def _nl_config(data: dict[str, Any]) -> NlConfig:
    return NlConfig(
        max_length=int(data.get("max_length", 128)),
        language=str(data.get("language", "en")),
        use_tags_as_context=bool(data.get("use_tags_as_context", True)),
        shuffle_tags=bool(data.get("shuffle_tags", True)),
    )


def _caption_config(data: dict[str, Any]) -> CaptionConfig:
    return CaptionConfig(
        joiner=str(data.get("joiner", ". ")),
        save_txt=bool(data.get("save_txt", True)),
        save_metadata_json=bool(data.get("save_metadata_json", True)),
        metadata_location=str(data.get("metadata_location", "caption_json")),
    )


def _ui_config(data: dict[str, Any]) -> UiDefaults:
    return UiDefaults(
        metadata_location=str(data.get("metadata_location", "caption_json")),
        nl_endpoint=str(data.get("nl_endpoint", "http://127.0.0.1:8000/v1")),
        nl_model_name=str(data.get("nl_model_name", "toriigate-0.5")),
        nl_api_key=str(data.get("nl_api_key", "")),
        nl_image_resize_mode=str(data.get("nl_image_resize_mode", "None")),
    )
