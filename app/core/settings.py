from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.tag_utils import TagRuleConfig, config_from_rules


@dataclass(frozen=True)
class CaptionConfig:
    order: tuple[str, ...] = ("tag", "nl")
    joiner: str = ". "
    save_txt: bool = True
    save_metadata_json: bool = True
    metadata_location: str = "caption_json"


@dataclass(frozen=True)
class NlConfig:
    max_length: int = 128
    language: str = "en"
    use_tags_as_context: bool = True


@dataclass(frozen=True)
class UiDefaults:
    metadata_location: str = "caption_json"
    nl_endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    nl_model_name: str = "gpt-3.5-turbo"
    nl_api_key: str = ""
    shuffle_tags: bool = True


@dataclass(frozen=True)
class AppConfig:
    tag: TagRuleConfig
    nl: NlConfig
    caption: CaptionConfig
    ui: UiDefaults


def load_app_config(config_root: str | Path | None = None) -> AppConfig:
    root = Path(config_root) if config_root is not None else Path(__file__).resolve().parents[1] / "config"
    rules_path = root / "rules.json"
    data = _read_json(rules_path)
    tag = config_from_rules(data)

    nl_data = _mapping(data.get("nl", {}))
    caption_data = _mapping(data.get("caption", {}))

    return AppConfig(
        tag=tag,
        nl=NlConfig(
            max_length=int(nl_data.get("max_length", 128)),
            language=str(nl_data.get("language", "en")),
            use_tags_as_context=bool(nl_data.get("use_tags_as_context", True)),
        ),
        caption=CaptionConfig(
            order=tuple(str(item) for item in caption_data.get("order", ("tag", "nl"))),
            joiner=str(caption_data.get("joiner", ". ")),
            save_txt=bool(caption_data.get("save_txt", True)),
            save_metadata_json=bool(caption_data.get("save_metadata_json", True)),
            metadata_location=str(caption_data.get("metadata_location", "caption_json")),
        ),
        ui=UiDefaults(
            metadata_location=str(caption_data.get("metadata_location", "caption_json")),
            nl_endpoint=str(caption_data.get("nl_endpoint", "http://127.0.0.1:8000/v1/chat/completions")),
            nl_model_name=str(caption_data.get("nl_model_name", "gpt-3.5-turbo")),
            nl_api_key=str(caption_data.get("nl_api_key", "")),
            shuffle_tags=bool(caption_data.get("shuffle_tags", True)),
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("rules.json 必须是对象")
    return data


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}