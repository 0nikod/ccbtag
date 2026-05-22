import json
from pathlib import Path

import pytest

from app.core.settings import load_app_config


def write_config(root: Path, payload: dict[str, object]) -> None:
    (root / "app.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def valid_payload() -> dict[str, object]:
    return {
        "tag": {
            "separator": ", ",
            "threshold": 0.35,
            "max_tags": 80,
            "replace_underscore": True,
            "sort_by_score": True,
            "blacklist": ["lowres"],
            "replace_rules": {"grey hair": "gray hair"},
        },
        "nl": {
            "max_length": 128,
            "language": "en",
            "use_tags_as_context": True,
            "shuffle_tags": False,
        },
        "caption": {
            "joiner": " | ",
            "save_txt": True,
            "save_metadata_json": True,
            "metadata_location": "same_folder",
        },
        "ui": {
            "metadata_location": "caption_json",
            "nl_endpoint": "http://127.0.0.1:8000/v1",
            "nl_model_name": "toriigate-0.5",
            "nl_api_key": "secret",
        },
    }


def test_load_app_config_reads_app_json(tmp_path: Path) -> None:
    write_config(tmp_path, valid_payload())

    config = load_app_config(tmp_path)

    assert config.tag.threshold == 0.35
    assert config.tag.blacklist == ("lowres",)
    assert config.nl.shuffle_tags is False
    assert config.caption.metadata_location == "same_folder"
    assert config.caption.joiner == " | "
    assert config.caption.save_txt is True
    assert config.caption.save_metadata_json is True
    assert config.ui.metadata_location == "caption_json"
    assert config.ui.nl_endpoint == "http://127.0.0.1:8000/v1"
    assert config.ui.nl_model_name == "toriigate-0.5"
    assert config.ui.nl_api_key == "secret"


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("tag", "threshold"), 1.5),
        (("tag", "max_tags"), 0),
        (("nl", "max_length"), 0),
        (("caption", "metadata_location"), "bad"),
        (("ui", "metadata_location"), "bad"),
    ],
)
def test_invalid_config_values_raise(
    tmp_path: Path, field_path: tuple[str, str], value: object
) -> None:
    payload = valid_payload()
    payload[field_path[0]][field_path[1]] = value
    write_config(tmp_path, payload)

    with pytest.raises(ValueError):
        load_app_config(tmp_path)


def test_save_flags_cannot_both_be_false(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["caption"]["save_txt"] = False
    payload["caption"]["save_metadata_json"] = False
    write_config(tmp_path, payload)

    with pytest.raises(ValueError):
        load_app_config(tmp_path)
