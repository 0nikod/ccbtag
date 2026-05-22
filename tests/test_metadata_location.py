from dataclasses import replace

from app.ui import events
from tests.helpers import build_services, make_config


def test_metadata_location_label_and_value_mappings() -> None:
    assert events.metadata_location_value("caption_json") == "caption_json"
    assert events.metadata_location_value("同目录") == "same_folder"
    assert events.metadata_location_label("caption_json") == "caption_json"
    assert events.metadata_location_label("same_folder") == "同目录"


def test_metadata_location_invalid_values_fallback() -> None:
    assert events.metadata_location_value("bad") == "caption_json"
    assert events.metadata_location_label("bad") == "caption_json"


def test_default_metadata_location_returns_ui_label() -> None:
    previous = events.SERVICES
    try:
        config = make_config(
            ui=replace(previous.config.ui, metadata_location="same_folder")
        )
        events.set_services_for_test(build_services(config=config))

        assert events.default_metadata_location() == "同目录"
    finally:
        events.set_services_for_test(previous)
