from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.core.dataset import deserialize_records, scan_dataset, serialize_records
from app.ui import events
from tests.helpers import (
    FakeRegistry,
    FakeTagModel,
    build_services,
    make_config,
    write_image,
)


class _Event:
    def __init__(self, index: int) -> None:
        self.index = index


def test_open_folder_failure_matches_open_outputs() -> None:
    payload = events.open_folder("/path/that/does/not/exist", "caption_json")

    assert len(payload) == 8
    assert payload[-1].startswith("打开失败:")


def test_empty_batch_actions_match_open_outputs() -> None:
    assert (
        len(events.batch_generate_tags([], 0, "", "", "PixAI Tagger v0.9", True)) == 8
    )
    assert len(events.batch_delete_tag([], 0, "", "", "solo")) == 8


def test_select_record_returns_editor_payload(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = scan_dataset(tmp_path)

    payload = events.select_record(
        serialize_records(records),
        0,
        "",
        "",
        _Event(1),
    )

    assert len(payload) == 7
    assert payload[5] == 1


def test_previous_and_next_return_editor_payload(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = serialize_records(scan_dataset(tmp_path))

    previous_payload = events.previous_record(records, 0, "", "")
    next_payload = events.next_record(records, 0, "", "")

    assert len(previous_payload) == 7
    assert len(next_payload) == 7


def test_generate_save_batch_and_tag_edit_return_dataset_payload(
    tmp_path: Path,
) -> None:
    write_image(tmp_path / "0001.png")
    records = serialize_records(scan_dataset(tmp_path))
    previous = events.SERVICES
    try:
        events.set_services_for_test(build_services(registry=FakeRegistry()))

        assert len(events.generate_tag(records, 0, "PixAI Tagger v0.9", "", "")) == 8
        assert (
            len(
                events.generate_nl(
                    records,
                    0,
                    "OpenAI Completions",
                    "http://127.0.0.1:8000/v1",
                    "model",
                    "",
                    "",
                    "",
                )
            )
            == 8
        )
        assert len(events.save_current(records, 0, "", "", "caption_json")) == 8
        assert (
            len(
                events.batch_generate_tags(
                    records, 0, "", "", "PixAI Tagger v0.9", True
                )
            )
            == 8
        )
        assert len(events.batch_delete_tag(records, 0, "", "", "solo")) == 8
    finally:
        events.set_services_for_test(previous)


def test_select_record_persists_unsaved_current_form(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = scan_dataset(tmp_path)

    payload = events.select_record(
        serialize_records(records),
        0,
        "1girl, solo",
        "A girl is standing.",
        _Event(1),
    )

    updated = deserialize_records(payload[0])
    assert updated[0].tags == ["1girl", "solo"]
    assert updated[0].nl == "A girl is standing."
    assert updated[0].edited is True
    assert updated[0].dirty is True
    assert payload[2] == ""
    assert payload[3] == ""


def test_batch_generate_tags_skips_current_unsaved_manual_edit(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = scan_dataset(tmp_path)
    previous = events.SERVICES
    try:
        events.set_services_for_test(
            build_services(
                registry=FakeRegistry(
                    tag_model=FakeTagModel(),
                )
            )
        )

        payload = events.batch_generate_tags(
            serialize_records(records),
            0,
            "manual tag",
            "",
            "PixAI Tagger v0.9",
            True,
        )

        updated = deserialize_records(payload[0])
        assert updated[0].tags == ["manual tag"]
        assert updated[0].tag_status == "edited"
        assert updated[0].tag_manual is True
        assert updated[1].tags == ["solo"]
        assert updated[1].dirty is True
        assert payload[6] == 0
    finally:
        events.set_services_for_test(previous)


def test_default_metadata_location_uses_config_ui() -> None:
    previous = events.SERVICES
    try:
        config = make_config(
            ui=replace(previous.config.ui, metadata_location="same_folder")
        )
        events.set_services_for_test(build_services(config=config))

        assert events.default_metadata_location() == "同目录"
    finally:
        events.set_services_for_test(previous)


def test_default_shuffle_tags_uses_config_nl() -> None:
    previous = events.SERVICES
    try:
        config = make_config(nl=replace(previous.config.nl, shuffle_tags=False))
        events.set_services_for_test(build_services(config=config))

        assert events.default_shuffle_tags() is False
    finally:
        events.set_services_for_test(previous)
