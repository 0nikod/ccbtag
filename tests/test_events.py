from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from app.core.dataset import deserialize_records, scan_dataset, serialize_records
from app.services.tag_category_service import TagCategoryService
from app.ui import events
from tests.helpers import (
    FakeRegistry,
    FakeTagModel,
    PredictionStub,
    build_services,
    make_config,
    write_image,
)


class _Event:
    def __init__(self, index: int) -> None:
        self.index = index


def write_tag_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE tags (id INTEGER, name TEXT, alias TEXT, post_count INTEGER, category INTEGER, is_deprecated INTEGER)"
        )
        connection.executemany(
            "INSERT INTO tags (id, name, alias, post_count, category, is_deprecated) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "solo", "", 1, 0, 0),
                (2, "kantoku", "", 1, 1, 0),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_open_folder_failure_matches_open_outputs() -> None:
    payload = events.open_folder("/path/that/does/not/exist", "caption_json")

    assert len(payload) == 8
    assert payload[-1].startswith("打开失败:")


def test_empty_batch_actions_match_open_outputs() -> None:
    assert (
        len(
            events.batch_generate_tags(
                [], 0, "", "", "PixAI Tagger v0.9", ["general"], True
            )
        )
        == 8
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

        assert (
            len(events.generate_tag(records, 0, "PixAI Tagger v0.9", ["general"], "", ""))
            == 8
        )
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
                    records,
                    0,
                    "",
                    "",
                    "PixAI Tagger v0.9",
                    ["general"],
                    True,
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
            ["general"],
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


def test_default_nl_image_resize_mode_uses_config_ui() -> None:
    previous = events.SERVICES
    try:
        config = make_config(
            ui=replace(previous.config.ui, nl_image_resize_mode="1MP")
        )
        events.set_services_for_test(build_services(config=config))

        assert events.default_nl_image_resize_mode() == "1MP"
    finally:
        events.set_services_for_test(previous)


def test_autosave_current_updates_state_and_draft(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    records = serialize_records(scan_dataset(tmp_path))

    payload = events.autosave_current(records, 0, "solo", "draft nl")

    updated = deserialize_records(payload[0])
    assert updated[0].tags == ["solo"]
    assert updated[0].nl == "draft nl"
    assert payload[2] == "solo. draft nl"
    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert metadata["draft"]["nl"]["text"] == "draft nl"


def test_open_folder_restores_draft_overlay(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    metadata_dir = tmp_path / "caption_json"
    metadata_dir.mkdir()
    (metadata_dir / "0001.caption.json").write_text(
        json.dumps(
            {
                "image": "0001.png",
                "tags": [{"name": "saved tag", "score": 0.8, "source": "model"}],
                "nl": {"text": "saved nl", "source": "model"},
                "final_caption": "saved tag. saved nl",
                "tag_manual": False,
                "nl_manual": False,
                "edited": False,
                "draft": {
                    "tags": [{"name": "draft tag", "score": None, "source": "manual"}],
                    "nl": {"text": "draft nl", "source": "manual"},
                    "final_caption": "draft tag. draft nl",
                    "tag_manual": True,
                    "nl_manual": True,
                    "edited": True,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = events.open_folder(str(tmp_path), "caption_json")

    assert payload[3] == "draft tag"
    assert payload[4] == "draft nl"


def test_generate_tag_keeps_only_selected_categories(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    records = serialize_records(scan_dataset(tmp_path))
    previous = events.SERVICES
    try:
        events.set_services_for_test(
            build_services(
                registry=FakeRegistry(
                    tag_model=FakeTagModel(
                        [
                            PredictionStub("solo", 0.9),
                            PredictionStub("kantoku", 0.8),
                        ]
                    )
                ),
                tag_categories=TagCategoryService(write_tag_db(tmp_path / "tags.sqlite")),
            )
        )

        payload = events.generate_tag(
            records,
            0,
            "PixAI Tagger v0.9",
            ["general"],
            "",
            "",
        )

        updated = deserialize_records(payload[0])
        assert updated[0].tags == ["solo"]
    finally:
        events.set_services_for_test(previous)
