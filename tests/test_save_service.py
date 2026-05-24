import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.dataset import scan_dataset
from tests.helpers import build_services, make_config, write_image


def test_save_current_syncs_form_and_updates_record_state(
    tmp_path: Path, caplog
) -> None:
    write_image(tmp_path / "0001.png")
    records = scan_dataset(tmp_path)
    services = build_services()

    with caplog.at_level(logging.INFO):
        result = services.save.save_current(
            records,
            0,
            "solo",
            "A girl is standing.",
            "caption_json",
        )

    assert result.records[0].saved is True
    assert result.records[0].dirty is False
    assert result.records[0].tags == ["solo"]
    assert result.records[0].nl == "A girl is standing."
    assert any("Saving current caption:" in record.message for record in caplog.records)
    assert any(
        "Saved current caption:" in record.message and "elapsed=" in record.message
        for record in caplog.records
    )


def test_save_all_syncs_current_form(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = scan_dataset(tmp_path)
    services = build_services()

    result = services.save.save_all(records, 0, "solo", "", "caption_json")

    assert result.records[0].tags == ["solo"]
    assert all(record.saved for record in result.records)


def test_save_flags_control_written_files(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    base_config = make_config()
    no_txt_config = make_config(
        caption=replace(base_config.caption, save_txt=False, save_metadata_json=True)
    )
    no_json_config = make_config(
        caption=replace(base_config.caption, save_txt=True, save_metadata_json=False)
    )

    no_txt_services = build_services(config=no_txt_config)
    no_json_services = build_services(config=no_json_config)

    no_txt_record = scan_dataset(tmp_path)[0]
    no_txt_services.save.save_current(
        no_txt_record and [no_txt_record], 0, "solo", "", "caption_json"
    )

    assert not (tmp_path / "0001.txt").exists()
    assert (tmp_path / "caption_json" / "0001.caption.json").exists()

    no_json_record = scan_dataset(tmp_path)[1]
    no_json_services.save.save_current([no_json_record], 0, "solo", "", "same_folder")

    assert (tmp_path / "0002.txt").exists()
    assert not (tmp_path / "0002.caption.json").exists()


def test_save_service_rejects_both_flags_false(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    base_config = make_config()
    config = make_config(
        caption=replace(
            base_config.caption,
            save_txt=False,
            save_metadata_json=False,
        )
    )
    services = build_services(config=config)
    records = scan_dataset(tmp_path)

    with pytest.raises(ValueError):
        services.save.save_current(records, 0, "solo", "", "caption_json")


def test_joiner_affects_txt_and_metadata_final_caption(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    base_config = make_config()
    config = make_config(caption=replace(base_config.caption, joiner=" | "))
    services = build_services(config=config)
    records = scan_dataset(tmp_path)

    services.save.save_current(
        records,
        0,
        "solo",
        "A girl is standing.",
        "caption_json",
    )

    assert (tmp_path / "0001.txt").read_text(
        encoding="utf-8"
    ).strip() == "solo | A girl is standing."
    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert metadata["final_caption"] == "solo | A girl is standing."


def test_metadata_location_parameter_overrides_config(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    base_config = make_config()
    config = make_config(
        caption=replace(base_config.caption, metadata_location="caption_json")
    )
    services = build_services(config=config)
    records = scan_dataset(tmp_path)

    services.save.save_current(records, 0, "solo", "", "same_folder")

    assert (tmp_path / "0001.caption.json").exists()


def test_metadata_location_defaults_to_config(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    base_config = make_config()
    config = make_config(
        caption=replace(base_config.caption, metadata_location="same_folder")
    )
    services = build_services(config=config)
    records = scan_dataset(tmp_path)

    services.save.save_current(records, 0, "solo", "", None)

    assert (tmp_path / "0001.caption.json").exists()


def test_save_current_clears_existing_draft(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    records = scan_dataset(tmp_path)
    services = build_services()

    services.dataset.sync_current_form(records, 0, "solo", "draft nl")
    services.save.save_current(records, 0, "solo", "draft nl", "caption_json")

    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert "draft" not in metadata


def test_save_current_removes_draft_only_metadata_when_json_save_disabled(
    tmp_path: Path,
) -> None:
    write_image(tmp_path / "0001.png")
    base_config = make_config()
    config = make_config(
        caption=replace(base_config.caption, save_txt=True, save_metadata_json=False)
    )
    services = build_services(config=config)
    records = scan_dataset(tmp_path)

    services.dataset.sync_current_form(records, 0, "solo", "draft nl")
    services.save.save_current(records, 0, "solo", "draft nl", "caption_json")

    assert not (tmp_path / "caption_json" / "0001.caption.json").exists()
