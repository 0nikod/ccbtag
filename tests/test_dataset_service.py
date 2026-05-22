from pathlib import Path
import json

from app.core.dataset import GENERATED, ImageRecord
from app.services.dataset_service import DatasetService


def test_open_folder_without_images_returns_empty_message(tmp_path: Path) -> None:
    result = DatasetService().open_folder(tmp_path)

    assert result.records == []
    assert result.index == 0
    assert result.message == "未找到图片"


def test_select_record_switches_after_syncing_current_form() -> None:
    records = [
        ImageRecord("1.png", "1.txt", "1.json", "1.png"),
        ImageRecord("2.png", "2.txt", "2.json", "2.png"),
    ]

    result = DatasetService().select_record(
        records, 0, "solo", "A girl is standing.", 1
    )

    assert result.index == 1
    assert records[0].tags == ["solo"]
    assert records[0].nl == "A girl is standing."
    assert records[0].dirty is True


def test_previous_and_next_do_not_cross_bounds() -> None:
    service = DatasetService()
    records = [
        ImageRecord("1.png", "1.txt", "1.json", "1.png"),
        ImageRecord("2.png", "2.txt", "2.json", "2.png"),
    ]

    assert service.previous_record(records, 0, "", "").index == 0
    assert service.next_record(records, 1, "", "").index == 1


def test_sync_current_form_keeps_update_record_text_behavior() -> None:
    record = ImageRecord(
        image_path="a.png",
        txt_path="a.txt",
        metadata_path="a.caption.json",
        file_name="a.png",
        tags=["1girl"],
        nl="A girl is standing.",
        tag_status=GENERATED,
        nl_status=GENERATED,
    )

    DatasetService().sync_current_form([record], 0, "1girl", "A girl is sitting.")

    assert record.tag_status == GENERATED
    assert record.nl_status == "edited"
    assert record.nl_manual is True
    assert record.tag_manual is False
    assert record.dirty is True


def test_clamp_index_handles_none_negative_and_overflow() -> None:
    service = DatasetService()
    records = [ImageRecord("1.png", "1.txt", "1.json", "1.png")]

    assert service.clamp_index(records, None) == 0
    assert service.clamp_index(records, -1) == 0
    assert service.clamp_index(records, 99) == 0


def test_sync_current_form_persists_draft(tmp_path: Path) -> None:
    from tests.helpers import write_image
    from app.core.dataset import scan_dataset

    write_image(tmp_path / "0001.png")
    records = scan_dataset(tmp_path)

    DatasetService().sync_current_form(records, 0, "solo", "draft nl")

    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert metadata["draft"]["tags"][0]["name"] == "solo"
    assert metadata["draft"]["nl"]["text"] == "draft nl"
