import json
from pathlib import Path

from app.core.dataset import scan_dataset, update_record_text
from tests.helpers import build_services, write_image


def test_process_nl_rules_writes_txt_and_draft_by_default(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    records = scan_dataset(tmp_path)
    update_record_text(
        records[0],
        "solo",
        "A girl stands, warm vibe, smiling. Art style is anime.",
    )

    result = build_services().txt_process.process_nl_rules(
        records,
        "art style, tags include",
        "atmosphere, vibe, art style, artist style, rendering style, aesthetic genre",
        modify_json=False,
        metadata_location="caption_json",
    )

    assert result.changed == 1
    assert records[0].tags == ["solo"]
    assert records[0].nl == "A girl stands, smiling."
    assert (tmp_path / "0001.txt").read_text(encoding="utf-8").strip() == (
        "solo. A girl stands, smiling."
    )
    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert "final_caption" not in metadata
    assert metadata["draft"]["nl"]["text"] == "A girl stands, smiling."


def test_process_nl_rules_can_sync_saved_json(tmp_path: Path) -> None:
    write_image(tmp_path / "0001.png")
    records = scan_dataset(tmp_path)
    update_record_text(
        records[0],
        "solo",
        "A girl stands. Soft atmosphere.",
    )

    result = build_services().txt_process.process_nl_rules(
        records,
        "art style, tags include",
        "atmosphere, vibe, art style, artist style, rendering style, aesthetic genre",
        modify_json=True,
        metadata_location="caption_json",
    )

    assert result.changed == 1
    metadata = json.loads(
        (tmp_path / "caption_json" / "0001.caption.json").read_text(encoding="utf-8")
    )
    assert metadata["final_caption"] == "solo. A girl stands."
    assert metadata["nl"]["text"] == "A girl stands."
    assert "draft" not in metadata
