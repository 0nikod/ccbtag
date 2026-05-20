import base64
from pathlib import Path

from app.core.dataset import deserialize_records, scan_dataset, serialize_records
from app.ui import events


def test_open_folder_failure_matches_open_outputs() -> None:
    payload = events.open_folder("/path/that/does/not/exist", "caption_json")

    assert len(payload) == 8
    assert payload[-1].startswith("打开失败:")


def test_empty_batch_actions_match_open_outputs() -> None:
    assert len(events.batch_generate_tags([], 0, "", "", "PixAI Tagger v0.9", True)) == 8
    assert len(events.batch_delete_tag([], 0, "", "", "solo")) == 8


def write_image(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


class _Event:
    def __init__(self, index: int) -> None:
        self.index = index


class _TagModel:
    def predict(self, image_path: str, **kwargs: object) -> list[object]:
        name = Path(image_path).stem
        return [_Prediction(f"{name}-generated", 0.9)]


class _Prediction:
    def __init__(self, tag: str, score: float) -> None:
        self.tag = tag
        self.score = score

    def to_dict(self) -> dict[str, object]:
        return {"tag": self.tag, "name": self.tag, "score": self.score, "source": "test"}


class _Registry:
    def get_by_display(self, task: str, display_name: str) -> object:
        assert task == "tag"
        assert display_name == "PixAI Tagger v0.9"
        return _TagModel()


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
    assert payload[3] == ""
    assert payload[4] == ""


def test_batch_generate_tags_skips_current_unsaved_manual_edit(tmp_path: Path, monkeypatch) -> None:
    write_image(tmp_path / "0001.png")
    write_image(tmp_path / "0002.png")
    records = scan_dataset(tmp_path)
    monkeypatch.setattr(events, "REGISTRY", _Registry())

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
    assert updated[1].tags == ["0002-generated"]
    assert payload[6] == 0
