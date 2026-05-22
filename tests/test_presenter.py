from app.core.dataset import ImageRecord
from app.core.settings import CaptionConfig
from app.ui.presenter import (
    dataset_payload,
    editor_payload,
    empty_dataset_payload,
    empty_editor_payload,
    final_caption_for,
)


def make_record() -> ImageRecord:
    return ImageRecord(
        image_path="sample.png",
        txt_path="sample.txt",
        metadata_path="sample.caption.json",
        file_name="sample.png",
        tags=["1girl", "solo"],
        nl="A girl is standing.",
    )


def test_payload_lengths() -> None:
    record = make_record()
    config = CaptionConfig(joiner=" | ")

    assert len(dataset_payload([record], 0, "ok", config)) == 8
    assert len(editor_payload([record], 0, "ok", config)) == 7
    assert len(empty_dataset_payload("empty")) == 8
    assert len(empty_editor_payload("empty")) == 7


def test_final_caption_for_uses_joiner_and_fixed_order() -> None:
    record = make_record()

    assert (
        final_caption_for(record, CaptionConfig(joiner=" | "))
        == "1girl, solo | A girl is standing."
    )
