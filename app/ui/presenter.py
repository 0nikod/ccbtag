from __future__ import annotations

from typing import Any, TypeAlias

from app.core.caption import join_caption, tag_text
from app.core.dataset import ImageRecord, serialize_records, table_rows
from app.core.preview import image_preview_html
from app.core.settings import CaptionConfig


DatasetPayload: TypeAlias = tuple[Any, ...]
EditorPayload: TypeAlias = tuple[Any, ...]


def final_caption_for(record: ImageRecord, caption_config: CaptionConfig) -> str:
    return join_caption(record.tags, record.nl, joiner=caption_config.joiner)


def dataset_payload(
    records: list[ImageRecord],
    index: int,
    message: str,
    caption_config: CaptionConfig,
) -> DatasetPayload:
    record = records[index]
    return (
        serialize_records(records),
        table_rows(records),
        image_preview_html(record.image_path),
        tag_text(record.tags),
        record.nl,
        final_caption_for(record, caption_config),
        index,
        message,
    )


def editor_payload(
    records: list[ImageRecord],
    index: int,
    message: str,
    caption_config: CaptionConfig,
) -> EditorPayload:
    record = records[index]
    return (
        serialize_records(records),
        image_preview_html(record.image_path),
        tag_text(record.tags),
        record.nl,
        final_caption_for(record, caption_config),
        index,
        message,
    )


def empty_dataset_payload(message: str) -> DatasetPayload:
    return ([], [], image_preview_html(None), "", "", "", 0, message)


def empty_editor_payload(message: str) -> EditorPayload:
    return ([], image_preview_html(None), "", "", "", 0, message)
