from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.core.caption import CaptionParts, join_caption, parse_caption_text, split_tag_text
from app.core.file_io import (
    is_image_path,
    metadata_candidates,
    metadata_path_for_image,
    read_json,
    read_text,
    txt_path_for_image,
    write_json,
    write_text,
)


EMPTY = "empty"
GENERATED = "generated"
EDITED = "edited"
ERROR = "error"
SAVED = "saved"


@dataclass
class ImageRecord:
    """One image and its editable caption state."""

    image_path: str
    txt_path: str
    metadata_path: str
    file_name: str
    tags: list[str] = field(default_factory=list)
    tag_details: list[dict[str, Any]] = field(default_factory=list)
    nl: str = ""
    tag_status: str = EMPTY
    nl_status: str = EMPTY
    edited: bool = False
    saved: bool = False
    error: str = ""

    @property
    def final_caption(self) -> str:
        return join_caption(self.tags, self.nl)

    @property
    def overall_status(self) -> str:
        if self.error or self.tag_status == ERROR or self.nl_status == ERROR:
            return "失败"
        if self.saved:
            return "已保存"
        if self.edited:
            return "已编辑"
        if self.tags and self.nl:
            return "已生成"
        if self.tags or self.nl:
            return "部分完成"
        return "未处理"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRecord":
        return cls(**data)


def record_from_image(image_path: Path, metadata_location: str = "caption_json") -> ImageRecord:
    txt_path = txt_path_for_image(image_path)
    metadata_path = metadata_path_for_image(image_path, metadata_location)
    parts = _load_caption_parts(image_path, metadata_location)
    return ImageRecord(
        image_path=str(image_path),
        txt_path=str(txt_path),
        metadata_path=str(metadata_path),
        file_name=image_path.name,
        tags=parts.tags,
        nl=parts.nl,
        tag_status=GENERATED if parts.tags else EMPTY,
        nl_status=GENERATED if parts.nl else EMPTY,
        saved=bool(txt_path.exists()),
    )


def scan_dataset(folder: str | Path, metadata_location: str = "caption_json") -> list[ImageRecord]:
    root = Path(folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"图片文件夹不存在: {root}")
    image_paths = sorted(path for path in root.iterdir() if is_image_path(path))
    return [record_from_image(path, metadata_location) for path in image_paths]


def update_record_text(
    record: ImageRecord,
    tags_text: str,
    nl_text: str,
) -> ImageRecord:
    new_tags = split_tag_text(tags_text)
    new_nl = (nl_text or "").strip()
    tag_changed = new_tags != record.tags
    nl_changed = new_nl != record.nl

    record.tags = new_tags
    record.nl = new_nl

    if not record.tags:
        record.tag_status = EMPTY
        record.tag_details = []
    elif tag_changed:
        record.tag_status = EDITED
        record.tag_details = []

    if not record.nl:
        record.nl_status = EMPTY
    elif nl_changed:
        record.nl_status = EDITED

    if tag_changed or nl_changed:
        record.saved = False
        record.error = ""
    record.edited = _has_manual_edits(record)
    return record


def set_generated_tags(
    record: ImageRecord,
    tags: Iterable[str],
    details: list[dict[str, Any]] | None = None,
) -> ImageRecord:
    record.tags = list(tags)
    record.tag_details = details or []
    record.tag_status = GENERATED if record.tags else EMPTY
    record.saved = False
    record.error = ""
    record.edited = _has_manual_edits(record)
    return record


def set_generated_nl(record: ImageRecord, nl: str) -> ImageRecord:
    record.nl = (nl or "").strip()
    record.nl_status = GENERATED if record.nl else EMPTY
    record.saved = False
    record.error = ""
    record.edited = _has_manual_edits(record)
    return record


def set_error(record: ImageRecord, message: str, task: str | None = None) -> ImageRecord:
    record.error = message
    if task == "tag":
        record.tag_status = ERROR
    elif task == "nl":
        record.nl_status = ERROR
    return record


def save_record(record: ImageRecord, metadata_location: str = "caption_json") -> ImageRecord:
    image_path = Path(record.image_path)
    record.txt_path = str(txt_path_for_image(image_path))
    record.metadata_path = str(metadata_path_for_image(image_path, metadata_location))
    final_caption = record.final_caption
    write_text(Path(record.txt_path), final_caption)
    write_json(Path(record.metadata_path), _metadata_payload(record, final_caption))
    record.saved = True
    record.error = ""
    return record


def table_rows(records: list[ImageRecord]) -> list[list[str]]:
    return [
        [
            str(index),
            record.file_name,
            record.overall_status,
            record.tag_status,
            record.nl_status,
            "是" if record.saved else "否",
            record.error,
        ]
        for index, record in enumerate(records)
    ]


def serialize_records(records: list[ImageRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def deserialize_records(data: list[dict[str, Any]] | None) -> list[ImageRecord]:
    return [ImageRecord.from_dict(item) for item in (data or [])]


def _load_caption_parts(image_path: Path, metadata_location: str) -> CaptionParts:
    for candidate in metadata_candidates(image_path, metadata_location):
        metadata = read_json(candidate)
        if metadata:
            return _parts_from_metadata(metadata)
    return parse_caption_text(read_text(txt_path_for_image(image_path)))


def _parts_from_metadata(metadata: dict[str, Any]) -> CaptionParts:
    raw_tags = metadata.get("tags", [])
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, dict):
                name = item.get("name") or item.get("tag")
                if name:
                    tags.append(str(name))
            elif isinstance(item, str):
                tags.append(item)
    raw_nl = metadata.get("nl", "")
    if isinstance(raw_nl, dict):
        nl = str(raw_nl.get("text", ""))
    else:
        nl = str(raw_nl)
    return CaptionParts(tags=split_tag_text(tags), nl=nl.strip())


def _metadata_payload(record: ImageRecord, final_caption: str) -> dict[str, Any]:
    details_by_name: dict[str, dict[str, Any]] = {}
    if record.tag_status != EDITED:
        for item in record.tag_details:
            name = item.get("name") or item.get("tag")
            if name:
                details_by_name[str(name)] = item

    normalized_details = []
    for tag in record.tags:
        item = details_by_name.get(tag, {})
        normalized_details.append(
            {
                "name": tag,
                "score": item.get("score"),
                "source": item.get("source", "manual"),
            }
        )
    return {
        "image": Path(record.image_path).name,
        "tags": normalized_details,
        "nl": {
            "text": record.nl,
            "source": "manual" if record.nl_status == EDITED else "model",
        },
        "final_caption": final_caption,
        "edited": record.edited,
    }


def _has_manual_edits(record: ImageRecord) -> bool:
    return record.tag_status == EDITED or record.nl_status == EDITED
