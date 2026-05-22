from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import UTC, datetime
from typing import Any, Iterable

from app.core.caption import (
    CaptionParts,
    join_caption,
    parse_caption_text,
    split_tag_text,
)
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
    tag_manual: bool = False
    nl_manual: bool = False
    edited: bool = False
    saved: bool = False
    dirty: bool = False
    error: str = ""

    @property
    def final_caption(self) -> str:
        return join_caption(self.tags, self.nl)

    @property
    def overall_status(self) -> str:
        if self.error or self.tag_status == ERROR or self.nl_status == ERROR:
            return "失败"
        if self.dirty:
            return "未保存"
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
        payload = dict(data)
        tags = payload.get("tags", [])
        nl = str(payload.get("nl", "") or "")
        tag_manual = payload.get("tag_manual")
        nl_manual = payload.get("nl_manual")
        edited = bool(payload.get("edited", False))

        if tag_manual is None:
            tag_manual = payload.get("tag_status") == EDITED or (edited and bool(tags))
        if nl_manual is None:
            nl_manual = payload.get("nl_status") == EDITED or (
                edited and bool(nl.strip())
            )

        payload["tag_manual"] = bool(tag_manual)
        payload["nl_manual"] = bool(nl_manual)
        payload["edited"] = bool(payload["tag_manual"] or payload["nl_manual"])
        payload["dirty"] = bool(payload.get("dirty", False))
        return cls(**payload)


@dataclass(frozen=True)
class PersistedRecordState:
    parts: CaptionParts
    tag_details: list[dict[str, Any]] = field(default_factory=list)
    tag_manual: bool = False
    nl_manual: bool = False
    edited: bool = False
    has_saved_metadata: bool = False
    has_draft: bool = False


def record_from_image(
    image_path: Path, metadata_location: str = "caption_json"
) -> ImageRecord:
    txt_path = txt_path_for_image(image_path)
    metadata_path = metadata_path_for_image(image_path, metadata_location)
    state = _load_record_state(image_path, metadata_location)
    edited = state.tag_manual or state.nl_manual
    return ImageRecord(
        image_path=str(image_path),
        txt_path=str(txt_path),
        metadata_path=str(metadata_path),
        file_name=image_path.name,
        tags=state.parts.tags,
        tag_details=state.tag_details,
        nl=state.parts.nl,
        tag_status=_status_for_content(bool(state.parts.tags), state.tag_manual),
        nl_status=_status_for_content(bool(state.parts.nl), state.nl_manual),
        tag_manual=state.tag_manual,
        nl_manual=state.nl_manual,
        edited=edited,
        saved=bool(txt_path.exists()) or state.has_saved_metadata,
        dirty=state.has_draft,
    )


def scan_dataset(
    folder: str | Path, metadata_location: str = "caption_json"
) -> list[ImageRecord]:
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

    if tag_changed:
        record.tag_manual = True
        record.tag_details = []
        record.tag_status = _status_for_content(bool(record.tags), record.tag_manual)
    elif not record.tags:
        record.tag_status = EMPTY
        record.tag_details = []
    elif record.tag_status != ERROR:
        record.tag_status = _status_for_content(True, record.tag_manual)

    if nl_changed:
        record.nl_manual = True
        record.nl_status = _status_for_content(bool(record.nl), record.nl_manual)
    elif not record.nl:
        record.nl_status = EMPTY
    elif record.nl_status != ERROR:
        record.nl_status = _status_for_content(True, record.nl_manual)

    if tag_changed or nl_changed:
        record.saved = False
        record.dirty = True
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
    record.tag_manual = False
    record.tag_status = _status_for_content(bool(record.tags), record.tag_manual)
    record.saved = False
    record.dirty = True
    record.error = ""
    record.edited = _has_manual_edits(record)
    return record


def set_generated_nl(record: ImageRecord, nl: str) -> ImageRecord:
    record.nl = (nl or "").strip()
    record.nl_manual = False
    record.nl_status = _status_for_content(bool(record.nl), record.nl_manual)
    record.saved = False
    record.dirty = True
    record.error = ""
    record.edited = _has_manual_edits(record)
    return record


def set_error(
    record: ImageRecord, message: str, task: str | None = None
) -> ImageRecord:
    record.error = message
    if task == "tag":
        record.tag_status = ERROR
    elif task == "nl":
        record.nl_status = ERROR
    return record


def save_record(
    record: ImageRecord,
    metadata_location: str = "caption_json",
    *,
    save_txt: bool = True,
    save_metadata_json: bool = True,
    joiner: str = ". ",
) -> ImageRecord:
    if not save_txt and not save_metadata_json:
        raise ValueError("save_txt 和 save_metadata_json 不能同时为 false")
    image_path = Path(record.image_path)
    _backfill_manual_flags(record)
    record.txt_path = str(txt_path_for_image(image_path))
    record.metadata_path = str(metadata_path_for_image(image_path, metadata_location))
    final_caption = join_caption(record.tags, record.nl, joiner=joiner)
    if save_txt:
        write_text(Path(record.txt_path), final_caption)
    if save_metadata_json:
        write_json(Path(record.metadata_path), _metadata_payload(record, final_caption))
    record.saved = True
    record.dirty = False
    record.error = ""
    return record


def save_record_draft(
    record: ImageRecord,
    metadata_location: str = "caption_json",
    *,
    joiner: str = ". ",
) -> ImageRecord:
    image_path = Path(record.image_path)
    if not image_path.exists():
        return record
    metadata_path = _record_metadata_path(record, metadata_location)
    payload = read_json(metadata_path) or {}
    payload["image"] = image_path.name
    payload["draft"] = _draft_payload(record, joiner)
    write_json(metadata_path, payload)
    record.metadata_path = str(metadata_path)
    return record


def clear_record_draft(
    record: ImageRecord,
    metadata_location: str = "caption_json",
) -> None:
    if not Path(record.image_path).exists():
        return
    metadata_path = _record_metadata_path(record, metadata_location)
    payload = read_json(metadata_path)
    if not payload or "draft" not in payload:
        return
    payload = dict(payload)
    payload.pop("draft", None)
    if _has_saved_metadata(payload):
        write_json(metadata_path, payload)
        return
    try:
        metadata_path.unlink()
    except FileNotFoundError:
        return


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


def _load_record_state(
    image_path: Path, metadata_location: str
) -> PersistedRecordState:
    for candidate in metadata_candidates(image_path, metadata_location):
        metadata = read_json(candidate)
        if metadata:
            return _state_from_metadata(metadata)
    return PersistedRecordState(
        parts=parse_caption_text(read_text(txt_path_for_image(image_path)))
    )


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


def _tag_details_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tags = metadata.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    tag_details: list[dict[str, Any]] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tag")
        if not name:
            continue
        tag_details.append(
            {
                "name": str(name),
                "score": item.get("score"),
                "source": item.get("source", "manual"),
            }
        )
    return tag_details


def _state_from_metadata(metadata: dict[str, Any]) -> PersistedRecordState:
    parts = _parts_from_metadata(metadata)
    tag_details = _tag_details_from_metadata(metadata)
    raw_tags = metadata.get("tags", [])
    raw_nl = metadata.get("nl", {})
    edited = bool(metadata.get("edited", False))

    if "tag_manual" in metadata:
        tag_manual = bool(metadata.get("tag_manual"))
    else:
        tag_manual = any(
            isinstance(item, dict)
            and str(item.get("source", "")).strip().lower() == "manual"
            for item in (raw_tags if isinstance(raw_tags, list) else [])
        ) or (edited and bool(parts.tags))

    if "nl_manual" in metadata:
        nl_manual = bool(metadata.get("nl_manual"))
    elif isinstance(raw_nl, dict):
        nl_manual = str(raw_nl.get("source", "")).strip().lower() == "manual" or (
            edited and bool(parts.nl)
        )
    else:
        nl_manual = edited and bool(parts.nl)

    state = PersistedRecordState(
        parts=parts,
        tag_details=tag_details,
        tag_manual=tag_manual,
        nl_manual=nl_manual,
        edited=bool(tag_manual or nl_manual),
        has_saved_metadata=_has_saved_metadata(metadata),
    )
    draft = metadata.get("draft")
    if isinstance(draft, dict):
        return _overlay_draft_state(state, draft)
    return state


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
        "tag_manual": record.tag_manual,
        "nl_manual": record.nl_manual,
        "edited": record.edited,
    }


def _draft_payload(record: ImageRecord, joiner: str) -> dict[str, Any]:
    payload = _metadata_payload(
        record, join_caption(record.tags, record.nl, joiner=joiner)
    )
    payload["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return payload


def _overlay_draft_state(
    baseline: PersistedRecordState, draft: dict[str, Any]
) -> PersistedRecordState:
    draft_state = _state_from_metadata_without_draft(draft)
    return PersistedRecordState(
        parts=draft_state.parts,
        tag_details=draft_state.tag_details,
        tag_manual=draft_state.tag_manual,
        nl_manual=draft_state.nl_manual,
        edited=draft_state.edited,
        has_saved_metadata=baseline.has_saved_metadata,
        has_draft=True,
    )


def _state_from_metadata_without_draft(
    metadata: dict[str, Any],
) -> PersistedRecordState:
    parts = _parts_from_metadata(metadata)
    tag_details = _tag_details_from_metadata(metadata)
    raw_tags = metadata.get("tags", [])
    raw_nl = metadata.get("nl", {})
    edited = bool(metadata.get("edited", False))

    if "tag_manual" in metadata:
        tag_manual = bool(metadata.get("tag_manual"))
    else:
        tag_manual = any(
            isinstance(item, dict)
            and str(item.get("source", "")).strip().lower() == "manual"
            for item in (raw_tags if isinstance(raw_tags, list) else [])
        ) or (edited and bool(parts.tags))

    if "nl_manual" in metadata:
        nl_manual = bool(metadata.get("nl_manual"))
    elif isinstance(raw_nl, dict):
        nl_manual = str(raw_nl.get("source", "")).strip().lower() == "manual" or (
            edited and bool(parts.nl)
        )
    else:
        nl_manual = edited and bool(parts.nl)

    return PersistedRecordState(
        parts=parts,
        tag_details=tag_details,
        tag_manual=tag_manual,
        nl_manual=nl_manual,
        edited=bool(tag_manual or nl_manual),
    )


def _record_metadata_path(record: ImageRecord, metadata_location: str) -> Path:
    metadata_path = Path(record.metadata_path)
    if metadata_path.name:
        return metadata_path
    return metadata_path_for_image(Path(record.image_path), metadata_location)


def _has_saved_metadata(metadata: dict[str, Any]) -> bool:
    return any(
        key in metadata
        for key in ("tags", "nl", "final_caption", "tag_manual", "nl_manual", "edited")
    )


def _has_manual_edits(record: ImageRecord) -> bool:
    return record.tag_manual or record.nl_manual


def _status_for_content(has_content: bool, manual: bool) -> str:
    if not has_content:
        return EMPTY
    return EDITED if manual else GENERATED


def _backfill_manual_flags(record: ImageRecord) -> None:
    """Keep old state payloads that only set `*_status=edited` compatible."""

    if record.tag_status == EDITED:
        record.tag_manual = True
    if record.nl_status == EDITED:
        record.nl_manual = True
    record.edited = _has_manual_edits(record)
