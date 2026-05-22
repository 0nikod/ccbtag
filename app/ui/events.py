from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from app.core.caption import join_caption, split_tag_text, tag_text
from app.core.dataset import (
    ImageRecord,
    deserialize_records,
    save_record,
    scan_dataset,
    serialize_records,
    set_error,
    set_generated_nl,
    set_generated_tags,
    table_rows,
    update_record_text,
)
from app.core.preview import image_preview_html
from app.core.settings import AppConfig, load_app_config
from app.core.tag_utils import (
    add_tags,
    apply_tag_rules,
    delete_tags,
    prediction_dicts_to_tags,
    replace_tags,
)
from app.models.registry import default_registry


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def load_rules() -> dict[str, Any]:
    with (CONFIG_ROOT / "rules.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


RULES = load_rules()
APP_CONFIG = load_app_config(CONFIG_ROOT)
TAG_RULES = APP_CONFIG.tag
NL_RULES = APP_CONFIG.nl
CAPTION_RULES = APP_CONFIG.caption
UI_DEFAULTS = APP_CONFIG.ui
REGISTRY = default_registry()


def app_config() -> AppConfig:
    return APP_CONFIG


def tag_model_choices() -> list[str]:
    return REGISTRY.display_choices("tag")


def nl_model_choices() -> list[str]:
    return REGISTRY.display_choices("nl")


def metadata_location_value(label: str) -> str:
    return "same_folder" if label == "同目录" else "caption_json"


def default_metadata_location() -> str:
    return CAPTION_RULES.metadata_location


def default_nl_endpoint() -> str:
    configs = REGISTRY.list_captioners()
    if configs:
        return str(configs[0].extras.get("default_endpoint", UI_DEFAULTS.nl_endpoint))
    return UI_DEFAULTS.nl_endpoint


def default_nl_model_name() -> str:
    configs = REGISTRY.list_captioners()
    if configs:
        return str(
            configs[0].extras.get("default_model_name", UI_DEFAULTS.nl_model_name)
        )
    return UI_DEFAULTS.nl_model_name


def default_nl_api_key() -> str:
    return UI_DEFAULTS.nl_api_key


def default_shuffle_tags() -> bool:
    return UI_DEFAULTS.shuffle_tags


def open_folder(folder: str, metadata_location_label: str) -> tuple[Any, ...]:
    try:
        records = scan_dataset(folder, metadata_location_value(metadata_location_label))
    except Exception as exc:
        return [], [], image_preview_html(None), "", "", "", 0, f"打开失败: {exc}"
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "未找到图片"
    return _selection_payload(records, 0, "已打开图片文件夹")


def select_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    event: gr.SelectData,
) -> tuple[Any, ...]:
    return _select_record(
        records_data, current_index, tags_text, nl_text, _event_index(event)
    )


def _select_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    target_index: int,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], image_preview_html(None), "", "", "", 0, "没有可选择的图片"
    _sync_current_form(records, current_index, tags_text, nl_text)
    index = _clamp_index(records, target_index)
    return _record_payload(records, index, f"当前图片: {records[index].file_name}")


def previous_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], image_preview_html(None), "", "", "", 0, "没有图片"
    current = _sync_current_form(records, current_index, tags_text, nl_text)
    index = max(current - 1, 0)
    return _record_payload(records, index, f"当前图片: {records[index].file_name}")


def next_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], image_preview_html(None), "", "", "", 0, "没有图片"
    current = _sync_current_form(records, current_index, tags_text, nl_text)
    index = min(current + 1, len(records) - 1)
    return _record_payload(records, index, f"当前图片: {records[index].file_name}")


def preview_caption(tags_text: str, nl_text: str) -> str:
    return join_caption(tags_text, nl_text, joiner=CAPTION_RULES.joiner)


def apply_rules_to_current(tags_text: str, nl_text: str) -> tuple[str, str]:
    cleaned = tag_text(
        apply_tag_rules(split_tag_text(tags_text), TAG_RULES), TAG_RULES.separator
    )
    return cleaned, join_caption(cleaned, nl_text, joiner=CAPTION_RULES.joiner)


def clear_tags(tags_text: str, nl_text: str) -> tuple[str, str]:
    return "", join_caption("", nl_text, joiner=CAPTION_RULES.joiner)


def clear_nl(tags_text: str, nl_text: str) -> tuple[str, str]:
    return "", join_caption(tags_text, "", joiner=CAPTION_RULES.joiner)


def generate_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tag_model_display: str,
    tags_text: str,
    nl_text: str,
) -> tuple[Any, ...]:
    records, record, index = _current_record(records_data, current_index)
    if record is None:
        return [], [], image_preview_html(None), "", "", "", 0, "没有当前图片"
    update_record_text(record, tags_text, nl_text)
    try:
        model = REGISTRY.get_by_display("tag", tag_model_display)
        predictions = [
            item.to_dict()
            for item in model.predict(record.image_path, threshold=TAG_RULES.threshold)
        ]
        tags = prediction_dicts_to_tags(predictions, TAG_RULES)
        set_generated_tags(record, tags, predictions)
        message = f"Tag 生成完成: {record.file_name}"
    except Exception as exc:
        set_error(record, str(exc), "tag")
        message = f"Tag 生成失败: {exc}"
    records[index] = record
    return _selection_payload(records, index, message)


def generate_nl(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    tags_text: str,
    nl_text: str,
    shuffle_tags: bool = True,
) -> tuple[Any, ...]:
    records, record, index = _current_record(records_data, current_index)
    if record is None:
        return [], [], image_preview_html(None), "", "", "", 0, "没有当前图片"
    update_record_text(record, tags_text, nl_text)
    try:
        model = REGISTRY.get_by_display("nl", nl_model_display)
        generated = model.predict(
            record.image_path,
            tags=record.tags,
            endpoint=nl_endpoint,
            model=nl_model_name,
            api_key=nl_api_key,
            max_length=NL_RULES.max_length,
            language=NL_RULES.language,
            use_tags_as_context=NL_RULES.use_tags_as_context,
            shuffle_tags=shuffle_tags,
        )
        set_generated_nl(record, generated)
        message = f"NL 生成完成: {record.file_name}"
    except Exception as exc:
        set_error(record, str(exc), "nl")
        message = f"NL 生成失败: {exc}"
    records[index] = record
    return _selection_payload(records, index, message)


def generate_tag_and_nl(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tag_model_display: str,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    tags_text: str,
    nl_text: str,
    shuffle_tags: bool = True,
) -> tuple[Any, ...]:
    payload = generate_tag(
        records_data, current_index, tag_model_display, tags_text, nl_text
    )
    updated_records = payload[0]
    updated_tags = payload[3]
    updated_nl = payload[4]
    return generate_nl(
        updated_records,
        current_index,
        nl_model_display,
        nl_endpoint,
        nl_model_name,
        nl_api_key,
        updated_tags,
        updated_nl,
        shuffle_tags,
    )


def save_current(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    metadata_location_label: str,
) -> tuple[Any, ...]:
    records, record, index = _current_record(records_data, current_index)
    if record is None:
        return [], [], image_preview_html(None), "", "", "", 0, "没有当前图片"
    update_record_text(record, tags_text, nl_text)
    save_record(record, metadata_location_value(metadata_location_label))
    records[index] = record
    return _selection_payload(records, index, f"已保存: {record.file_name}")


def save_all(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    metadata_location_label: str,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    for record in records:
        save_record(record, metadata_location_value(metadata_location_label))
    return _selection_payload(records, index, f"已保存全部: {len(records)} 张")


def batch_generate_tags(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    tag_model_display: str,
    skip_edited: bool,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    message = _batch_generate(records, tag_model_display, None, skip_edited, progress)
    return _selection_payload(records, index, message)


def batch_generate_nl(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    skip_edited: bool,
    shuffle_tags: bool = True,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    message = _batch_generate(
        records,
        None,
        nl_model_display,
        skip_edited,
        progress,
        nl_endpoint,
        nl_model_name,
        nl_api_key,
        shuffle_tags,
    )
    return _selection_payload(records, index, message)


def batch_generate_both(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    tag_model_display: str,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    skip_edited: bool,
    shuffle_tags: bool = True,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    message = _batch_generate(
        records,
        tag_model_display,
        nl_model_display,
        skip_edited,
        progress,
        nl_endpoint,
        nl_model_name,
        nl_api_key,
        shuffle_tags,
    )
    return _selection_payload(records, index, message)


def batch_delete_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    delete_text: str,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    for record in records:
        next_tags = delete_tags(tag_text(record.tags), delete_text, TAG_RULES)
        update_record_text(record, next_tags, record.nl)
    return _selection_payload(records, index, "批量删除 Tag 完成")


def batch_replace_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    old: str,
    new: str,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    for record in records:
        next_tags = replace_tags(tag_text(record.tags), old, new, TAG_RULES)
        update_record_text(record, next_tags, record.nl)
    return _selection_payload(records, index, "批量替换 Tag 完成")


def batch_add_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    add_text: str,
    prepend: bool,
) -> tuple[Any, ...]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], image_preview_html(None), "", "", "", 0, "没有图片"
    index = _sync_current_form(records, current_index, tags_text, nl_text)
    for record in records:
        next_tags = add_tags(
            tag_text(record.tags), add_text, TAG_RULES, prepend=prepend
        )
        update_record_text(record, next_tags, record.nl)
    return _selection_payload(records, index, "批量添加 Tag 完成")


def _batch_generate(
    records: list[ImageRecord],
    tag_model_display: str | None,
    nl_model_display: str | None,
    skip_edited: bool,
    progress: Any,
    nl_endpoint: str = "",
    nl_model_name: str = "",
    nl_api_key: str = "",
    shuffle_tags: bool = True,
) -> str:
    total = len(records)
    errors = 0
    tag_model = (
        REGISTRY.get_by_display("tag", tag_model_display) if tag_model_display else None
    )
    nl_model = (
        REGISTRY.get_by_display("nl", nl_model_display) if nl_model_display else None
    )
    for index, record in enumerate(records):
        if progress:
            progress(
                (index + 1) / total, desc=f"{index + 1}/{total} {record.file_name}"
            )
        if skip_edited and record.edited:
            continue
        try:
            if tag_model:
                predictions = [
                    item.to_dict()
                    for item in tag_model.predict(
                        record.image_path, threshold=TAG_RULES.threshold
                    )
                ]
                tags = prediction_dicts_to_tags(predictions, TAG_RULES)
                set_generated_tags(record, tags, predictions)
            if nl_model:
                generated = nl_model.predict(
                    record.image_path,
                    tags=record.tags,
                    endpoint=nl_endpoint,
                    model=nl_model_name,
                    api_key=nl_api_key,
                    max_length=NL_RULES.max_length,
                    language=NL_RULES.language,
                    use_tags_as_context=NL_RULES.use_tags_as_context,
                    shuffle_tags=shuffle_tags,
                )
                set_generated_nl(record, generated)
        except Exception as exc:
            errors += 1
            set_error(record, str(exc))
    return f"批量生成完成: {total} 张，失败 {errors} 张"


def _current_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
) -> tuple[list[ImageRecord], ImageRecord | None, int]:
    records = deserialize_records(records_data)
    if not records:
        return records, None, 0
    index = _clamp_index(records, current_index)
    return records, records[index], index


def _selection_payload(
    records: list[ImageRecord], index: int, message: str
) -> tuple[Any, ...]:
    return (
        serialize_records(records),
        table_rows(records),
        image_preview_html(records[index].image_path),
        tag_text(records[index].tags),
        records[index].nl,
        records[index].final_caption,
        index,
        message,
    )


def _record_payload(
    records: list[ImageRecord], index: int, message: str
) -> tuple[Any, ...]:
    record = records[index]
    return (
        serialize_records(records),
        image_preview_html(record.image_path),
        tag_text(record.tags),
        record.nl,
        record.final_caption,
        index,
        message,
    )


def _event_index(event: Any) -> int:
    index = getattr(event, "index", 0)
    if isinstance(index, (list, tuple)):
        return int(index[0])
    return int(index or 0)


def _clamp_index(records: list[ImageRecord], current_index: int | None) -> int:
    return min(max(current_index or 0, 0), len(records) - 1)


def _sync_current_form(
    records: list[ImageRecord],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> int:
    """Commit the visible editor fields before navigation or batch actions."""

    index = _clamp_index(records, current_index)
    update_record_text(records[index], tags_text, nl_text)
    return index
