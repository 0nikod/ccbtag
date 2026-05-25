from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
from typing import Any

import gradio as gr

from app.core.caption import join_caption, split_tag_text, tag_text
from app.core.dataset import deserialize_records, serialize_records, table_rows
from app.core.settings import AppConfig
from app.core.tag_utils import apply_tag_rules
from app.services import (
    AppServices,
    BatchGenerateOptions,
    NlRequest,
    create_app_services,
)
from app.services.tag_category_service import (
    DEFAULT_KEPT_TAG_CATEGORIES,
    normalize_kept_tag_categories,
    tag_category_choices as available_tag_category_choices,
)
from app.ui import presenter


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"

METADATA_LOCATION_LABELS = {
    "caption_json": "caption_json",
    "same_folder": "同目录",
}
METADATA_LOCATION_VALUES = {
    label: value for value, label in METADATA_LOCATION_LABELS.items()
}

SERVICES = create_app_services(CONFIG_ROOT)
_BATCH_STOP_LOCK = Lock()
_BATCH_STOP_EVENTS: dict[str, Event] = {}


def set_services_for_test(services: AppServices) -> None:
    global SERVICES
    SERVICES = services


def app_config() -> AppConfig:
    return SERVICES.config


def tag_model_choices() -> list[str]:
    return SERVICES.registry.display_choices("tag")


def nl_model_choices() -> list[str]:
    return SERVICES.registry.display_choices("nl")


def metadata_location_value(label: str) -> str:
    return METADATA_LOCATION_VALUES.get(label, "caption_json")


def metadata_location_label(value: str) -> str:
    return METADATA_LOCATION_LABELS.get(value, "caption_json")


def metadata_location_choices() -> list[str]:
    return [
        METADATA_LOCATION_LABELS["caption_json"],
        METADATA_LOCATION_LABELS["same_folder"],
    ]


def default_metadata_location() -> str:
    return metadata_location_label(SERVICES.config.ui.metadata_location)


def default_nl_endpoint() -> str:
    configs = SERVICES.registry.list_captioners()
    if configs:
        return str(
            configs[0].extras.get("default_endpoint", SERVICES.config.ui.nl_endpoint)
        )
    return SERVICES.config.ui.nl_endpoint


def default_nl_model_name() -> str:
    configs = SERVICES.registry.list_captioners()
    if configs:
        return str(
            configs[0].extras.get(
                "default_model_name", SERVICES.config.ui.nl_model_name
            )
        )
    return SERVICES.config.ui.nl_model_name


def default_nl_api_key() -> str:
    return SERVICES.config.ui.nl_api_key


def default_nl_image_resize_mode() -> str:
    return getattr(SERVICES.config.ui, "nl_image_resize_mode", "None")


def default_batch_txt_sentence_keywords() -> str:
    return ", ".join(SERVICES.config.ui.batch_txt_sentence_keywords)


def default_batch_txt_fragment_keywords() -> str:
    return ", ".join(SERVICES.config.ui.batch_txt_fragment_keywords)


def default_shuffle_tags() -> bool:
    return SERVICES.config.nl.shuffle_tags


def tag_category_choices() -> list[str]:
    return available_tag_category_choices()


def default_kept_tag_categories() -> list[str]:
    return list(DEFAULT_KEPT_TAG_CATEGORIES)


def open_folder(folder: str, metadata_location_label: str) -> presenter.DatasetPayload:
    try:
        result = SERVICES.dataset.open_folder(
            folder, metadata_location_value(metadata_location_label)
        )
    except Exception as exc:
        return presenter.empty_dataset_payload(f"打开失败: {exc}")
    if not result.records:
        return presenter.empty_dataset_payload(result.message)
    return presenter.dataset_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def select_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    event: gr.SelectData,
) -> presenter.EditorPayload:
    return _select_record(
        records_data, current_index, tags_text, nl_text, _event_index(event)
    )


def _select_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    target_index: int,
) -> presenter.EditorPayload:
    records = deserialize_records(records_data)
    result = SERVICES.dataset.select_record(
        records, current_index, tags_text, nl_text, target_index
    )
    if not result.records:
        return presenter.empty_editor_payload(result.message)
    return presenter.editor_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def previous_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> presenter.EditorPayload:
    records = deserialize_records(records_data)
    result = SERVICES.dataset.previous_record(
        records, current_index, tags_text, nl_text
    )
    if not result.records:
        return presenter.empty_editor_payload(result.message)
    return presenter.editor_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def next_record(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> presenter.EditorPayload:
    records = deserialize_records(records_data)
    result = SERVICES.dataset.next_record(records, current_index, tags_text, nl_text)
    if not result.records:
        return presenter.empty_editor_payload(result.message)
    return presenter.editor_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def preview_caption(tags_text: str, nl_text: str) -> str:
    return join_caption(tags_text, nl_text, joiner=SERVICES.config.caption.joiner)


def autosave_current(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
) -> tuple[list[dict[str, Any]], list[list[str]], str, int]:
    records = deserialize_records(records_data)
    if not records:
        return [], [], preview_caption(tags_text, nl_text), 0
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    return (
        serialize_records(records),
        table_rows(records),
        preview_caption(tags_text, nl_text),
        index,
    )


def apply_rules_to_current(tags_text: str, nl_text: str) -> tuple[str, str]:
    cleaned = tag_text(
        apply_tag_rules(split_tag_text(tags_text), SERVICES.config.tag),
        SERVICES.config.tag.separator,
    )
    return cleaned, join_caption(
        cleaned, nl_text, joiner=SERVICES.config.caption.joiner
    )


def clear_tags(tags_text: str, nl_text: str) -> tuple[str, str]:
    return "", join_caption("", nl_text, joiner=SERVICES.config.caption.joiner)


def clear_nl(tags_text: str, nl_text: str) -> tuple[str, str]:
    return "", join_caption(tags_text, "", joiner=SERVICES.config.caption.joiner)


def generate_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tag_model_display: str,
    kept_categories: list[str] | None,
    tags_text: str,
    nl_text: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有当前图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    result = SERVICES.generation.generate_tags(
        records[index],
        tag_model_display,
        normalize_kept_tag_categories(kept_categories),
    )
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def generate_nl(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    tags_text: str,
    nl_text: str,
    shuffle_tags: bool | None = None,
    image_resize_mode: str = "None",
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有当前图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    request = NlRequest(
        nl_endpoint, nl_model_name, nl_api_key, shuffle_tags, image_resize_mode
    )
    result = SERVICES.generation.generate_nl(records[index], nl_model_display, request)
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def generate_tag_and_nl(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tag_model_display: str,
    kept_categories: list[str] | None,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    tags_text: str,
    nl_text: str,
    shuffle_tags: bool | None = None,
    image_resize_mode: str = "None",
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有当前图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    request = NlRequest(
        nl_endpoint, nl_model_name, nl_api_key, shuffle_tags, image_resize_mode
    )
    result = SERVICES.generation.generate_both(
        records[index],
        tag_model_display,
        nl_model_display,
        request,
        normalize_kept_tag_categories(kept_categories),
    )
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def save_current(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    metadata_location_label: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    result = SERVICES.save.save_current(
        records,
        current_index,
        tags_text,
        nl_text,
        metadata_location_value(metadata_location_label),
    )
    if not result.records:
        return presenter.empty_dataset_payload(result.message)
    return presenter.dataset_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def save_all(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    metadata_location_label: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    result = SERVICES.save.save_all(
        records,
        current_index,
        tags_text,
        nl_text,
        metadata_location_value(metadata_location_label),
    )
    if not result.records:
        return presenter.empty_dataset_payload(result.message)
    return presenter.dataset_payload(
        result.records, result.index, result.message, SERVICES.config.caption
    )


def batch_generate_tags(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    tag_model_display: str,
    kept_categories: list[str] | None,
    skip_edited: bool,
    progress: gr.Progress | None = None,
    request: gr.Request | None = None,
):
    records = deserialize_records(records_data)
    if not records:
        yield presenter.empty_dataset_payload("没有图片")
        return
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    yield from _stream_batch_generation(
        records,
        index,
        BatchGenerateOptions(
            tag_model_display=tag_model_display,
            kept_tag_categories=normalize_kept_tag_categories(kept_categories),
            skip_edited=skip_edited,
        ),
        progress,
        request,
    )


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
    shuffle_tags: bool | None = None,
    image_resize_mode: str = "None",
    progress: gr.Progress | None = None,
    request: gr.Request | None = None,
):
    records = deserialize_records(records_data)
    if not records:
        yield presenter.empty_dataset_payload("没有图片")
        return
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    yield from _stream_batch_generation(
        records,
        index,
        BatchGenerateOptions(
            nl_model_display=nl_model_display,
            skip_edited=skip_edited,
            nl_request=NlRequest(
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                shuffle_tags,
                image_resize_mode,
            ),
        ),
        progress,
        request,
    )


def batch_generate_both(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    tag_model_display: str,
    kept_categories: list[str] | None,
    nl_model_display: str,
    nl_endpoint: str,
    nl_model_name: str,
    nl_api_key: str,
    skip_edited: bool,
    shuffle_tags: bool | None = None,
    image_resize_mode: str = "None",
    progress: gr.Progress | None = None,
    request: gr.Request | None = None,
):
    records = deserialize_records(records_data)
    if not records:
        yield presenter.empty_dataset_payload("没有图片")
        return
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    yield from _stream_batch_generation(
        records,
        index,
        BatchGenerateOptions(
            tag_model_display=tag_model_display,
            kept_tag_categories=normalize_kept_tag_categories(kept_categories),
            nl_model_display=nl_model_display,
            skip_edited=skip_edited,
            nl_request=NlRequest(
                nl_endpoint,
                nl_model_name,
                nl_api_key,
                shuffle_tags,
                image_resize_mode,
            ),
        ),
        progress,
        request,
    )


def stop_batch_generation(request: gr.Request | None = None) -> str:
    _batch_stop_event(request).set()
    return "正在请求停止批量生成..."


def cleanup_session_state(request: gr.Request | None = None) -> None:
    session_key = _batch_session_key(request)
    with _BATCH_STOP_LOCK:
        _BATCH_STOP_EVENTS.pop(session_key, None)


def batch_delete_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    delete_text: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    result = SERVICES.tag_edit.delete_tags(records, delete_text)
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def batch_replace_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    old: str,
    new: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    result = SERVICES.tag_edit.replace_tags(records, old, new)
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def batch_add_tag(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    add_text: str,
    prepend: bool,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    result = SERVICES.tag_edit.add_tags(records, add_text, prepend)
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def batch_process_txt_rules(
    records_data: list[dict[str, Any]],
    current_index: int | None,
    tags_text: str,
    nl_text: str,
    sentence_keywords: str,
    fragment_keywords: str,
    modify_json: bool,
    metadata_location_label: str,
) -> presenter.DatasetPayload:
    records = deserialize_records(records_data)
    if not records:
        return presenter.empty_dataset_payload("没有图片")
    index = SERVICES.dataset.sync_current_form(
        records, current_index, tags_text, nl_text
    )
    result = SERVICES.txt_process.process_nl_rules(
        records,
        sentence_keywords,
        fragment_keywords,
        modify_json=modify_json,
        metadata_location=metadata_location_value(metadata_location_label),
    )
    return presenter.dataset_payload(
        records, index, result.message, SERVICES.config.caption
    )


def _stream_batch_generation(
    records, index: int, options: BatchGenerateOptions, progress, request
):
    stop_event = _batch_stop_event(request)
    stop_event.clear()
    try:
        for result in SERVICES.batch.generate_iter(
            records,
            options,
            progress,
            should_stop=stop_event.is_set,
        ):
            yield presenter.dataset_payload(
                records, index, result.message, SERVICES.config.caption
            )
    finally:
        stop_event.clear()


def _batch_stop_event(request: gr.Request | None) -> Event:
    session_key = _batch_session_key(request)
    with _BATCH_STOP_LOCK:
        event = _BATCH_STOP_EVENTS.get(session_key)
        if event is None:
            event = Event()
            _BATCH_STOP_EVENTS[session_key] = event
        return event


def _batch_session_key(request: gr.Request | None) -> str:
    if request and request.session_hash:
        return request.session_hash
    return "__default__"


def _event_index(event: Any) -> int:
    index = getattr(event, "index", 0)
    if isinstance(index, (list, tuple)):
        return int(index[0])
    return int(index or 0)
