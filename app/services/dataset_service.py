from __future__ import annotations

from dataclasses import dataclass

from app.core.dataset import (
    ImageRecord,
    save_record_draft,
    scan_dataset,
    update_record_text,
)


@dataclass
class SelectionResult:
    records: list[ImageRecord]
    index: int
    message: str


class DatasetService:
    def __init__(self, joiner: str = ". ") -> None:
        self.joiner = joiner

    def open_folder(
        self, folder: str, metadata_location: str = "caption_json"
    ) -> SelectionResult:
        records = scan_dataset(folder, metadata_location)
        if not records:
            return SelectionResult([], 0, "未找到图片")
        return SelectionResult(records, 0, "已打开图片文件夹")

    def select_record(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
        target_index: int,
    ) -> SelectionResult:
        if not records:
            return SelectionResult([], 0, "没有可选择的图片")
        self.sync_current_form(records, current_index, tags_text, nl_text)
        index = self.clamp_index(records, target_index)
        return SelectionResult(records, index, f"当前图片: {records[index].file_name}")

    def previous_record(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
    ) -> SelectionResult:
        if not records:
            return SelectionResult([], 0, "没有图片")
        current = self.sync_current_form(records, current_index, tags_text, nl_text)
        index = max(current - 1, 0)
        return SelectionResult(records, index, f"当前图片: {records[index].file_name}")

    def next_record(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
    ) -> SelectionResult:
        if not records:
            return SelectionResult([], 0, "没有图片")
        current = self.sync_current_form(records, current_index, tags_text, nl_text)
        index = min(current + 1, len(records) - 1)
        return SelectionResult(records, index, f"当前图片: {records[index].file_name}")

    def sync_current_form(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
    ) -> int:
        index = self.clamp_index(records, current_index)
        update_record_text(records[index], tags_text, nl_text)
        if records[index].dirty:
            save_record_draft(records[index], joiner=self.joiner)
        return index

    def clamp_index(self, records: list[ImageRecord], current_index: int | None) -> int:
        if not records:
            return 0
        return min(max(current_index or 0, 0), len(records) - 1)
