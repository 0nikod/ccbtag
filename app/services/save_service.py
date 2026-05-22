from __future__ import annotations

from app.core.dataset import ImageRecord, clear_record_draft, save_record
from app.core.settings import AppConfig
from app.services.dataset_service import DatasetService, SelectionResult


class SaveService:
    def __init__(self, config: AppConfig, dataset: DatasetService) -> None:
        self.config = config
        self.dataset = dataset

    def save_current(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
        metadata_location: str | None = None,
    ) -> SelectionResult:
        if not records:
            return SelectionResult([], 0, "没有当前图片")
        index = self.dataset.sync_current_form(
            records, current_index, tags_text, nl_text
        )
        record = records[index]
        self._save(record, metadata_location)
        return SelectionResult(records, index, f"已保存: {record.file_name}")

    def save_all(
        self,
        records: list[ImageRecord],
        current_index: int | None,
        tags_text: str,
        nl_text: str,
        metadata_location: str | None = None,
    ) -> SelectionResult:
        if not records:
            return SelectionResult([], 0, "没有图片")
        index = self.dataset.sync_current_form(
            records, current_index, tags_text, nl_text
        )
        for record in records:
            self._save(record, metadata_location)
        return SelectionResult(records, index, f"已保存全部: {len(records)} 张")

    def _save(self, record: ImageRecord, metadata_location: str | None) -> None:
        location = metadata_location or self.config.caption.metadata_location
        save_record(
            record,
            location,
            save_txt=self.config.caption.save_txt,
            save_metadata_json=self.config.caption.save_metadata_json,
            joiner=self.config.caption.joiner,
        )
        clear_record_draft(record, location)
