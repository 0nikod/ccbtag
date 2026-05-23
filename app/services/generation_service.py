from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.dataset import (
    ImageRecord,
    save_record_draft,
    set_error,
    set_generated_nl,
    set_generated_tags,
)
from app.core.settings import AppConfig
from app.core.tag_utils import TagRuleConfig, prediction_dicts_to_tags
from app.models.registry import ModelRegistry
from app.services.tag_category_service import TagCategoryService


@dataclass(frozen=True)
class NlRequest:
    endpoint: str
    model_name: str
    api_key: str = ""
    shuffle_tags: bool | None = None
    image_resize_mode: str = "None"


@dataclass(frozen=True)
class GenerateResult:
    ok: bool
    message: str


class GenerationService:
    def __init__(
        self,
        config: AppConfig,
        registry: ModelRegistry,
        tag_rules: TagRuleConfig | None = None,
        tag_categories: TagCategoryService | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.tag_rules = tag_rules or config.tag
        self.tag_categories = tag_categories or TagCategoryService()

    def generate_tags(
        self,
        record: ImageRecord,
        tag_model_display: str,
        kept_categories: Iterable[str] | None = None,
    ) -> GenerateResult:
        try:
            model = self.registry.get_by_display("tag", tag_model_display)
            predictions = [
                item.to_dict()
                for item in model.predict(
                    record.image_path, threshold=self.tag_rules.threshold
                )
            ]
            predictions = self.tag_categories.filter_predictions(
                predictions, kept_categories
            )
            tags = prediction_dicts_to_tags(predictions, self.tag_rules)
            set_generated_tags(record, tags, predictions)
            save_record_draft(
                record,
                self.config.caption.metadata_location,
                joiner=self.config.caption.joiner,
            )
            return GenerateResult(True, f"Tag 生成完成: {record.file_name}")
        except Exception as exc:
            set_error(record, str(exc), "tag")
            return GenerateResult(False, f"Tag 生成失败: {exc}")

    def generate_nl(
        self,
        record: ImageRecord,
        nl_model_display: str,
        request: NlRequest,
    ) -> GenerateResult:
        try:
            model = self.registry.get_by_display("nl", nl_model_display)
            shuffle_tags = (
                self.config.nl.shuffle_tags
                if request.shuffle_tags is None
                else request.shuffle_tags
            )
            generated = model.predict(
                record.image_path,
                tags=record.tags,
                endpoint=request.endpoint,
                model=request.model_name,
                api_key=request.api_key,
                max_length=self.config.nl.max_length,
                language=self.config.nl.language,
                use_tags_as_context=self.config.nl.use_tags_as_context,
                shuffle_tags=shuffle_tags,
                image_resize_mode=request.image_resize_mode,
            )
            set_generated_nl(record, generated)
            save_record_draft(
                record,
                self.config.caption.metadata_location,
                joiner=self.config.caption.joiner,
            )
            return GenerateResult(True, f"NL 生成完成: {record.file_name}")
        except Exception as exc:
            set_error(record, str(exc), "nl")
            return GenerateResult(False, f"NL 生成失败: {exc}")

    def generate_both(
        self,
        record: ImageRecord,
        tag_model_display: str,
        nl_model_display: str,
        request: NlRequest,
        kept_categories: Iterable[str] | None = None,
    ) -> GenerateResult:
        tag_result = self.generate_tags(record, tag_model_display, kept_categories)
        if not tag_result.ok:
            return tag_result
        return self.generate_nl(record, nl_model_display, request)
