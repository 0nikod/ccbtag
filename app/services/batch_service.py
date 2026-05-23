from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.dataset import ImageRecord
from app.services.generation_service import GenerationService, NlRequest


@dataclass(frozen=True)
class BatchGenerateOptions:
    tag_model_display: str | None = None
    nl_model_display: str | None = None
    skip_edited: bool = True
    nl_request: NlRequest | None = None
    kept_tag_categories: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BatchResult:
    total: int
    errors: int
    message: str


class BatchService:
    def __init__(self, generation: GenerationService) -> None:
        self.generation = generation

    def generate(
        self,
        records: list[ImageRecord],
        options: BatchGenerateOptions,
        progress: Any = None,
    ) -> BatchResult:
        total = len(records)
        if not records:
            return BatchResult(0, 0, "批量生成完成: 0 张，失败 0 张")
        if options.nl_model_display and options.nl_request is None:
            raise ValueError("缺少 nl_request")

        errors = 0
        for index, record in enumerate(records):
            if progress:
                progress(
                    (index + 1) / total, desc=f"{index + 1}/{total} {record.file_name}"
                )
            if options.skip_edited and record.edited:
                continue

            if options.tag_model_display and options.nl_model_display:
                result = self.generation.generate_tags(
                    record,
                    options.tag_model_display,
                    options.kept_tag_categories,
                )
                if not result.ok:
                    errors += 1
                    continue
                result = self.generation.generate_nl(
                    record,
                    options.nl_model_display,
                    options.nl_request,
                )
                if not result.ok:
                    errors += 1
                continue

            if options.tag_model_display:
                result = self.generation.generate_tags(
                    record,
                    options.tag_model_display,
                    options.kept_tag_categories,
                )
                if not result.ok:
                    errors += 1
                continue

            if options.nl_model_display:
                result = self.generation.generate_nl(
                    record,
                    options.nl_model_display,
                    options.nl_request,
                )
                if not result.ok:
                    errors += 1

        return BatchResult(total, errors, f"批量生成完成: {total} 张，失败 {errors} 张")
