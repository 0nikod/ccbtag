from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.dataset import ImageRecord
from app.services.generation_service import GenerationService, NlRequest


logger = logging.getLogger(__name__)


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
        started = time.perf_counter()
        total = len(records)
        mode = _batch_mode(options)
        if not records:
            logger.info(
                "Skipped batch generation: mode=%s total=0 elapsed=%.3fs",
                mode,
                time.perf_counter() - started,
            )
            return BatchResult(0, 0, "批量生成完成: 0 张，失败 0 张")
        if options.nl_model_display and options.nl_request is None:
            logger.error("Batch generation requires nl_request: mode=%s", mode)
            raise ValueError("缺少 nl_request")

        eligible_records = [
            record for record in records if not (options.skip_edited and record.edited)
        ]
        skipped = total - len(eligible_records)
        logger.info(
            "Starting batch generation: mode=%s total=%d eligible=%d skipped=%d tag_model=%s nl_model=%s",
            mode,
            total,
            len(eligible_records),
            skipped,
            options.tag_model_display,
            options.nl_model_display,
        )

        if options.tag_model_display and options.nl_model_display:
            result = self._generate_both_in_two_phases(
                eligible_records,
                total,
                options,
                progress,
            )
            logger.info(
                "Completed batch generation: mode=%s total=%d eligible=%d skipped=%d errors=%d elapsed=%.3fs",
                mode,
                total,
                len(eligible_records),
                skipped,
                result.errors,
                time.perf_counter() - started,
            )
            return result

        errors = 0
        for index, record in enumerate(records):
            if progress:
                progress(
                    (index + 1) / total, desc=f"{index + 1}/{total} {record.file_name}"
                )
            if options.skip_edited and record.edited:
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

        logger.info(
            "Completed batch generation: mode=%s total=%d eligible=%d skipped=%d errors=%d elapsed=%.3fs",
            mode,
            total,
            len(eligible_records),
            skipped,
            errors,
            time.perf_counter() - started,
        )
        return BatchResult(total, errors, f"批量生成完成: {total} 张，失败 {errors} 张")

    def _generate_both_in_two_phases(
        self,
        records: list[ImageRecord],
        total: int,
        options: BatchGenerateOptions,
        progress: Any = None,
    ) -> BatchResult:
        errors = 0
        tagged_records: list[ImageRecord] = []

        for index, record in enumerate(records):
            if progress:
                progress(
                    (index + 1) / total,
                    desc=f"Tag {index + 1}/{total} {record.file_name}",
                )
            result = self.generation.generate_tags(
                record,
                options.tag_model_display or "",
                options.kept_tag_categories,
            )
            if not result.ok:
                errors += 1
                continue
            tagged_records.append(record)

        logger.info(
            "Switching batch generation to NL phase: total=%d tagged=%d",
            total,
            len(tagged_records),
        )
        self.generation.registry.unload_task("tag")

        for index, record in enumerate(tagged_records):
            if progress:
                progress(
                    (index + 1) / total,
                    desc=f"NL {index + 1}/{total} {record.file_name}",
                )
            result = self.generation.generate_nl(
                record,
                options.nl_model_display or "",
                options.nl_request,
            )
            if not result.ok:
                errors += 1

        return BatchResult(total, errors, f"批量生成完成: {total} 张，失败 {errors} 张")


def _batch_mode(options: BatchGenerateOptions) -> str:
    if options.tag_model_display and options.nl_model_display:
        return "both"
    if options.tag_model_display:
        return "tag"
    if options.nl_model_display:
        return "nl"
    return "none"
