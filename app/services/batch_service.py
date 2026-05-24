from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
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
    updated: int = 0
    stopped: bool = False


class BatchService:
    def __init__(self, generation: GenerationService) -> None:
        self.generation = generation

    def generate(
        self,
        records: list[ImageRecord],
        options: BatchGenerateOptions,
        progress: Any = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> BatchResult:
        final_result: BatchResult | None = None
        for result in self.generate_iter(records, options, progress, should_stop):
            final_result = result
        if final_result is not None:
            return final_result

        started = time.perf_counter()
        mode = _batch_mode(options)
        logger.info(
            "Skipped batch generation: mode=%s total=0 elapsed=%.3fs",
            mode,
            time.perf_counter() - started,
        )
        return BatchResult(0, 0, "批量生成完成: 0 张，失败 0 张")

    def generate_iter(
        self,
        records: list[ImageRecord],
        options: BatchGenerateOptions,
        progress: Any = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[BatchResult]:
        started = time.perf_counter()
        total = len(records)
        mode = _batch_mode(options)
        if not records:
            logger.info(
                "Skipped batch generation: mode=%s total=0 elapsed=%.3fs",
                mode,
                time.perf_counter() - started,
            )
            yield BatchResult(0, 0, "批量生成完成: 0 张，失败 0 张")
            return
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
            yield from self._generate_both_in_two_phases(
                eligible_records,
                total,
                options,
                progress,
                should_stop,
            )
            return

        errors = 0
        updated = 0
        for index, record in enumerate(records):
            if _should_stop(should_stop):
                result = BatchResult(
                    total,
                    errors,
                    _stopped_message(updated, total, errors),
                    updated=updated,
                    stopped=True,
                )
                logger.info(
                    "Stopped batch generation: mode=%s total=%d eligible=%d skipped=%d updated=%d errors=%d elapsed=%.3fs",
                    mode,
                    total,
                    len(eligible_records),
                    skipped,
                    updated,
                    errors,
                    time.perf_counter() - started,
                )
                yield result
                return
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
                else:
                    updated += 1
                yield BatchResult(
                    total,
                    errors,
                    _progress_message("Tag", record.file_name, updated, total, errors),
                    updated=updated,
                )
                continue

            if options.nl_model_display:
                result = self.generation.generate_nl(
                    record,
                    options.nl_model_display,
                    options.nl_request,
                )
                if not result.ok:
                    errors += 1
                else:
                    updated += 1
                yield BatchResult(
                    total,
                    errors,
                    _progress_message("NL", record.file_name, updated, total, errors),
                    updated=updated,
                )

        result = BatchResult(
            total,
            errors,
            f"批量生成完成: {total} 张，失败 {errors} 张",
            updated=updated,
        )
        logger.info(
            "Completed batch generation: mode=%s total=%d eligible=%d skipped=%d updated=%d errors=%d elapsed=%.3fs",
            mode,
            total,
            len(eligible_records),
            skipped,
            updated,
            errors,
            time.perf_counter() - started,
        )
        yield result

    def _generate_both_in_two_phases(
        self,
        records: list[ImageRecord],
        total: int,
        options: BatchGenerateOptions,
        progress: Any = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[BatchResult]:
        started = time.perf_counter()
        errors = 0
        updated_records: set[str] = set()
        tagged_records: list[ImageRecord] = []

        for index, record in enumerate(records):
            if _should_stop(should_stop):
                result = BatchResult(
                    total,
                    errors,
                    _stopped_message(len(updated_records), total, errors),
                    updated=len(updated_records),
                    stopped=True,
                )
                logger.info(
                    "Stopped batch generation: mode=both total=%d tagged=%d updated=%d errors=%d elapsed=%.3fs",
                    total,
                    len(tagged_records),
                    len(updated_records),
                    errors,
                    time.perf_counter() - started,
                )
                yield result
                return
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
            else:
                tagged_records.append(record)
                updated_records.add(record.image_path)
            yield BatchResult(
                total,
                errors,
                _progress_message(
                    "Tag",
                    record.file_name,
                    len(updated_records),
                    total,
                    errors,
                ),
                updated=len(updated_records),
            )

        logger.info(
            "Switching batch generation to NL phase: total=%d tagged=%d",
            total,
            len(tagged_records),
        )
        self.generation.registry.unload_task("tag")

        for index, record in enumerate(tagged_records):
            if _should_stop(should_stop):
                result = BatchResult(
                    total,
                    errors,
                    _stopped_message(len(updated_records), total, errors),
                    updated=len(updated_records),
                    stopped=True,
                )
                logger.info(
                    "Stopped batch generation: mode=both total=%d tagged=%d updated=%d errors=%d elapsed=%.3fs",
                    total,
                    len(tagged_records),
                    len(updated_records),
                    errors,
                    time.perf_counter() - started,
                )
                yield result
                return
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
            else:
                updated_records.add(record.image_path)
            yield BatchResult(
                total,
                errors,
                _progress_message(
                    "NL",
                    record.file_name,
                    len(updated_records),
                    total,
                    errors,
                ),
                updated=len(updated_records),
            )

        result = BatchResult(
            total,
            errors,
            f"批量生成完成: {total} 张，失败 {errors} 张",
            updated=len(updated_records),
        )
        logger.info(
            "Completed batch generation: mode=both total=%d tagged=%d updated=%d errors=%d elapsed=%.3fs",
            total,
            len(tagged_records),
            len(updated_records),
            errors,
            time.perf_counter() - started,
        )
        yield result


def _batch_mode(options: BatchGenerateOptions) -> str:
    if options.tag_model_display and options.nl_model_display:
        return "both"
    if options.tag_model_display:
        return "tag"
    if options.nl_model_display:
        return "nl"
    return "none"


def _should_stop(should_stop: Callable[[], bool] | None) -> bool:
    return bool(should_stop and should_stop())


def _progress_message(
    phase: str, file_name: str, updated: int, total: int, errors: int
) -> str:
    return f"{phase} 已更新: {file_name} ({updated}/{total})，失败 {errors} 张"


def _stopped_message(updated: int, total: int, errors: int) -> str:
    return f"批量生成已停止: 已更新 {updated}/{total} 张，失败 {errors} 张"
