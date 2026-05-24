from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.caption import tag_text
from app.core.dataset import ImageRecord, save_record_draft, update_record_text
from app.core.tag_utils import TagRuleConfig, add_tags, delete_tags, replace_tags


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagEditResult:
    changed: int
    message: str


class TagEditService:
    def __init__(self, tag_rules: TagRuleConfig, joiner: str = ". ") -> None:
        self.tag_rules = tag_rules
        self.joiner = joiner

    def delete_tags(
        self,
        records: list[ImageRecord],
        delete_text: str,
    ) -> TagEditResult:
        started = time.perf_counter()
        logger.info(
            "Batch delete tags: total=%d delete_text=%s",
            len(records),
            delete_text,
        )
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = delete_tags(tag_text(record.tags), delete_text, self.tag_rules)
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                save_record_draft(record, joiner=self.joiner)
                changed += 1
        logger.info(
            "Batch delete tags completed: total=%d changed=%d elapsed=%.3fs",
            len(records),
            changed,
            time.perf_counter() - started,
        )
        return TagEditResult(changed, f"批量删除 Tag 完成: {changed} 张")

    def replace_tags(
        self,
        records: list[ImageRecord],
        old: str,
        new: str,
    ) -> TagEditResult:
        started = time.perf_counter()
        logger.info(
            "Batch replace tags: total=%d old=%s new=%s",
            len(records),
            old,
            new,
        )
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = replace_tags(tag_text(record.tags), old, new, self.tag_rules)
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                save_record_draft(record, joiner=self.joiner)
                changed += 1
        logger.info(
            "Batch replace tags completed: total=%d changed=%d elapsed=%.3fs",
            len(records),
            changed,
            time.perf_counter() - started,
        )
        return TagEditResult(changed, f"批量替换 Tag 完成: {changed} 张")

    def add_tags(
        self,
        records: list[ImageRecord],
        add_text: str,
        prepend: bool = False,
    ) -> TagEditResult:
        started = time.perf_counter()
        logger.info(
            "Batch add tags: total=%d add_text=%s prepend=%s",
            len(records),
            add_text,
            prepend,
        )
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = add_tags(
                tag_text(record.tags), add_text, self.tag_rules, prepend=prepend
            )
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                save_record_draft(record, joiner=self.joiner)
                changed += 1
        logger.info(
            "Batch add tags completed: total=%d changed=%d elapsed=%.3fs",
            len(records),
            changed,
            time.perf_counter() - started,
        )
        return TagEditResult(changed, f"批量添加 Tag 完成: {changed} 张")
