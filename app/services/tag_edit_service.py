from __future__ import annotations

from dataclasses import dataclass

from app.core.caption import tag_text
from app.core.dataset import ImageRecord, update_record_text
from app.core.tag_utils import TagRuleConfig, add_tags, delete_tags, replace_tags


@dataclass(frozen=True)
class TagEditResult:
    changed: int
    message: str


class TagEditService:
    def __init__(self, tag_rules: TagRuleConfig) -> None:
        self.tag_rules = tag_rules

    def delete_tags(
        self,
        records: list[ImageRecord],
        delete_text: str,
    ) -> TagEditResult:
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = delete_tags(tag_text(record.tags), delete_text, self.tag_rules)
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                changed += 1
        return TagEditResult(changed, f"批量删除 Tag 完成: {changed} 张")

    def replace_tags(
        self,
        records: list[ImageRecord],
        old: str,
        new: str,
    ) -> TagEditResult:
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = replace_tags(tag_text(record.tags), old, new, self.tag_rules)
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                changed += 1
        return TagEditResult(changed, f"批量替换 Tag 完成: {changed} 张")

    def add_tags(
        self,
        records: list[ImageRecord],
        add_text: str,
        prepend: bool = False,
    ) -> TagEditResult:
        changed = 0
        for record in records:
            before = list(record.tags)
            next_tags = add_tags(
                tag_text(record.tags), add_text, self.tag_rules, prepend=prepend
            )
            update_record_text(record, next_tags, record.nl)
            if record.tags != before:
                changed += 1
        return TagEditResult(changed, f"批量添加 Tag 完成: {changed} 张")
