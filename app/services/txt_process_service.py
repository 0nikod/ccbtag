from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.caption import clean_nl_by_rules, split_tag_text, tag_text
from app.core.dataset import (
    ImageRecord,
    clear_record_draft,
    save_record,
    save_record_draft,
    update_record_text,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TxtProcessResult:
    changed: int
    message: str


class TxtProcessService:
    def __init__(self, joiner: str = ". ") -> None:
        self.joiner = joiner

    def process_nl_rules(
        self,
        records: list[ImageRecord],
        sentence_keywords: str,
        fragment_keywords: str,
        *,
        modify_json: bool = False,
        metadata_location: str = "caption_json",
    ) -> TxtProcessResult:
        started = time.perf_counter()
        logger.info(
            "Batch process txt rules: total=%d modify_json=%s metadata_location=%s",
            len(records),
            modify_json,
            metadata_location,
        )
        changed = 0
        sentence_terms = split_tag_text(sentence_keywords)
        fragment_terms = split_tag_text(fragment_keywords)

        for record in records:
            cleaned_nl = clean_nl_by_rules(record.nl, sentence_terms, fragment_terms)
            if cleaned_nl == record.nl:
                continue

            update_record_text(record, tag_text(record.tags), cleaned_nl)
            save_record(
                record,
                metadata_location,
                save_txt=True,
                save_metadata_json=modify_json,
                joiner=self.joiner,
            )
            if modify_json:
                clear_record_draft(record, metadata_location)
            else:
                save_record_draft(record, metadata_location, joiner=self.joiner)
            changed += 1

        logger.info(
            "Batch process txt rules completed: total=%d changed=%d modify_json=%s elapsed=%.3fs",
            len(records),
            changed,
            modify_json,
            time.perf_counter() - started,
        )
        return TxtProcessResult(changed, f"TXT 规则处理完成: {changed} 张")
