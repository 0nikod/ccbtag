from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from app.core.caption import split_tag_text, tag_text


@dataclass(frozen=True)
class TagRuleConfig:
    """User-visible tag cleanup rules loaded from app config."""

    threshold: float = 0.35
    max_tags: int = 80
    replace_underscore: bool = True
    sort_by_score: bool = True
    blacklist: tuple[str, ...] = ()
    replace_rules: Mapping[str, str] | None = None
    separator: str = ", "


def config_from_rules(rules: Mapping[str, object]) -> TagRuleConfig:
    tag_rules = dict(rules.get("tag", {}) if isinstance(rules, Mapping) else {})
    return TagRuleConfig(
        threshold=float(tag_rules.get("threshold", 0.35)),
        max_tags=int(tag_rules.get("max_tags", 80)),
        replace_underscore=bool(tag_rules.get("replace_underscore", True)),
        sort_by_score=bool(tag_rules.get("sort_by_score", True)),
        blacklist=tuple(str(item) for item in tag_rules.get("blacklist", [])),
        replace_rules=dict(tag_rules.get("replace_rules", {})),
        separator=str(tag_rules.get("separator", ", ")),
    )


def normalize_tag(tag: str, rules: TagRuleConfig) -> str:
    value = tag.strip()
    if rules.replace_underscore:
        value = value.replace("_", " ")
    replace_rules = rules.replace_rules or {}
    return str(replace_rules.get(value, value)).strip()


def apply_tag_rules(tags: Iterable[str], rules: TagRuleConfig) -> list[str]:
    """Apply replacements, blacklist, and de-duplication in display order."""

    blacklist = {normalize_tag(item, rules).casefold() for item in rules.blacklist}
    result: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        cleaned = normalize_tag(raw_tag, rules)
        key = cleaned.casefold()
        if not cleaned or key in blacklist or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result[: rules.max_tags]


def prediction_dicts_to_tags(
    predictions: Iterable[Mapping[str, object]],
    rules: TagRuleConfig,
) -> list[str]:
    """Filter model output dictionaries into the editable tag list."""

    rows = list(predictions)
    if rules.sort_by_score:
        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    tags: list[str] = []
    for row in rows:
        score = float(row.get("score", 0.0))
        if score < rules.threshold:
            continue
        tags.append(str(row.get("tag", "")).strip())
    return apply_tag_rules(tags, rules)


def delete_tags(tags_text: str, delete_text: str, rules: TagRuleConfig) -> str:
    delete_set = {
        normalize_tag(tag, rules).casefold() for tag in split_tag_text(delete_text)
    }
    kept = [
        tag
        for tag in split_tag_text(tags_text)
        if normalize_tag(tag, rules).casefold() not in delete_set
    ]
    return tag_text(apply_tag_rules(kept, rules), rules.separator)


def replace_tags(tags_text: str, old: str, new: str, rules: TagRuleConfig) -> str:
    old_key = normalize_tag(old, rules).casefold()
    new_value = normalize_tag(new, rules)
    replaced: list[str] = []
    for tag in split_tag_text(tags_text):
        if normalize_tag(tag, rules).casefold() == old_key:
            if new_value:
                replaced.append(new_value)
        else:
            replaced.append(tag)
    return tag_text(apply_tag_rules(replaced, rules), rules.separator)


def add_tags(
    tags_text: str, add_text: str, rules: TagRuleConfig, prepend: bool = False
) -> str:
    current = split_tag_text(tags_text)
    additions = split_tag_text(add_text)
    combined = additions + current if prepend else current + additions
    return tag_text(apply_tag_rules(combined, rules), rules.separator)
