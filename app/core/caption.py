from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_SEPARATOR = ", "
DEFAULT_JOINER = ". "
DEFAULT_BATCH_TXT_SENTENCE_KEYWORDS = (
    "art style",
    "tags include",
)
DEFAULT_BATCH_TXT_FRAGMENT_KEYWORDS = (
    "atmosphere",
    "vibe",
    "art style",
    "artist style",
    "rendering style",
    "aesthetic genre",
)


@dataclass(frozen=True)
class CaptionParts:
    """Structured caption sections used by the editor and metadata file."""

    tags: list[str]
    nl: str


def split_tag_text(value: str | Iterable[str] | None) -> list[str]:
    """Convert comma-separated tag text or an iterable into normalized tags.

    The editor keeps tags as a list internally. Users still edit a simple text
    box, so this function is the boundary between UI text and structured data.
    """

    if value is None:
        return []
    if isinstance(value, str):
        raw_tags = value.split(",")
    else:
        raw_tags = list(value)
    return [tag.strip() for tag in raw_tags if str(tag).strip()]


def tag_text(tags: Iterable[str], separator: str = DEFAULT_SEPARATOR) -> str:
    """Serialize tags for display and `.txt` export."""

    return separator.join(split_tag_text(tags))


def clean_nl_text(value: str | None) -> str:
    """Normalize natural-language text without changing user wording."""

    return " ".join((value or "").strip().split())


def clean_nl_by_rules(
    value: str | None,
    sentence_keywords: Iterable[str] | None = None,
    fragment_keywords: Iterable[str] | None = None,
) -> str:
    """Apply batch txt cleanup rules to the NL portion only."""

    sentences = _split_sentences(value)
    sentence_terms = _normalized_keywords(
        sentence_keywords or DEFAULT_BATCH_TXT_SENTENCE_KEYWORDS
    )
    fragment_terms = _normalized_keywords(
        fragment_keywords or DEFAULT_BATCH_TXT_FRAGMENT_KEYWORDS
    )

    filtered_sentences = [
        sentence
        for sentence in sentences
        if not _contains_keyword(sentence, sentence_terms)
    ]
    cleaned_sentences = [
        _remove_keyword_fragments(sentence, fragment_terms)
        for sentence in filtered_sentences
    ]
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]

    if cleaned_sentences and _contains_keyword(cleaned_sentences[-1], fragment_terms):
        cleaned_sentences.pop()
    return _join_sentences(cleaned_sentences)


def join_caption(
    tags: Iterable[str] | str | None,
    nl: str | None,
    separator: str = DEFAULT_SEPARATOR,
    joiner: str = DEFAULT_JOINER,
) -> str:
    """Join tags and natural language using the fixed `tag -> nl` order.

    If either section is empty, the non-empty section is returned unchanged.
    A tag-only caption intentionally has no trailing period because many LoRA
    datasets expect raw comma-separated tags.
    """

    tags_text = tag_text(split_tag_text(tags), separator).rstrip(" .")
    nl_text = clean_nl_text(nl)
    if tags_text and nl_text:
        return f"{tags_text}{joiner}{nl_text}"
    if tags_text:
        return tags_text
    return nl_text


def parse_caption_text(text: str | None) -> CaptionParts:
    """Best-effort parser for legacy `.txt` captions.

    Structured metadata is preferred. This fallback uses the first period as the
    split point because the planned export format is `tags. natural language`.
    The result is deliberately editable rather than treated as authoritative.
    """

    normalized = (text or "").strip()
    if not normalized:
        return CaptionParts(tags=[], nl="")
    if "." not in normalized:
        if "," in normalized or " " not in normalized:
            return CaptionParts(tags=split_tag_text(normalized), nl="")
        return CaptionParts(tags=[], nl=clean_nl_text(normalized))
    tag_part, nl_part = normalized.split(".", 1)
    if _looks_like_tag_block(tag_part):
        return CaptionParts(tags=split_tag_text(tag_part), nl=clean_nl_text(nl_part))
    return CaptionParts(tags=[], nl=clean_nl_text(normalized))


def _looks_like_tag_block(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    return "," in normalized or " " not in normalized


def _split_sentences(value: str | None) -> list[str]:
    return [clean_nl_text(part) for part in (value or "").split(".") if part.strip()]


def _join_sentences(sentences: Iterable[str]) -> str:
    cleaned = [clean_nl_text(sentence) for sentence in sentences if sentence.strip()]
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."


def _normalized_keywords(keywords: Iterable[str]) -> tuple[str, ...]:
    return tuple(keyword.strip().lower() for keyword in keywords if keyword.strip())


def _contains_keyword(value: str, keywords: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in keywords)


def _remove_keyword_fragments(sentence: str, keywords: Iterable[str]) -> str:
    parts = [part.strip() for part in sentence.split(",")]
    if not parts:
        return ""

    kept_parts = [parts[0]] if parts[0] else []
    for fragment in parts[1:]:
        if fragment and not _contains_keyword(fragment, keywords):
            kept_parts.append(fragment)
    return ", ".join(part for part in kept_parts if part)
