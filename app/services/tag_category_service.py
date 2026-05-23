from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Mapping


TAG_CATEGORY_IDS = {
    "general": 0,
    "artist": 1,
    "copyright": 3,
    "character": 4,
    "meta": 5,
}
DEFAULT_KEPT_TAG_CATEGORIES = ("general",)
DEFAULT_TAG_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "tags_processed.sqlite"


def tag_category_choices() -> list[str]:
    return list(TAG_CATEGORY_IDS)


def normalize_kept_tag_categories(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in TAG_CATEGORY_IDS:
            raise ValueError(f"未知标签类别: {value}")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


class TagCategoryService:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path or DEFAULT_TAG_DB_PATH).expanduser()
        self._categories_by_key: dict[str, int] | None = None

    def filter_predictions(
        self,
        predictions: Iterable[Mapping[str, object]],
        kept_categories: Iterable[str] | None,
    ) -> list[dict[str, object]]:
        normalized_categories = normalize_kept_tag_categories(kept_categories)
        if normalized_categories is None:
            return [dict(prediction) for prediction in predictions]
        keep_ids = {TAG_CATEGORY_IDS[name] for name in normalized_categories}
        filtered: list[dict[str, object]] = []
        for prediction in predictions:
            row = dict(prediction)
            tag = str(row.get("tag") or row.get("name") or "").strip()
            if not tag:
                continue
            category = self.category_for_tag(tag)
            if category in keep_ids:
                filtered.append(row)
        return filtered

    def category_for_tag(self, tag: str) -> int | None:
        categories_by_key = self._load_categories()
        for key in _lookup_keys(tag):
            category = categories_by_key.get(key)
            if category is not None:
                return category
        return None

    def _load_categories(self) -> dict[str, int]:
        if self._categories_by_key is not None:
            return self._categories_by_key
        if not self.database_path.exists():
            raise FileNotFoundError(f"标签类别数据库不存在: {self.database_path}")
        categories_by_key: dict[str, int] = {}
        connection = sqlite3.connect(self.database_path)
        try:
            cursor = connection.execute("SELECT name, alias, category FROM tags")
            for name, alias, category in cursor:
                if name is not None:
                    categories_by_key.setdefault(str(name).strip().casefold(), int(category))
                if alias:
                    for item in str(alias).split(","):
                        key = item.strip().casefold()
                        if key:
                            categories_by_key.setdefault(key, int(category))
        finally:
            connection.close()
        self._categories_by_key = categories_by_key
        return categories_by_key


def _lookup_keys(tag: str) -> tuple[str, ...]:
    normalized = str(tag).strip().casefold()
    if not normalized:
        return ()
    keys = [normalized]
    underscored = normalized.replace(" ", "_")
    spaced = normalized.replace("_", " ")
    if underscored not in keys:
        keys.append(underscored)
    if spaced not in keys:
        keys.append(spaced)
    return tuple(keys)
