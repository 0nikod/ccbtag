from pathlib import Path
import sqlite3

from app.services.tag_category_service import TagCategoryService


def write_tag_db(path: Path) -> TagCategoryService:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE tags (id INTEGER, name TEXT, alias TEXT, post_count INTEGER, category INTEGER, is_deprecated INTEGER)"
        )
        connection.executemany(
            "INSERT INTO tags (id, name, alias, post_count, category, is_deprecated) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "blue_eyes", "blue eyes", 1, 0, 0),
                (2, "kantoku", "", 1, 1, 0),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return TagCategoryService(path)


def test_category_lookup_handles_underscore_and_alias(tmp_path: Path) -> None:
    service = write_tag_db(tmp_path / "tags.sqlite")

    assert service.category_for_tag("blue_eyes") == 0
    assert service.category_for_tag("blue eyes") == 0


def test_filter_predictions_keeps_only_requested_categories(tmp_path: Path) -> None:
    service = write_tag_db(tmp_path / "tags.sqlite")

    filtered = service.filter_predictions(
        [
            {"tag": "blue_eyes", "score": 0.9},
            {"tag": "kantoku", "score": 0.8},
            {"tag": "unknown_tag", "score": 0.7},
        ],
        ("general",),
    )

    assert [item["tag"] for item in filtered] == ["blue_eyes"]
