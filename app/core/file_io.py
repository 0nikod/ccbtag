from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def txt_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def metadata_path_for_image(image_path: Path, location: str = "caption_json") -> Path:
    if location == "same_folder":
        return image_path.with_suffix(".caption.json")
    return image_path.parent / "caption_json" / f"{image_path.stem}.caption.json"


def metadata_candidates(image_path: Path, preferred_location: str = "caption_json") -> list[Path]:
    preferred = metadata_path_for_image(image_path, preferred_location)
    fallbacks = [
        metadata_path_for_image(image_path, "caption_json"),
        metadata_path_for_image(image_path, "same_folder"),
    ]
    result: list[Path] = []
    for candidate in [preferred, *fallbacks]:
        if candidate not in result:
            result.append(candidate)
    return result
