from __future__ import annotations

import base64
import html
import io
import mimetypes
from pathlib import Path

from PIL import Image


def image_preview_html(image_path: str | Path | None, max_size: int = 1200) -> str:
    """Build an in-memory image preview without copying files to Gradio or /tmp.

    The dataset image path remains the source of truth. This function only reads
    the image and returns a data URL for browser preview.
    """

    if not image_path:
        return _empty_preview("未选择图片")
    path = Path(image_path)
    if not path.exists():
        return _empty_preview(f"图片不存在: {path}")
    try:
        data_url, dimensions = _thumbnail_data_url(path, max_size)
    except Exception:
        data_url = _raw_data_url(path)
        dimensions = "尺寸未知"
    escaped_name = html.escape(path.name)
    escaped_dimensions = html.escape(dimensions)
    return (
        "<figure style='height:min(68vh,720px);min-height:520px;margin:0;display:flex;flex-direction:column;"
        "background:#f7f4ee;border:1px solid #ded6c9;border-radius:8px;overflow:hidden;'>"
        "<figcaption style='display:flex;justify-content:space-between;gap:12px;padding:10px 12px;"
        "font-size:13px;color:#62584d;border-bottom:1px solid #e5ded3;'>"
        f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{escaped_name}</span>"
        f"<span>{escaped_dimensions}</span>"
        "</figcaption>"
        "<div style='flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:10px;'>"
        f"<img alt='{escaped_name}' src='{data_url}' style='max-width:100%;max-height:100%;object-fit:contain;' />"
        "</div>"
        "</figure>"
    )


def _thumbnail_data_url(path: Path, max_size: int) -> tuple[str, str]:
    with Image.open(path) as image:
        dimensions = f"{image.width} x {image.height}"
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA")
        buffer = io.BytesIO()
        image_format = "PNG" if image.mode == "RGBA" else "JPEG"
        mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
        image.save(buffer, format=image_format, quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", dimensions


def _raw_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _empty_preview(message: str) -> str:
    escaped = html.escape(message)
    return (
        "<div style='height:min(68vh,720px);min-height:520px;display:flex;align-items:center;justify-content:center;"
        "background:#f7f4ee;border:1px dashed #b8aa98;border-radius:8px;color:#6f6255;'>"
        f"{escaped}</div>"
    )
