import base64
from pathlib import Path

from app.core.preview import image_preview_html


def write_image(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


def test_image_preview_uses_data_url_without_file_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    write_image(image_path)

    html = image_preview_html(image_path)

    assert "data:image/" in html
    assert str(image_path) not in html
