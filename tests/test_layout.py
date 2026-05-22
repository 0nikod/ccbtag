from app.core.dataset import ImageRecord, table_rows
from app.ui.layout import TABLE_HEADERS


def test_table_headers_match_table_rows() -> None:
    rows = table_rows(
        [
            ImageRecord(
                image_path="sample.png",
                txt_path="sample.txt",
                metadata_path="sample.caption.json",
                file_name="sample.png",
            )
        ]
    )

    assert len(TABLE_HEADERS) == len(rows[0])
