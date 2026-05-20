import base64
import unittest
from pathlib import Path

from app.core.dataset import (
    GENERATED,
    ImageRecord,
    save_record,
    scan_dataset,
    set_generated_nl,
    set_generated_tags,
    table_rows,
    update_record_text,
)


def write_image(path: Path) -> None:
    # 1x1 transparent PNG; enough for dataset scanning tests without Pillow.
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


class DatasetTest(unittest.TestCase):
    def test_scan_dataset_reads_txt_caption(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0001.png"
            write_image(image_path)
            (root / "0001.txt").write_text("1girl, solo. A girl is standing.\n", encoding="utf-8")

            records = scan_dataset(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tags, ["1girl", "solo"])
            self.assertEqual(records[0].nl, "A girl is standing.")

    def test_save_record_writes_txt_and_caption_json(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0001.png"
            write_image(image_path)
            record = scan_dataset(root)[0]
            record.tags = ["1girl", "solo"]
            record.nl = "A girl is standing."

            save_record(record, "caption_json")

            self.assertEqual(
                (root / "0001.txt").read_text(encoding="utf-8").strip(),
                "1girl, solo. A girl is standing.",
            )
            self.assertTrue((root / "caption_json" / "0001.caption.json").exists())

    def test_metadata_follows_edited_tags(self) -> None:
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0001.png"
            write_image(image_path)
            record = scan_dataset(root)[0]
            record.tags = ["solo"]
            record.tag_status = "edited"
            record.tag_details = [{"name": "old tag", "score": 0.99, "source": "model"}]

            save_record(record, "caption_json")

            metadata = json.loads((root / "caption_json" / "0001.caption.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["tags"], [{"name": "solo", "score": None, "source": "manual"}])
            self.assertTrue(metadata["tag_manual"])
            self.assertFalse(metadata["nl_manual"])

    def test_update_record_text_marks_only_changed_section(self) -> None:
        record = ImageRecord(
            image_path="a.png",
            txt_path="a.txt",
            metadata_path="a.caption.json",
            file_name="a.png",
            tags=["1girl"],
            nl="A girl is standing.",
            tag_status=GENERATED,
            nl_status=GENERATED,
        )

        update_record_text(record, "1girl", "A girl is sitting.")

        self.assertEqual(record.tag_status, GENERATED)
        self.assertEqual(record.nl_status, "edited")
        self.assertTrue(record.edited)
        self.assertTrue(record.nl_manual)
        self.assertFalse(record.tag_manual)
        self.assertTrue(record.dirty)
        self.assertEqual(record.overall_status, "未保存")

    def test_model_regeneration_clears_manual_flag_when_no_manual_section_remains(self) -> None:
        record = ImageRecord(
            image_path="a.png",
            txt_path="a.txt",
            metadata_path="a.caption.json",
            file_name="a.png",
            tags=["1girl"],
            nl="A girl is standing.",
            tag_status="edited",
            nl_status="edited",
            edited=True,
        )

        set_generated_tags(record, ["solo"], [{"name": "solo", "score": 0.9, "source": "model"}])
        set_generated_nl(record, "A girl is sitting.")

        self.assertEqual(record.tag_status, GENERATED)
        self.assertEqual(record.nl_status, GENERATED)
        self.assertFalse(record.edited)
        self.assertFalse(record.tag_manual)
        self.assertFalse(record.nl_manual)
        self.assertTrue(record.dirty)

    def test_table_rows_match_ui_columns(self) -> None:
        record = ImageRecord(
            image_path="a.png",
            txt_path="a.txt",
            metadata_path="a.caption.json",
            file_name="a.png",
        )

        rows = table_rows([record])

        self.assertEqual(rows, [["0", "a.png", "未处理", "empty", "empty", "否", ""]])

    def test_reload_preserves_manual_clear_flags_from_metadata(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0001.png"
            write_image(image_path)
            record = scan_dataset(root)[0]
            record.tags = ["solo"]
            record.tag_status = GENERATED

            update_record_text(record, "", "")
            save_record(record, "caption_json")

            reopened = scan_dataset(root)[0]

            self.assertEqual(reopened.tags, [])
            self.assertEqual(reopened.tag_status, "empty")
            self.assertTrue(reopened.edited)
            self.assertTrue(reopened.tag_manual)
            self.assertFalse(reopened.dirty)


if __name__ == "__main__":
    unittest.main()
