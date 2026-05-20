import base64
import unittest
from pathlib import Path

from app.core.dataset import scan_dataset, save_record


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


if __name__ == "__main__":
    unittest.main()
