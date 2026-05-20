import unittest

from app.core.caption import join_caption, parse_caption_text, split_tag_text


class CaptionTest(unittest.TestCase):
    def test_join_caption_with_tags_and_nl(self) -> None:
        self.assertEqual(
            join_caption(["1girl", "solo", "long hair"], "A girl is standing."),
            "1girl, solo, long hair. A girl is standing.",
        )

    def test_join_caption_keeps_single_section(self) -> None:
        self.assertEqual(join_caption(["1girl", "solo"], ""), "1girl, solo")
        self.assertEqual(join_caption([], "A girl is standing."), "A girl is standing.")

    def test_parse_caption_text_splits_first_period(self) -> None:
        parts = parse_caption_text("1girl, solo. A girl is standing.")
        self.assertEqual(parts.tags, ["1girl", "solo"])
        self.assertEqual(parts.nl, "A girl is standing.")

    def test_split_tag_text_ignores_empty_items(self) -> None:
        self.assertEqual(split_tag_text("1girl, , solo,"), ["1girl", "solo"])


if __name__ == "__main__":
    unittest.main()
