import unittest

from app.core.caption import (
    clean_nl_by_rules,
    join_caption,
    parse_caption_text,
    split_tag_text,
)


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

    def test_parse_caption_text_keeps_sentence_as_nl(self) -> None:
        parts = parse_caption_text("A girl is standing.")
        self.assertEqual(parts.tags, [])
        self.assertEqual(parts.nl, "A girl is standing.")

    def test_split_tag_text_ignores_empty_items(self) -> None:
        self.assertEqual(split_tag_text("1girl, , solo,"), ["1girl", "solo"])

    def test_clean_nl_by_rules_removes_matching_sentences_case_insensitively(self) -> None:
        self.assertEqual(
            clean_nl_by_rules("A girl stands. ART STYLE is anime. Tags Include solo."),
            "A girl stands.",
        )

    def test_clean_nl_by_rules_removes_fragments_and_last_sentence(self) -> None:
        self.assertEqual(
            clean_nl_by_rules(
                "A girl stands, warm vibe, smiling. Soft atmosphere."
            ),
            "A girl stands, smiling.",
        )


if __name__ == "__main__":
    unittest.main()
