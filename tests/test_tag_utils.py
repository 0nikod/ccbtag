import unittest

from app.core.tag_utils import (
    TagRuleConfig,
    add_tags,
    delete_tags,
    prediction_dicts_to_tags,
    replace_tags,
)


def rules() -> TagRuleConfig:
    return TagRuleConfig(
        threshold=0.35,
        max_tags=3,
        replace_underscore=True,
        sort_by_score=True,
        blacklist=("watermark",),
        replace_rules={"grey hair": "gray hair"},
    )


class TagUtilsTest(unittest.TestCase):
    def test_prediction_dicts_to_tags_filters_sorts_and_cleans(self) -> None:
        predictions = [
            {"tag": "watermark", "score": 0.99},
            {"tag": "grey_hair", "score": 0.8},
            {"tag": "solo", "score": 0.7},
            {"tag": "low score", "score": 0.1},
        ]
        self.assertEqual(
            prediction_dicts_to_tags(predictions, rules()), ["gray hair", "solo"]
        )

    def test_delete_replace_and_add_tags(self) -> None:
        self.assertEqual(
            delete_tags("1girl, solo, watermark", "solo", rules()), "1girl"
        )
        self.assertEqual(
            replace_tags("grey hair, solo", "grey hair", "gray hair", rules()),
            "gray hair, solo",
        )
        self.assertEqual(
            add_tags("solo", "1girl", rules(), prepend=True), "1girl, solo"
        )


if __name__ == "__main__":
    unittest.main()
