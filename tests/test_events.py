from app.ui import events


def test_open_folder_failure_matches_open_outputs() -> None:
    payload = events.open_folder("/path/that/does/not/exist", "caption_json")

    assert len(payload) == 8
    assert payload[-1].startswith("打开失败:")


def test_empty_batch_actions_match_open_outputs() -> None:
    assert len(events.batch_generate_tags([], "PixAI Tagger v0.9", True)) == 8
    assert len(events.batch_delete_tag([], "solo")) == 8
