from app.core.dataset import ImageRecord, update_record_text
from app.services.tag_edit_service import TagEditService
from tests.helpers import make_config


def make_records() -> list[ImageRecord]:
    first = ImageRecord("1.png", "1.txt", "1.json", "1.png")
    second = ImageRecord("2.png", "2.txt", "2.json", "2.png")
    update_record_text(first, "solo, smile", "")
    update_record_text(second, "solo", "")
    first.dirty = False
    second.dirty = False
    return [first, second]


def test_delete_replace_and_add_update_all_records() -> None:
    service = TagEditService(make_config().tag)

    delete_result = service.delete_tags(make_records(), "solo")
    replace_records = make_records()
    replace_result = service.replace_tags(replace_records, "smile", "happy")
    add_records = make_records()
    add_result = service.add_tags(add_records, "portrait", prepend=False)
    prepend_records = make_records()
    service.add_tags(prepend_records, "portrait", prepend=True)

    assert delete_result.changed == 2
    assert replace_result.changed == 1
    assert add_result.changed == 2
    assert replace_records[0].tags == ["solo", "happy"]
    assert add_records[0].tags[-1] == "portrait"
    assert prepend_records[0].tags[0] == "portrait"


def test_tag_edits_mark_records_dirty_and_edited() -> None:
    records = make_records()
    for record in records:
        record.dirty = False
    result = TagEditService(make_config().tag).delete_tags(records, "solo")

    assert result.changed == 2
    assert all(record.dirty is True for record in records)
    assert all(record.edited is True for record in records)
