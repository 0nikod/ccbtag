from dataclasses import replace

from app.core.dataset import ImageRecord
from app.services.generation_service import GenerationService, NlRequest
from tests.helpers import (
    FakeNlModel,
    FakeRegistry,
    FakeTagModel,
    PredictionStub,
    make_config,
)


def make_record() -> ImageRecord:
    return ImageRecord(
        image_path="sample.png",
        txt_path="sample.txt",
        metadata_path="sample.caption.json",
        file_name="sample.png",
    )


def test_generate_tags_success_updates_record_and_clears_error() -> None:
    service = GenerationService(
        make_config(),
        FakeRegistry(tag_model=FakeTagModel([PredictionStub("solo", 0.9)])),
    )
    record = make_record()
    record.error = "old"

    result = service.generate_tags(record, "PixAI Tagger v0.9")

    assert result.ok is True
    assert record.tags == ["solo"]
    assert record.error == ""


def test_generate_tags_failure_sets_tag_error() -> None:
    service = GenerationService(
        make_config(),
        FakeRegistry(tag_model=FakeTagModel(error=RuntimeError("boom"))),
    )
    record = make_record()

    result = service.generate_tags(record, "PixAI Tagger v0.9")

    assert result.ok is False
    assert record.tag_status == "error"
    assert record.error == "boom"


def test_generate_nl_success_updates_record_and_clears_error() -> None:
    service = GenerationService(
        make_config(),
        FakeRegistry(nl_model=FakeNlModel(response="A girl is standing.")),
    )
    record = make_record()
    record.tags = ["solo"]
    record.error = "old"

    result = service.generate_nl(
        record,
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model"),
    )

    assert result.ok is True
    assert record.nl == "A girl is standing."
    assert record.error == ""


def test_generate_nl_failure_sets_nl_error() -> None:
    service = GenerationService(
        make_config(),
        FakeRegistry(nl_model=FakeNlModel(error=RuntimeError("nl boom"))),
    )
    record = make_record()

    result = service.generate_nl(
        record,
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model"),
    )

    assert result.ok is False
    assert record.nl_status == "error"
    assert record.error == "nl boom"


def test_shuffle_tags_none_uses_config_default() -> None:
    registry = FakeRegistry(nl_model=FakeNlModel())
    base_config = make_config()
    config = make_config(nl=replace(base_config.nl, shuffle_tags=False))
    record = make_record()

    GenerationService(config, registry).generate_nl(
        record,
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model", shuffle_tags=None),
    )

    assert registry.nl_model.calls[-1]["shuffle_tags"] is False


def test_shuffle_tags_request_can_override_config_default() -> None:
    registry = FakeRegistry(nl_model=FakeNlModel())
    base_config = make_config()
    config = make_config(nl=replace(base_config.nl, shuffle_tags=True))
    record = make_record()

    GenerationService(config, registry).generate_nl(
        record,
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model", shuffle_tags=False),
    )

    assert registry.nl_model.calls[-1]["shuffle_tags"] is False


def test_image_resize_mode_is_passed_to_nl_model() -> None:
    registry = FakeRegistry(nl_model=FakeNlModel())
    config = make_config()
    record = make_record()

    GenerationService(config, registry).generate_nl(
        record,
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model", image_resize_mode="1MP"),
    )

    assert registry.nl_model.calls[-1]["image_resize_mode"] == "1MP"


def test_generate_both_stops_when_tag_generation_fails() -> None:
    registry = FakeRegistry(
        tag_model=FakeTagModel(error=RuntimeError("tag boom")),
        nl_model=FakeNlModel(),
    )
    record = make_record()

    result = GenerationService(make_config(), registry).generate_both(
        record,
        "PixAI Tagger v0.9",
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model"),
    )

    assert result.ok is False
    assert registry.nl_model.calls == []


def test_generate_both_keeps_generated_tags_when_nl_fails() -> None:
    registry = FakeRegistry(
        tag_model=FakeTagModel([PredictionStub("solo", 0.9)]),
        nl_model=FakeNlModel(error=RuntimeError("nl boom")),
    )
    record = make_record()

    result = GenerationService(make_config(), registry).generate_both(
        record,
        "PixAI Tagger v0.9",
        "OpenAI Completions",
        NlRequest("http://127.0.0.1:8000/v1", "model"),
    )

    assert result.ok is False
    assert record.tags == ["solo"]
    assert record.tag_status == "generated"
    assert record.nl_status == "error"
