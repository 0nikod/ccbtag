from dataclasses import replace

import pytest

from app.core.dataset import ImageRecord, update_record_text
from app.services.batch_service import BatchGenerateOptions, BatchService
from app.services.generation_service import GenerationService, NlRequest
from tests.helpers import (
    FakeNlModel,
    FakeRegistry,
    FakeTagModel,
    PredictionStub,
    make_config,
)


def make_records() -> list[ImageRecord]:
    return [
        ImageRecord("1.png", "1.txt", "1.json", "1.png"),
        ImageRecord("2.png", "2.txt", "2.json", "2.png"),
    ]


def make_service(registry: FakeRegistry, *, shuffle_tags: bool = True) -> BatchService:
    base_config = make_config()
    config = make_config(nl=replace(base_config.nl, shuffle_tags=shuffle_tags))
    return BatchService(GenerationService(config, registry))


def test_empty_records_are_safe() -> None:
    result = make_service(FakeRegistry()).generate([], BatchGenerateOptions())

    assert result.total == 0
    assert result.errors == 0


def test_skip_edited_true_skips_manual_record() -> None:
    records = make_records()
    update_record_text(records[0], "manual", "")
    registry = FakeRegistry(tag_model=FakeTagModel([PredictionStub("solo", 0.9)]))

    make_service(registry).generate(
        records,
        BatchGenerateOptions(tag_model_display="PixAI Tagger v0.9", skip_edited=True),
    )

    assert records[0].tags == ["manual"]
    assert records[1].tags == ["solo"]


def test_skip_edited_false_does_not_skip_manual_record() -> None:
    records = make_records()
    update_record_text(records[0], "manual", "")
    registry = FakeRegistry(tag_model=FakeTagModel([PredictionStub("solo", 0.9)]))

    make_service(registry).generate(
        records,
        BatchGenerateOptions(tag_model_display="PixAI Tagger v0.9", skip_edited=False),
    )

    assert records[0].tags == ["solo"]


def test_tag_only_nl_only_and_both_modes_work() -> None:
    tag_registry = FakeRegistry(tag_model=FakeTagModel([PredictionStub("solo", 0.9)]))
    nl_registry = FakeRegistry(nl_model=FakeNlModel(response="generated nl"))
    both_registry = FakeRegistry(
        tag_model=FakeTagModel([PredictionStub("solo", 0.9)]),
        nl_model=FakeNlModel(response="generated nl"),
    )

    tag_records = make_records()
    nl_records = make_records()
    both_records = make_records()

    make_service(tag_registry).generate(
        tag_records,
        BatchGenerateOptions(tag_model_display="PixAI Tagger v0.9"),
    )
    make_service(nl_registry).generate(
        nl_records,
        BatchGenerateOptions(
            nl_model_display="OpenAI Completions",
            nl_request=NlRequest("http://127.0.0.1:8000/v1", "model"),
        ),
    )
    make_service(both_registry).generate(
        both_records,
        BatchGenerateOptions(
            tag_model_display="PixAI Tagger v0.9",
            nl_model_display="OpenAI Completions",
            nl_request=NlRequest("http://127.0.0.1:8000/v1", "model"),
        ),
    )

    assert tag_records[0].tags == ["solo"]
    assert nl_records[0].nl == "generated nl"
    assert both_records[0].tags == ["solo"]
    assert both_records[0].nl == "generated nl"


def test_single_item_failures_do_not_stop_batch_and_errors_are_counted() -> None:
    records = make_records()
    registry = FakeRegistry(
        tag_model=FakeTagModel(error=RuntimeError("tag boom")),
    )

    result = make_service(registry).generate(
        records,
        BatchGenerateOptions(tag_model_display="PixAI Tagger v0.9"),
    )

    assert result.errors == 2
    assert records[0].tag_status == "error"
    assert records[1].tag_status == "error"


def test_tag_success_nl_failure_sets_only_nl_error() -> None:
    records = make_records()
    registry = FakeRegistry(
        tag_model=FakeTagModel([PredictionStub("solo", 0.9)]),
        nl_model=FakeNlModel(error=RuntimeError("nl boom")),
    )

    result = make_service(registry).generate(
        records,
        BatchGenerateOptions(
            tag_model_display="PixAI Tagger v0.9",
            nl_model_display="OpenAI Completions",
            nl_request=NlRequest("http://127.0.0.1:8000/v1", "model"),
        ),
    )

    assert result.errors == 2
    assert records[0].tag_status == "generated"
    assert records[0].nl_status == "error"


def test_missing_nl_request_raises_for_nl_only_and_both() -> None:
    service = make_service(FakeRegistry())

    with pytest.raises(ValueError):
        service.generate(
            make_records(),
            BatchGenerateOptions(nl_model_display="OpenAI Completions"),
        )

    with pytest.raises(ValueError):
        service.generate(
            make_records(),
            BatchGenerateOptions(
                tag_model_display="PixAI Tagger v0.9",
                nl_model_display="OpenAI Completions",
            ),
        )
